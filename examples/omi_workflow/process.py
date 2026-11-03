"""GPU registration/integration with auditable frame selection and quicklooks.

Defaults to a 4x4 review product. Set binning=1 for native-resolution masters.
No outside imagery, generative restoration, or local colour patch mapping.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import astroalign
import numpy as np
import sep
import tifffile
from astropy.io import fits
from PIL import Image
from scipy.ndimage import gaussian_filter

from alcor_imaging import StackConfig, stack, warp_affine
from archive import atomic_json
from quality import bin_mean, display_mono


def catalog(image):
    a=np.ascontiguousarray(np.nan_to_num(image),dtype=np.float32)
    bg=sep.Background(a,bw=64,bh=64)
    obj=sep.extract(a-bg.back(),5,err=bg.rms(),minarea=5)
    obj=obj[(obj['a']>0)&(obj['b']>0)&(obj['b']/np.maximum(obj['a'],1e-6)>0.45)&
            (obj['npix']<400)&(obj['flux']/np.maximum(obj['peak'],1)>3)]
    obj=obj[np.argsort(obj['flux'])[::-1]][:100]
    return np.column_stack([obj['x'],obj['y']])


def transform(source, reference):
    if len(source)<12 or len(reference)<12:
        raise ValueError('Fewer than 12 stars for registration')
    t,(s,d)=astroalign.find_transform(source,reference,max_control_points=80)
    rms=float(np.sqrt(np.mean(np.sum((t(s)-d)**2,axis=1))))
    if not 0.98<t.scale<1.02 or rms>0.8 or len(s)<12:
        raise ValueError(f'Registration failed validation: matches={len(s)}, RMS={rms:.3f}, scale={t.scale:.4f}')
    return t.params,{'matches':len(s),'rms_binned_px':rms,'scale':float(t.scale)}


def rebin_matrix(matrix, source_bin=4, dest_bin=1):
    factor=source_bin/dest_bin
    s=np.array([[factor,0,(factor-1)/2],[0,factor,(factor-1)/2],[0,0,1.]])
    return s@matrix@np.linalg.inv(s)


def compatible_key(r):
    return (tuple(r['shape']),r.get('camera'),r.get('gain'),r.get('offset'),r.get('bin_x'),r.get('bin_y'))


def process_target(root,target,night=None,binning=4,backend='gpu',include_review=False):
    root=Path(root)
    rows=json.loads((root/'quality'/'frame_report.json').read_text())
    overrides=json.loads((root/'quality'/'overrides.json').read_text()) if (root/'quality'/'overrides.json').exists() else {}
    selected=[]
    for r in rows:
        if r.get('object')!=target or (night and r.get('night')!=night):continue
        status=overrides.get(r['path'],{}).get('status',r['status'])
        if status not in ({'candidate','review','accepted'} if include_review else {'candidate','accepted'}):continue
        if str(r.get('image_type','')).upper() not in {'LIGHT','LIGHT FRAME'}:continue
        if float(r.get('exposure') or 0)<1:continue
        selected.append(r)
    if not selected:raise ValueError('No accepted light frames for this target/night')
    if len({compatible_key(r) for r in selected})!=1:
        raise ValueError('Incompatible detector/gain/geometry groups: process them separately')
    if binning not in {1,2,4}:raise ValueError('binning must be 1, 2 or 4')
    label=re.sub(r'[^A-Za-z0-9_-]+','_',target).strip('_')
    run_hash=hashlib.sha256(json.dumps({'paths':sorted(r['path'] for r in selected),'binning':binning,'version':1},sort_keys=True).encode()).hexdigest()[:10]
    out=root/'processed'/label/(str(night or 'compatible_nights')+f'_bin{binning}_{run_hash}')
    out.mkdir(parents=True,exist_ok=True)
    def cache(r):return np.load(root/'quality'/'cache'/(r['id']+'.npy'))
    ref=min(selected,key=lambda r:(r.get('filter') not in {'Ha','L'},r.get('fwhm_px') or 100))
    ref_small=cache(ref);ref_cat=catalog(ref_small)
    grouped=defaultdict(list);registration=[]
    for r in sorted(selected,key=lambda r:r['path']):
        try:
            small=cache(r)
            matrix,stats=(np.eye(3),{'matches':len(ref_cat),'rms_binned_px':0.,'scale':1.}) if r['id']==ref['id'] else transform(catalog(small),ref_cat)
            rec={'path':r['path'],'filter':r.get('filter'),'status':'registered','matrix_bin4':matrix.tolist(),**stats}
            grouped[r['filter']].append((r,matrix))
        except Exception as e:
            rec={'path':r['path'],'status':'registration_rejected','reason':str(e)}
        registration.append(rec);print(rec['status'],r['path'],flush=True)
        atomic_json(out/'registration.json',registration)
    masters={};integration={}
    h,w=ref['shape'];shape=(h//binning,w//binning)
    for filt,items in grouped.items():
        if len(items)<3:
            integration[filt]={'status':'insufficient_frames','frames':len(items)};continue
        aligned=[];weights=[];seconds=0
        for r,matrix in items:
            if binning==4:a=cache(r)
            else:
                with fits.open(root/'raw'/r['path'],memmap=False) as hdus:
                    a=np.asarray(next(x.data for x in hdus if int(x.header.get('NAXIS',0))==2),dtype=np.float32)
                if binning!=1:a=bin_mean(a,binning)
            # Only a scalar pedestal. Do not fit away extended nebulae automatically.
            a=(a-float(r['background_adu']))/float(r['exposure'])
            aligned.append(warp_affine(a,rebin_matrix(matrix,4,binning),shape,backend=backend))
            weights.append((float(r['exposure'])/max(float(r['noise_adu']),1))**2)
            seconds+=float(r['exposure'])
        weights=np.asarray(weights);weights=np.clip(weights/np.median(weights),0.25,4)
        master=stack(aligned,StackConfig(method='sigma_clip_mean',sigma=3,max_iterations=3,tile_size=384,backend=backend),weights=weights)
        del aligned
        masters[filt]=master
        head=fits.Header();head['OBJECT']=target;head['FILTER']=filt;head['EXPTIME']=seconds
        head['NCOMBINE']=len(items);head['SOFTBIN']=binning;head['BUNIT']='ADU/s';head['CALSTAT']='LIGHTS_ONLY'
        head['HISTORY']='Scalar background removed; registered and sigma-clipped. No dark/flat correction.'
        fits.PrimaryHDU(master,header=head).writeto(out/(f'master_{filt}_bin{binning}.fits'),overwrite=True)
        display_mono(master,(1800,1800)).save(out/(f'master_{filt}_preview.png'))
        integration[filt]={'status':'integrated','frames':len(items),'seconds':seconds,'weights':weights.tolist()}
        print('MASTER',filt,len(items),seconds,flush=True)
    palette=None
    if all(k in masters for k in ['SII','Ha','OIII']):channels=['SII','Ha','OIII'];palette='SHO mapped colour'
    elif all(k in masters for k in ['Ha','OIII']):channels=['Ha','OIII','OIII'];palette='HOO mapped colour'
    elif all(k in masters for k in ['R','G','B']):channels=['R','G','B'];palette='RGB approximate colour (not photometrically calibrated)'
    elif masters:
        k='Ha' if 'Ha' in masters else next(iter(masters));channels=[k,k,k];palette=k+' monochrome'
    else:raise ValueError('No channel has at least three registered frames')
    rgb=np.stack([masters[k] for k in channels],axis=-1)
    valid=np.all(np.isfinite(rgb),axis=-1)
    # Bounding crop, with invalid edge coverage retained in a separate mask.
    yy,xx=np.where(valid)
    if not len(yy):raise ValueError('No common registered coverage')
    y0,y1=int(yy.min()),int(yy.max()+1);x0,x1=int(xx.min()),int(xx.max()+1)
    rgb=np.nan_to_num(rgb[y0:y1,x0:x1]);valid=valid[y0:y1,x0:x1]
    gains=[]
    for c in range(3):
        vals=rgb[...,c][valid];span=max(float(np.percentile(vals,99.5)),1e-6)
        gains.append(1/span)
    # Global display gains, recorded explicitly; no local hue replacement.
    rgb*=np.asarray(gains,dtype=np.float32)
    rgb=gaussian_filter(rgb,sigma=(max(0.6,3/binning),max(0.6,3/binning),0))
    lum=np.maximum(np.mean(rgb,axis=-1),1e-6)
    target_lum=np.arcsinh(8*lum)/np.arcsinh(8)
    rendered=rgb*(target_lum/lum)[...,None]
    rendered=rendered/np.maximum(1,np.max(rendered,axis=-1,keepdims=True))
    rendered=np.clip(rendered+0.008,0,1);rendered[~valid]=0
    Image.fromarray(np.round(rendered*255).astype('uint8')).save(out/'colour_quicklook.png')
    tifffile.imwrite(out/'colour_quicklook_16bit.tif',np.round(rendered*65535).astype('uint16'),photometric='rgb')
    fits.PrimaryHDU(valid.astype('uint8')).writeto(out/'common_coverage.fits',overwrite=True)
    report={'target':target,'night':night,'binning':binning,'backend':backend,'palette':palette,
            'reference':ref['path'],'channels':integration,'display_gains':gains,'crop_xy':[x0,y0,x1,y1],
            'limitations':['Lights only: no matching darks/flats supplied.','Quicklook uses global display balance, not measured filter transmission.',
                          'Scalar sky offsets only; gradients/halos may remain.','Candidate is a screening result, not guaranteed production suitability.'],
            'selection':{'include_review':include_review,'overrides':overrides},'output':str(out)}
    atomic_json(out/'processing_report.json',report)
    return report
