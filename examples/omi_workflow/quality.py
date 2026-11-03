"""Conservative FITS screening and review sheets, run on cloud compute.

No frame is discarded. Thresholds are screening heuristics, not a claim that
Moon illumination alone establishes image usability. Elongated detections are
included when measuring tracking: filtering them first hides failed guiding.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import sep
from astropy.io import fits
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d, median_filter

from archive import atomic_json, validate_fits

QC_VERSION = 4


def bin_mean(a, factor=4):
    h, w = a.shape
    return a[:h//factor*factor, :w//factor*factor].reshape(
        h//factor, factor, w//factor, factor).mean((1, 3), dtype=np.float32)


def mad(a):
    a = np.asarray(a)
    return float(1.4826 * np.nanmedian(np.abs(a - np.nanmedian(a))))


def display_mono(a, size=None):
    a = np.nan_to_num(a.astype(np.float32))
    lo, hi = np.percentile(a, [10, 99.7])
    scaled = np.clip((a-lo)/max(hi-lo, 1e-6), 0, None)
    scaled = np.clip(np.arcsinh(12*scaled)/np.arcsinh(12), 0, 1)
    im = Image.fromarray(np.round(scaled*255).astype('uint8'))
    if size:
        im.thumbnail(size, Image.Resampling.LANCZOS)
    return im


def star_measurements(data):
    """Native-pixel samples across the sensor; no elongation cut here."""
    h, w = data.shape; rows = []; crops = []; noise = []
    for fy, fx in [(0.5,0.5),(0.22,0.22),(0.22,0.78),(0.78,0.22),(0.78,0.78)]:
        y = max(0, min(h-512, int(h*fy)-256)); x = max(0, min(w-512, int(w*fx)-256))
        cut = np.ascontiguousarray(data[y:y+512,x:x+512],dtype=np.float32)
        noise.append(mad(cut[:,1:]-cut[:,:-1])/np.sqrt(2))
        bg = sep.Background(cut, bw=64,bh=64)
        obj = sep.extract(cut-bg.back(), 7, err=bg.rms(), minarea=6)
        obj = obj[(obj['a']>1.1)&(obj['b']>0.8)&(obj['npix']<700)&
                  (obj['flux']/np.maximum(obj['peak'],1)>5)&
                  (obj['peak']+float(bg.globalback)<60000)]
        rows.extend([(float(o['a']),float(o['b']),float(o['theta'])) for o in obj])
        if len(obj):
            o = obj[np.argmax(obj['flux'])]; sx,sy=int(o['x']),int(o['y'])
            sx=max(40,min(cut.shape[1]-40,sx));sy=max(40,min(cut.shape[0]-40,sy))
            crops.append(cut[sy-40:sy+40,sx-40:sx+40])
        else:
            crops.append(cut[216:296,216:296])
    if rows:
        a,b,theta=np.asarray(rows).T; ratio=b/a
        summary={"sample_stars":len(rows),"axis_ratio":float(np.median(ratio)),
                 "trail_fraction":float(np.mean(ratio<0.5)),
                 "fwhm_px":float(np.median(2.355*np.sqrt(a*b))),
                 "trail_coherence":float(abs(np.mean(np.exp(2j*theta))))}
    else:
        summary={"sample_stars":0,"axis_ratio":None,"trail_fraction":None,
                 "fwhm_px":None,"trail_coherence":None}
    summary['noise_adu']=float(np.median(noise))
    return summary,crops


def refine_field_metrics(result,small):
    """Wide-field morphology catches long arcs missed by sparse native crops."""
    a=np.ascontiguousarray(small,dtype=np.float32)
    bg=sep.Background(a,bw=64,bh=64)
    obj=sep.extract(a-bg.back(),5,err=bg.rms(),minarea=5)
    obj=obj[(obj['a']>0.65)&(obj['b']>0.45)&(obj['npix']<2000)&
            (obj['flux']/np.maximum(obj['peak'],1)>3)]
    result['field_stars']=len(obj)
    if len(obj):
        ratio=obj['b']/obj['a']
        result['field_axis_ratio']=float(np.median(ratio))
        result['field_trail_fraction']=float(np.mean(ratio<0.5))
        result['field_fwhm_native_px']=float(np.median(2.355*np.sqrt(obj['a']*obj['b']))*4)
        if result['status']!='planetary_or_test' and len(obj)>=20:
            if np.median(ratio)<0.45 and np.mean(ratio<0.5)>0.65:
                result['status']='reject'
                result['reasons'].append('severe field-wide stellar trails/arcs')
            elif np.median(ratio)<0.65 and result['status']=='candidate':
                result['status']='review';result['reasons'].append('field-wide stellar elongation')
            elif (len(obj)>=30 and np.median(ratio)>=0.75 and
                  result['reasons']==['too few stellar detections for reliable tracking assessment']):
                # Sparse native crops need not veto a well-measured full field.
                result['status']='candidate';result['reasons']=[]
    result['row_jump_noise_ratio']=float(result.get('row_jump_adu',0))/max(float(result.get('noise_adu',1)),1e-6)
    if result['row_jump_noise_ratio']>3 and result['status'] not in {'reject','planetary_or_test'}:
        result['status']='review';result['reasons'].append('abrupt full-width row discontinuity')
    result['reasons']=list(dict.fromkeys(result['reasons']))
    result['qc_version']=QC_VERSION
    return result


def upgrade_cached(result,root):
    root=Path(root)
    cache=root/'quality'/'cache'/(result['id']+'.npy')
    if not cache.exists():return inspect_one(root/'raw'/result['path'],root)
    result=refine_field_metrics(result,np.load(cache))
    atomic_json(root/'quality'/'frames'/(result['id']+'.json'),result)
    return result


def inspect_one(path, root):
    root=Path(root);path=Path(path);rel=path.relative_to(root/'raw')
    key=hashlib.sha256(str(rel).encode()).hexdigest()[:20]
    out=root/'quality'/'frames';out.mkdir(parents=True,exist_ok=True)
    result={"path":str(rel),"id":key,"qc_version":QC_VERSION,
            "night":rel.parts[0] if re.fullmatch(r'\d{4}-\d{2}-\d{2}',rel.parts[0]) else 'root'}
    try:
        validate_fits(path)
        with fits.open(path,memmap=False) as hdus:
            hdu=next(h for h in hdus if int(h.header.get('NAXIS',0))==2)
            header=hdu.header.copy(); data=np.asarray(hdu.data,dtype=np.float32)
        for keyname,kw in [('object','OBJECT'),('filter','FILTER'),('exposure','EXPTIME'),
                           ('date_obs','DATE-OBS'),('camera','INSTRUME'),('gain','GAIN'),
                           ('offset','OFFSET'),('temperature','CCD-TEMP'),('bin_x','XBINNING'),
                           ('bin_y','YBINNING'),('ra','RA'),('dec','DEC'),('image_type','IMAGETYP')]:
            result[keyname]=header.get(kw)
        result['shape']=list(data.shape)
        finite=np.isfinite(data);result['nonfinite_fraction']=float(1-np.mean(finite))
        if result['nonfinite_fraction']>0.01 or not np.any(finite):
            raise ValueError('Invalid/nonfinite pixel payload')
        data=np.nan_to_num(data,nan=float(np.nanmedian(data)))
        result['saturated_fraction']=float(np.mean(data>=65000))
        small=bin_mean(data,4)
        bg=sep.Background(np.ascontiguousarray(small),bw=64,bh=64)
        result['background_adu']=float(bg.globalback)
        result['gradient_adu']=float(np.percentile(bg.back(),95)-np.percentile(bg.back(),5))
        stars,crops=star_measurements(data);result.update(stars)
        # Detect narrow row/column offsets without fitting away nebular structure.
        row=np.median(data[:,::8],axis=1);col=np.median(data[::8,:],axis=0)
        row_res=row-median_filter(row,size=15,mode='nearest')
        col_res=col-median_filter(col,size=15,mode='nearest')
        result['band_amplitude_adu']=float(max(np.percentile(abs(row_res),95),np.percentile(abs(col_res),95)))
        result['band_noise_ratio']=result['band_amplitude_adu']/max(result['noise_adu'],1e-6)
        result['row_jump_adu']=float(np.max(abs(np.diff(gaussian_filter1d(row,2)))))
        result['signal_span_adu']=float(np.percentile(small,99)-np.percentile(small,20))
        result['status']='candidate'; reasons=[]
        if result['signal_span_adu']<=0 or result['noise_adu']<=0:
            result['status']='reject';reasons.append('constant or degenerate image')
        if result['sample_stars']>=12 and result['axis_ratio']<0.40 and result['trail_fraction']>0.7:
            result['status']='reject';reasons.append('severe elongated stellar profiles')
        elif result['sample_stars']>=8 and result['axis_ratio']<0.65:
            result['status']='review';reasons.append('elongated stellar profiles')
        if result['band_noise_ratio']>0.7:
            if result['status']!='reject':result['status']='review'
            reasons.append('row/column pattern needs visual review')
        if result['sample_stars']<8:
            if result['status']!='reject':result['status']='review'
            reasons.append('too few stellar detections for reliable tracking assessment')
        if float(result.get('exposure') or 0)<1 or 'SNAPSHOT' in rel.parts:
            result['status']='planetary_or_test';reasons.append('separate planetary/test workflow')
        result['reasons']=reasons
        result=refine_field_metrics(result,small)
        sheet=Image.new('RGB',(960,760),(18,20,25));draw=ImageDraw.Draw(sheet)
        thumb=display_mono(small,(940,570));sheet.paste(thumb,((960-thumb.width)//2,40))
        draw.text((12,8),f"{result.get('object')} | {result.get('filter')} | {result.get('exposure')} s | {result['night']}",fill='white')
        for i,crop in enumerate(crops):
            sheet.paste(display_mono(crop).resize((150,150),Image.Resampling.NEAREST),(10+i*190,600))
        draw.text((12,575),f"{result['status']}   axis b/a={result['axis_ratio']}   FWHM={result['fwhm_px']} px   native star crops below",fill='white')
        sheet.save(out/(key+'.jpg'),quality=88)
        # Small linear data and catalog support quick, deterministic review stacks.
        cache=root/'quality'/'cache';cache.mkdir(parents=True,exist_ok=True)
        np.save(cache/(key+'.npy'),small)
    except Exception as e:
        status='non_image' if str(e)=='FITS has no image HDU' else 'reject'
        result.update(status=status,reasons=[type(e).__name__+': '+str(e)])
    atomic_json(out/(key+'.json'),result)
    return result


def scan(root, workers=4):
    root=Path(root);results=[]
    paths=sorted(p for p in (root/'raw').rglob('*') if p.suffix.lower() in {'.fits','.fit','.fts'})
    report=root/'quality'/'frame_report.json'
    previous={r['path']:r for r in json.loads(report.read_text())} if report.exists() else {}
    pending=[];upgrades=[]
    for i,p in enumerate(paths):
        rel=p.relative_to(root/'raw');key=hashlib.sha256(str(rel).encode()).hexdigest()[:20]
        saved=root/'quality'/'frames'/(key+'.json')
        old=previous.get(str(rel))
        if old is None and saved.exists():old=json.loads(saved.read_text())
        if old and old.get('qc_version')==QC_VERSION:
            results.append(old)
        elif old and old.get('qc_version') in {2,3}:
            upgrades.append(old)
        else:
            pending.append(p)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(upgrade_cached,r,root) for r in upgrades]+[pool.submit(inspect_one,p,root) for p in pending]
        for job in as_completed(jobs):
            result=job.result();results.append(result)
            print(f"QC {len(results)}/{len(paths)} {result['status']}: {result['path']}",flush=True)
            if len(results)%25==0:
                atomic_json('/content/omi_qc_live.json',results)
                atomic_json(report,results)
    atomic_json(root/'quality'/'frame_report.json',results)
    columns=['night','object','filter','exposure','status','sample_stars','fwhm_px','axis_ratio',
             'trail_fraction','background_adu','noise_adu','gradient_adu','band_noise_ratio',
             'saturated_fraction','reasons','path']
    with (root/'quality'/'frame_report.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for r in results:writer.writerow(r)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args()
    scan(args.root)
