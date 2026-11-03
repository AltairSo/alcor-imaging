"""Observation-derived presentation rendering, separate from archival integration.

Restores the earlier B20 global-colour workflow: common sky anchors, star-excluded
diffuse scales, fixed spatial smoothing, and soft monotonic highlight curves.
No external image, generated detail, local hue replacement or star replacement.
Run on Colab; linear masters are never altered.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sep
import tifffile
from astropy.io import fits
from PIL import Image, ImageCms
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_wavelet

from archive import atomic_json


def mad_noise(a):
    d=np.diff(a[::2],axis=1)
    d=d[np.isfinite(d)]
    return max(float(np.median(np.abs(d-np.median(d)))/.67448975/np.sqrt(2)),1e-9)


def common_crop(arrays):
    valid=np.logical_and.reduce([np.isfinite(a) for a in arrays])
    h,w=valid.shape
    for margin in range(8,min(h,w)//4,2):
        if valid[margin:-margin,margin:-margin].all():
            return [margin,h-margin,margin,w-margin]
    raise ValueError('No conservative rectangular common coverage')


def star_mask(a):
    a=np.ascontiguousarray(a,dtype=np.float32)
    bg=sep.Background(a,bw=48,bh=48)
    obj=sep.extract(a-bg.back(),4.5,err=bg.rms(),minarea=3)
    obj=obj[(obj['a']>.5)&(obj['a']<8)&(obj['b']/obj['a']>.5)&(obj['npix']<600)]
    mask=np.zeros(a.shape,np.float32)
    h,w=a.shape
    for o in obj:
        x,y=float(o['x']),float(o['y'])
        r=float(np.clip(3.7*np.sqrt(o['a']*o['b']),3.25,12))
        ir=int(np.ceil(r+2));ix,iy=int(x),int(y)
        ya,yb=max(0,iy-ir),min(h,iy+ir+1);xa,xb=max(0,ix-ir),min(w,ix+ir+1)
        yy,xx=np.mgrid[ya:yb,xa:xb]
        m=np.clip((r+1-np.hypot(xx-x,yy-y))/2,0,1)
        mask[ya:yb,xa:xb]=np.maximum(mask[ya:yb,xa:xb],m)
    return mask,obj


def prepare(source, *, plane=False):
    """Estimate only global display parameters on the archived bin4 masters."""
    source=Path(source)
    report=json.loads((source/'processing_report.json').read_text())
    channels=[c for c,r in report['channels'].items() if r['status']=='integrated']
    a={}
    for c in channels:
        path=source/f'master_{c}_bin4.fits'
        if path.exists():a[c]=fits.getdata(path).astype(np.float32)
        else:
            b=int(report['binning']);factor=4//b
            v=fits.getdata(source/f'master_{c}_bin{b}.fits').astype(np.float32)
            h,w=v.shape
            a[c]=v[:h//factor*factor,:w//factor*factor].reshape(h//factor,factor,w//factor,factor).mean((1,3))
    crop=common_crop(list(a.values()));y0,y1,x0,x1=crop
    a={c:v[y0:y1,x0:x1] for c,v in a.items()}
    ref='G' if 'G' in a else ('Ha' if 'Ha' in a else channels[0])
    mask,stars=star_mask(a[ref]);noise={c:mad_noise(v) for c,v in a.items()}
    tiles=[];xy=[];h,w=a[ref].shape
    for y in range(16,h-16,32):
        for x in range(16,w-16,32):
            tiles.append([np.median(a[c][y-12:y+12,x-12:x+12]) for c in channels])
            xy.append([1,x/w-.5,y/h-.5])
    tiles=np.asarray(tiles);xy=np.asarray(xy)
    rank=lambda v:np.argsort(np.argsort(v))/len(v)
    sky_channels=[c for c in ['Ha','OIII'] if c in a] or [ref]
    score=np.mean([rank(tiles[:,channels.index(c)]) for c in sky_channels],axis=0)
    anchors=score<np.percentile(score,12)
    backgrounds={c:float(np.median(tiles[anchors,i])-.5*noise[c]) for i,c in enumerate(channels)}
    slopes={c:[0.,0.] for c in channels}
    if plane:
        # Slopes only: common low-emission tiles, constrained and auditable.
        chosen=(score<np.percentile(score,35)) & (np.max(np.abs(xy[:,1:]),axis=1)>.25)
        enough=all(np.count_nonzero(chosen&(xy[:,1]*sx>0)&(xy[:,2]*sy>0))>=4 for sx in [-1,1] for sy in [-1,1])
        if enough:
            for i,c in enumerate(channels):
                use=chosen.copy()
                for _ in range(4):
                    coef=np.linalg.lstsq(xy[use],tiles[use,i],rcond=None)[0]
                    resid=tiles[:,i]-xy@coef
                    sig=max(float(np.median(np.abs(resid[use]-np.median(resid[use])))*1.4826),noise[c]*.1)
                    use=chosen&(np.abs(resid-np.median(resid[use]))<3*sig)
                limit=max(float(np.ptp(np.percentile(tiles[chosen,i],[10,90]))),noise[c])
                scale=min(1.,limit/max(abs(coef[1])+abs(coef[2]),1e-12))
                slopes[c]=(coef[1:]*scale).tolist()
                adjusted=tiles[anchors,i]-xy[anchors,1:]@np.asarray(slopes[c])
                backgrounds[c]=float(np.median(adjusted)-.5*noise[c])
    keep=1-mask;den=gaussian_filter(keep,4)
    scales={}
    for c in channels:
        v=a[c]-backgrounds[c]
        if plane:
            yy,xx=np.ogrid[:h,:w]
            v=v-slopes[c][0]*((xx+.5)/w-.5)-slopes[c][1]*((yy+.5)/h-.5)
        interp=gaussian_filter(v*keep,4)/np.maximum(den,.001)
        diffuse=v*keep+interp*mask
        scales[c]=max(float(np.percentile(gaussian_filter(diffuse,1),99.3)),8*noise[c])
    gains={};phot=[]
    if {'R','G','B'}<=a.keys():
        yy,xx=np.mgrid[-12:13,-12:13];rr=np.hypot(xx,yy);disk=rr<4.5;ann=(rr>8)&(rr<12)
        for o in stars:
            x,y=int(o['x']),int(o['y'])
            if not(14<x<w-14 and 14<y<h-14):continue
            flux=[];peak=[]
            for c in 'RGB':
                p=a[c][y-12:y+13,x-12:x+13];sky=np.median(p[ann])
                flux.append(float(np.sum(p[disk]-sky)));peak.append(float(p[disk].max()-sky))
            if all(f>40*noise[c] and pk>8*noise[c] for c,f,pk in zip('RGB',flux,peak)) and max(peak)<25:
                phot.append(flux)
        if len(phot)<20:raise ValueError('Too few unsaturated reference stars for empirical RGB balance')
        phot=np.asarray(phot)
        gains={'R':float(np.median(phot[:,1]/phot[:,0])),'G':1.,'B':float(np.median(phot[:,1]/phot[:,2]))}
        if not all(.2<v<5 for v in gains.values()):raise ValueError('Implausible stellar colour gains')
    return {'channels':channels,'crop_bin4':crop,'backgrounds':backgrounds,'slopes':slopes,
            'scales':scales,'noise_bin4':noise,'rgb_gains':gains,'rgb_balance_stars':len(phot),
            'star_count':len(stars),'sky_anchor_tiles':int(anchors.sum()),'plane':plane,
            'star_mask_use':'Only for global diffuse-scale estimation; no star replacement'}


def tone(a, strength=6., shoulder=2.):
    v=np.arcsinh(strength*np.maximum(a,0))
    return .012+.975*v/(v+shoulder)


def compress_gamut(rgb, gray):
    """Reduce chroma only as needed to retain hue at the display boundaries."""
    gray=np.clip(gray,.012,.99)
    d=rgb-gray[...,None]
    lo=np.maximum(-d.min(-1),1e-9);hi=np.maximum(d.max(-1),1e-9)
    scale=np.minimum(1,np.minimum((gray-.012)/lo,(.99-gray)/hi))
    return np.clip(gray[...,None]+d*scale[...,None],.012,.99)


def render(report, *, settings=None, output_name='finished', parameters=None):
    defaults={'denoise':.88,'smooth_native':3.2,'saturation':1.55,'power':1.08,
              'oxygen':.7,'local_contrast':.22,'strength':8,'shoulder':1.9}
    if 'R' in report['channels']:
        defaults.update(saturation=1.15,strength=6,shoulder=2,local_contrast=.15)
    if len(report['channels'])==1:
        defaults.update(saturation=1,local_contrast=.24,strength=8,shoulder=1.8)
    settings={**defaults,**(settings or {})}
    source=Path(report.get('source_run',report['output']))
    inp=Path(report['output']);binning=int(report['binning'])
    p=parameters or prepare(source,plane=settings.get('plane',False))
    out=inp/output_name;out.mkdir(exist_ok=True)
    crop=[int(v*4/binning) for v in p['crop_bin4']];y0,y1,x0,x1=crop
    ch={};metrics={}
    for c in p['channels']:
        a=fits.getdata(inp/f'master_{c}_bin{binning}.fits').astype(np.float32)[y0:y1,x0:x1]
        if not np.isfinite(a).all():raise ValueError('Invalid pixels inside presentation crop')
        a-=p['backgrounds'][c]
        h,w=a.shape;yy,xx=np.ogrid[:h,:w]
        a-=p['slopes'][c][0]*((xx+.5)/w-.5)+p['slopes'][c][1]*((yy+.5)/h-.5)
        halo=settings.get('halo_models',{}).get(c)
        if halo:
            # Optional measured smooth halo, with no inward extrapolation into
            # an unsampled stellar core. Model and strength remain in the report.
            scale=4/binning;cx,cy=halo['center_bin4']
            rr=np.hypot((xx+.5)/scale-.5-cx,(yy+.5)/scale-.5-cy)
            rr=np.maximum(rr,halo.get('minimum_radius_bin4',0))
            model=halo['amplitude']/(1+(rr/halo['radius_bin4'])**2)**halo['beta']
            a-=float(settings.get('halo_strength',.85))*model
        noise=mad_noise(a)
        den=denoise_wavelet(a,sigma=noise,method='BayesShrink',mode='soft',wavelet='sym4',wavelet_levels=3,rescale_sigma=False,channel_axis=None)
        mix=float(settings.get('denoise',.8));a=(1-mix)*a+mix*den
        sigma=float(settings.get('smooth_native',2.6))/binning
        ch[c]=gaussian_filter(a,sigma).astype(np.float32)
        metrics[c]={'noise_before':noise,'noise_after':mad_noise(ch[c]),'smooth_sigma_pixels':sigma}
    palettes={};sc=p['scales']
    if {'Ha','OIII'}<=ch.keys():
        H=np.maximum(ch['Ha']/sc['Ha'],0);O=np.maximum(ch['OIII']/sc['OIII'],0)*float(settings.get('oxygen',.75))
        palettes['HOO']=np.stack([.85*H,.08*H+.4*O,.55*O],-1)
        # Explicit two-filter alternative, not a claimed SHO measurement.
        palettes['HOO_Amber_Blue']=np.stack([.85*H,.39*H+.24*O,.04*H+.72*O],-1)
    if {'SII','Ha','OIII'}<=ch.keys():
        S=np.maximum(ch['SII']/sc['SII'],0)
        palettes['SHO']=np.stack([.35*S+.35*H,.28*H+.10*S+.20*O,.70*O],-1)
    if {'R','G','B'}<=ch.keys():
        rgb=np.stack([ch[c]*p['rgb_gains'][c] for c in 'RGB'],-1)
        rgbscale=float(settings.get('rgb_scale',min(sc[c]*p['rgb_gains'][c] for c in 'RGB')))
        palettes['RGB']=rgb/rgbscale
    mono='Ha' if 'Ha' in ch else p['channels'][0]
    if not palettes:palettes[mono+'_Monochrome']=np.repeat(np.maximum(ch[mono]/sc[mono],0)[...,None],3,-1)
    icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    exports={}
    for name,rgb in palettes.items():
        img=tone(rgb,strength=float(settings.get('strength',6)),shoulder=float(settings.get('shoulder',2)))
        # A common display curve and saturation adjustment; no spatial hue map.
        img=.012+.975*np.power(np.clip((img-.012)/.975,0,1),float(settings.get('power',1.08)))
        lum=img@np.array([.2126,.7152,.0722],np.float32)
        saturation=float(settings.get('saturation',1.45 if name.startswith(('HOO','SHO')) else 1.15))
        img=lum[...,None]+saturation*(img-lum[...,None])
        amount=float(settings.get('local_contrast',.18))
        detail=gaussian_filter(lum,2/binning)-gaussian_filter(lum,40/binning)
        ratio=np.clip((lum+amount*detail)/np.maximum(lum,.012),.8,1.2)
        img*=ratio[...,None]
        img=compress_gamut(img,lum*ratio)
        im=Image.fromarray(np.uint8(np.rint(img*255)))
        im.save(out/(name+'.png'),icc_profile=icc)
        t=np.uint16(np.rint(img*65535));tifffile.imwrite(out/(name+'_16bit.tif'),t,photometric='rgb',extratags=[(34675,'B',len(icc),icc,False)])
        assert np.array_equal(t,tifffile.imread(out/(name+'_16bit.tif')))
        im.thumbnail((2400,1800),Image.Resampling.LANCZOS);im.save(out/(name+'_preview.jpg'),quality=96,icc_profile=icc)
        exports[name]={'shape':list(img.shape),'min':float(img.min()),'max':float(img.max()),
                       'black_fraction':float((img.max(-1)==0).mean()),'clipped_fraction':float((img.max(-1)>=1).mean())}
    result={'target':report['target'],'input_run':str(inp),'source_run':str(source),'binning':binning,
            'parameters':p,'settings':settings,'metrics':metrics,'exports':exports,'output':str(out),
            'limitations':['Lights only; not dark/flat calibrated.','Global mapped narrowband colours; RGB uses empirical stellar balance, not catalogue calibration.',
                          'Monotonic per-channel display curve; presentation pixels are not photometric fluxes.'],
            'method':'B20 global matrix and soft highlight curve; wavelet noise reduction; no external or generated detail.'}
    atomic_json(out/'render_report.json',result)
    print('FINISHED',report['target'],out,exports,flush=True)
    return result
