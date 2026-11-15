"""Stronger presentation smoothing and honest downsampled HD exports.

Run on Colab after the soft presentation. All outputs are separate derivatives;
no upscaling, external images, generated content, or changes to linear masters.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import sep
import tifffile
from astropy.io import fits
from PIL import Image, ImageCms, ImageDraw
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means


def smooth_hd(folder, palette='SHO', strength=.055, hd_width=1920):
    folder=Path(folder);start=time.time()
    if not 0 < strength <= .12:
        raise ValueError('Presentation strength must be in (0,.12]')
    p=json.loads((folder/'render_report.json').read_text())
    inp=folder/(palette+'_Soft_Clouds_16bit.tif')
    rgb=tifffile.imread(inp).astype(np.float32)/65535
    height,width=rgb.shape[:2]
    if not 0 < hd_width <= width:
        raise ValueError('HD width must downsample the existing observation')
    scale=4/int(p['binning'])
    y0,y1,x0,x1=[int(v*scale) for v in p['parameters']['crop_bin4']]
    channels=p['parameters']['channels']
    ref='G' if 'G' in channels else ('Ha' if 'Ha' in channels else channels[0])
    raw=fits.getdata(Path(p['input_run'])/f"master_{ref}_bin{p['binning']}.fits")[y0:y1,x0:x1]
    detection=np.ascontiguousarray(raw,dtype=np.float32)
    assert detection.shape==rgb.shape[:2]
    bg=sep.Background(detection,bw=48,bh=48)
    stars=sep.extract(detection-bg.back(),4.5,err=bg.rms(),minarea=3)
    stars=stars[(stars['a']>.5)&(stars['a']<8)&(stars['b']/stars['a']>.5)&(stars['npix']<600)]
    mask=np.zeros((height,width),np.float32)
    for s in stars:
        x,y=float(s['x']),float(s['y'])
        sigma=float(np.clip(.85*np.sqrt(s['a']*s['b']),.85,2.5))
        r=int(np.ceil(4*sigma));ix,iy=int(round(x)),int(round(y))
        ya,yb=max(0,iy-r),min(height,iy+r+1);xa,xb=max(0,ix-r),min(width,ix+r+1)
        yy,xx=np.ogrid[ya:yb,xa:xb]
        core=np.exp(-.5*((xx-x)**2+(yy-y)**2)/sigma**2)
        core/=core.max()
        mask[ya:yb,xa:xb]=np.maximum(mask[ya:yb,xa:xb],core)
    print('Compact stellar cores protected:',len(stars),flush=True)
    den=denoise_nl_means(rgb,h=strength,patch_size=5,patch_distance=11,
                        fast_mode=True,channel_axis=-1,preserve_range=True)
    den=gaussian_filter(den,(.65,.65,0))
    # Restore only measured compact cores; wide masks would reintroduce grain.
    out=den+mask[...,None]*(rgb-den)
    # Very mild contrast at cloud scales, excluding stars and the faintest sky.
    coeff=np.array([.2126,.7152,.0722],np.float32)
    lum=out@coeff;broad=gaussian_filter(lum,24)
    cloud=np.clip((broad-.04)/.16,0,1)*(1-mask)
    detail=gaussian_filter(lum,6)-broad
    newlum=np.clip(lum+.12*cloud*detail,.005,.98)
    out*=np.clip(newlum/np.maximum(lum,.005),.9,1.1)[...,None]
    out=np.clip(out,0,1)
    assert np.isfinite(out).all()
    protected=mask>=1
    core_error=float(np.max(np.abs(out[protected]-rgb[protected])))
    assert core_error<1e-6
    icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    def export(a,stem):
        t=np.uint16(np.rint(a*65535))
        tifffile.imwrite(folder/(stem+'_16bit.tif'),t,photometric='rgb',
                         extratags=[(34675,'B',len(icc),icc,False)])
        assert np.array_equal(t,tifffile.imread(folder/(stem+'_16bit.tif')))
        im=Image.fromarray(np.uint8(np.rint(a*255)))
        im.save(folder/(stem+'.png'),icc_profile=icc)
        assert Image.open(folder/(stem+'.png')).size==(a.shape[1],a.shape[0])
        im.save(folder/(stem+'.jpg'),quality=97,subsampling=0,icc_profile=icc)
        return {'shape':list(a.shape),'png':str(folder/(stem+'.png')),
                'tiff':str(folder/(stem+'_16bit.tif')),'sha256':hashlib.sha256((folder/(stem+'.png')).read_bytes()).hexdigest()}
    exports={'full_resolution':export(out,palette+'_Smooth_FullResolution')}
    hd_height=round(height*hd_width/width)
    hd=np.stack([np.asarray(Image.fromarray(out[...,c]).resize((hd_width,hd_height),Image.Resampling.LANCZOS)) for c in range(3)],-1)
    hd=np.clip(hd,0,1)
    exports['hd']=export(hd,f'{palette}_Smooth_HD_{hd_width}x{hd_height}')
    prior_path=folder/(palette+'_Clean_Clouds_16bit.tif')
    prior=tifffile.imread(prior_path).astype(np.float32)/65535 if prior_path.exists() else rgb
    prior_lum=prior@coeff;out_lum=out@coeff
    quiet=(mask<.02)&(gaussian_filter(prior_lum,20)<np.percentile(gaussian_filter(prior_lum,20),35))
    def scatter(a):
        r=(a-gaussian_filter(a,4))[quiet]
        return float(1.4826*np.median(np.abs(r-np.median(r))))
    report={'input':str(inp),'comparison_input':str(prior_path),'exports':exports,
            'strength':strength,'patch_size':5,'patch_distance':11,'post_smoothing_sigma':.65,
            'protected_stars':len(stars),'stellar_core_max_change':core_error,
            'quiet_scatter_previous':scatter(prior_lum),'quiet_scatter_smooth':scatter(out_lum),
            'elapsed_seconds':time.time()-start,
            'limits':'Strong presentation smoothing suppresses faint small-scale signal as well as noise. HD is a full-field downsample, not recovered or invented resolution. Original and linear masters are retained.'}
    (folder/(palette+'_Smooth_HD_report.json')).write_text(json.dumps(report,indent=2))
    cx,cy=int(width*.57),int(height*.62)
    xa,xb=max(0,cx-400),min(width,cx+400);ya,yb=max(0,cy-350),min(height,cy+350)
    canvas=Image.new('RGB',(2*(xb-xa),yb-ya+40),(12,12,14));draw=ImageDraw.Draw(canvas)
    for i,(a,label) in enumerate([(prior,'Previous clean version'),(out,'Stronger smooth presentation')]):
        canvas.paste(Image.fromarray(np.uint8(np.rint(a[ya:yb,xa:xb]*255))),(i*(xb-xa),40))
        draw.text((i*(xb-xa)+12,10),label,fill='white',font_size=18)
    canvas.save(folder/(palette+'_Smooth_comparison.jpg'),quality=97,icc_profile=icc)
    print(json.dumps(report,indent=2),flush=True)
    return report
