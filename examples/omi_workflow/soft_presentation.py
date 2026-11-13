"""Gentle, observation-derived cloud smoothing and stellar attenuation.

Run on Colab after finish.render. Original presentations and linear masters
remain unchanged. A measured star mask protects profiles during smoothing;
only positive compact stellar light is then attenuated, never replaced.
"""
from pathlib import Path
import json

import numpy as np
import sep
import tifffile
from astropy.io import fits
from PIL import Image, ImageCms, ImageDraw
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_wavelet

from finish import compress_gamut, mad_noise


def soften(folder, palette='SHO', star_reduction=.23):
    folder=Path(folder)
    p=json.loads((folder/'render_report.json').read_text())
    inp=folder/(palette+'_16bit.tif')
    rgb=tifffile.imread(inp).astype(np.float32)/65535
    lum=rgb@np.array([.2126,.7152,.0722],np.float32)
    h,w=lum.shape
    scale=4/int(p['binning'])
    y0,y1,x0,x1=[int(v*scale) for v in p['parameters']['crop_bin4']]
    ref='G' if 'G' in p['parameters']['channels'] else 'Ha'
    refpath=Path(p['input_run'])/f"master_{ref}_bin{p['binning']}.fits"
    detection=np.ascontiguousarray(fits.getdata(refpath)[y0:y1,x0:x1],dtype=np.float32)
    assert detection.shape==lum.shape
    bg=sep.Background(detection,bw=64,bh=64)
    stars=sep.extract(detection-bg.back(),5,err=bg.rms(),minarea=5)
    stars=stars[(stars['a']>.8)&(stars['a']<8)&(stars['b']>.65*stars['a'])&
                (stars['npix']<500)&(stars['flux']/np.maximum(stars['peak'],1e-8)>5)]
    mask=np.zeros_like(lum)
    smooth_sigma=float(p['metrics'][ref]['smooth_sigma_pixels'])
    for s in stars:
        x,y=float(s['x']),float(s['y'])
        radius=float(np.clip(1.65*np.sqrt(s['a']*s['b']+smooth_sigma**2),2.5,10))
        ir=int(np.ceil(3*radius));ix,iy=int(x),int(y)
        ya,yb=max(0,iy-ir),min(h,iy+ir+1);xa,xb=max(0,ix-ir),min(w,ix+ir+1)
        yy,xx=np.ogrid[ya:yb,xa:xb]
        core=np.exp(-.5*((xx-x)**2+(yy-y)**2)/radius**2)
        mask[ya:yb,xa:xb]=np.maximum(mask[ya:yb,xa:xb],core)
    print('Measured stars:',len(stars),flush=True)
    noise=mad_noise(lum)
    den=denoise_wavelet(lum,sigma=1.3*noise,method='BayesShrink',mode='soft',
                        wavelet='sym4',wavelet_levels=4,rescale_sigma=False)
    den=gaussian_filter(den,.85)
    diffuse=1-mask
    soft=lum+.85*diffuse*(den-lum)
    # Gentle luminance-only midtone lift and medium-scale cloud contrast.
    broad=gaussian_filter(lum,20)
    cloud=np.clip((broad-.025)/.14,0,1)*diffuse
    detail=gaussian_filter(soft,3)-gaussian_filter(soft,22)
    lifted=soft+cloud*(.08*soft*(1-soft)+.12*detail)
    lifted=np.clip(lifted,.012,.97)
    # Fixed-scale chroma smoothing, without spatial hue remapping.
    chroma=rgb-lum[...,None]
    chroma+=.7*diffuse[...,None]*(gaussian_filter(chroma,(1.2,1.2,0))-chroma)
    out=lifted[...,None]+chroma*(lifted/np.maximum(lum,.02))[...,None]
    out=compress_gamut(out,lifted)
    # Subtract only a fraction of the local positive stellar excess. This
    # bounded operation cannot create dark star holes or invent missing detail.
    local=gaussian_filter(lifted,7)
    excess=np.maximum(lifted-local,0)
    reduction=star_reduction*mask*excess
    ratio=(lifted-reduction)/np.maximum(lifted,.012)
    out*=ratio[...,None]
    assert np.isfinite(out).all() and out.min()>=0 and out.max()<=1
    stem=palette+'_Soft_Clouds'
    icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    t=np.uint16(np.rint(out*65535))
    tifffile.imwrite(folder/(stem+'_16bit.tif'),t,photometric='rgb',
                     extratags=[(34675,'B',len(icc),icc,False)])
    assert np.array_equal(t,tifffile.imread(folder/(stem+'_16bit.tif')))
    im=Image.fromarray(np.uint8(np.rint(out*255)))
    im.save(folder/(stem+'.png'),icc_profile=icc)
    im.thumbnail((2400,1800),Image.Resampling.LANCZOS)
    im.save(folder/(stem+'_preview.jpg'),quality=96,icc_profile=icc)
    # Use identical faint, nonstellar pixels to compare fine-scale scatter.
    quiet=(mask<.03)&(broad<np.percentile(broad,35))
    def scatter(v):
        r=(v-gaussian_filter(v,2))[quiet]
        return float(1.4826*np.median(np.abs(r-np.median(r))))
    outlum=out@np.array([.2126,.7152,.0722],np.float32)
    sx=np.clip(np.rint(stars['x']).astype(int),0,w-1)
    sy=np.clip(np.rint(stars['y']).astype(int),0,h-1)
    core_ratios=(outlum[sy,sx]/np.maximum(lum[sy,sx],.012))
    report={'input':str(inp),'output':str(folder/(stem+'.png')),'shape':list(out.shape),
            'star_detection_channel':ref,'measured_stars':len(stars),
            'stellar_excess_attenuation':star_reduction,
            'median_stellar_center_luminance_ratio':float(np.median(core_ratios)),
            'quiet_high_frequency_scatter_before':scatter(lum),
            'quiet_high_frequency_scatter_after':scatter(outlum),
            'method':'Measured-star-protected wavelet and Gaussian smoothing; fixed chroma smoothing; mild cloud luminance contrast; bounded stellar-excess attenuation.',
            'limits':'Presentation only. No generated content, external pixels, star replacement, deconvolution, or changes to linear masters.'}
    (folder/(stem+'_report.json')).write_text(json.dumps(report,indent=2))
    canvas=Image.new('RGB',(2200,850),(12,12,14));draw=ImageDraw.Draw(canvas)
    for i,(path,title) in enumerate([(inp,'Original SHO'),(folder/(stem+'_16bit.tif'),'Softer clouds / gentler stars')]):
        a=tifffile.imread(path);thumb=Image.fromarray(np.uint8(np.rint(a/257)))
        thumb.thumbnail((1080,790),Image.Resampling.LANCZOS)
        canvas.paste(thumb,(i*1100+(1100-thumb.width)//2,45+(790-thumb.height)//2))
        draw.text((i*1100+20,15),title,fill='white',font_size=22)
    canvas.save(folder/'SHO_Soft_Clouds_comparison.jpg',quality=96,icc_profile=icc)
    # Exact pixel-scale diagnostic cutouts of the same central field.
    cx,cy=int(w*.57),int(h*.62);box=(max(0,cx-400),max(0,cy-350),min(w,cx+400),min(h,cy+350))
    detail_canvas=Image.new('RGB',(1600,740),(12,12,14));d=ImageDraw.Draw(detail_canvas)
    for i,(a,label) in enumerate([(rgb,'Original: 1 display pixel = 1 export pixel'),(out,'Softer: 1 display pixel = 1 export pixel')]):
        crop=Image.fromarray(np.uint8(np.rint(a[box[1]:box[3],box[0]:box[2]]*255)))
        detail_canvas.paste(crop,(i*800,40));d.text((i*800+12,12),label,fill='white',font_size=16)
    detail_canvas.save(folder/'SHO_Soft_Clouds_detail.jpg',quality=97,icc_profile=icc)
    print(json.dumps(report,indent=2),flush=True)
    return report
