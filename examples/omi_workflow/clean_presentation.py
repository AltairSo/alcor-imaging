"""Optional, inspected correlated-grain reduction for a finished presentation.

Run in Colab after soft_presentation.soften. This uses local observed patches,
not a learned or generative model. Preserve the input and linear FITS masters.
"""
from pathlib import Path
import json
import time

import numpy as np
import tifffile
from astropy.io import fits
from PIL import Image, ImageCms, ImageDraw
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means

from finish import star_mask


def clean_grain(folder, palette='SHO', strength=.03, mix=.9):
    if not (0 < strength <= .1 and 0 <= mix <= 1):
        raise ValueError('Expected strength in (0,.1] and mix in [0,1]')
    t0=time.time();folder=Path(folder)
    p=json.loads((folder/'render_report.json').read_text())
    inp=folder/(palette+'_Soft_Clouds_16bit.tif')
    rgb=tifffile.imread(inp).astype(np.float32)/65535
    scale=4/int(p['binning'])
    y0,y1,x0,x1=[int(v*scale) for v in p['parameters']['crop_bin4']]
    channels=p['parameters']['channels']
    ref='G' if 'G' in channels else ('Ha' if 'Ha' in channels else channels[0])
    detection=fits.getdata(Path(p['input_run'])/f"master_{ref}_bin{p['binning']}.fits")[y0:y1,x0:x1]
    assert detection.shape==rgb.shape[:2]
    mask,stars=star_mask(detection)
    print('Protected measured stars:',len(stars),flush=True)
    den=denoise_nl_means(rgb,h=strength,patch_size=5,patch_distance=9,
                        fast_mode=True,channel_axis=-1,preserve_range=True)
    out=rgb+(mix*(1-mask))[...,None]*(den-rgb)
    assert np.isfinite(out).all() and out.min()>=0 and out.max()<=1
    protected=mask>=1
    star_change=float(np.max(np.abs(out[protected]-rgb[protected]))) if protected.any() else None
    assert star_change is None or star_change==0
    stem=palette+'_Clean_Clouds'
    icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    t=np.uint16(np.rint(out*65535))
    tifffile.imwrite(folder/(stem+'_16bit.tif'),t,photometric='rgb',
                     extratags=[(34675,'B',len(icc),icc,False)])
    assert np.array_equal(t,tifffile.imread(folder/(stem+'_16bit.tif')))
    im=Image.fromarray(np.uint8(np.rint(out*255)));im.save(folder/(stem+'.png'),icc_profile=icc)
    im.thumbnail((2400,1800),Image.Resampling.LANCZOS)
    im.save(folder/(stem+'_preview.jpg'),quality=96,icc_profile=icc)
    lum=rgb@np.array([.2126,.7152,.0722],np.float32)
    broad=gaussian_filter(lum,20)
    quiet=(mask<.03)&(broad<np.percentile(broad,35))
    def scatter(v):
        a=(v-gaussian_filter(v,4))[quiet]
        return float(1.4826*np.median(np.abs(a-np.median(a))))
    report={'input':str(inp),'output':str(folder/(stem+'.png')),'shape':list(out.shape),
            'strength':strength,'diffuse_mix':mix,'patch_size':5,'patch_distance':9,
            'protected_measured_stars':len(stars),'protected_core_max_change':star_change,
            'quiet_scatter_before':scatter(lum),
            'quiet_scatter_after':scatter(out@np.array([.2126,.7152,.0722],np.float32)),
            'elapsed_seconds':time.time()-t0,
            'method':'Measured-star-protected joint RGB non-local means on the existing soft presentation.',
            'limits':'Presentation only. Smoothing can suppress faint small-scale signal. No generated content, external image, replacement stars, or changes to linear masters.'}
    (folder/(stem+'_report.json')).write_text(json.dumps(report,indent=2))
    h,w=rgb.shape[:2];cx,cy=int(w*.57),int(h*.62)
    xa,xb=max(0,cx-400),min(w,cx+400);ya,yb=max(0,cy-350),min(h,cy+350)
    canvas=Image.new('RGB',(2*(xb-xa),yb-ya+40),(12,12,14));draw=ImageDraw.Draw(canvas)
    for i,(a,label) in enumerate([(rgb,'Gentle finish'),(out,'Clean clouds: protected stars')]):
        canvas.paste(Image.fromarray(np.uint8(np.rint(a[ya:yb,xa:xb]*255))),(i*(xb-xa),40))
        draw.text((i*(xb-xa)+12,10),label,fill='white',font_size=18)
    canvas.save(folder/(stem+'_detail.jpg'),quality=97,icc_profile=icc)
    print(json.dumps(report,indent=2),flush=True)
    return report
