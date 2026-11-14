import io
import json
import zipfile

import numpy as np
import pytest
from astropy.io import fits

import archive
from quality import star_measurements, refine_field_metrics, inspect_one


def test_reference_override_can_anchor_inspected_field():
    from process import choose_reference
    wrong={'path':'wrong-field-ha.fits','filter':'Ha','fwhm_px':2.0}
    correct={'path':'correct-field-oiii.fits','filter':'OIII','fwhm_px':3.0}
    assert choose_reference([wrong,correct],correct['path']) is correct
    assert choose_reference([wrong,correct]) is wrong


def test_reference_override_cannot_bypass_frame_selection():
    from process import choose_reference
    selected=[{'path':'accepted.fits','filter':'Ha','fwhm_px':3.0}]
    with pytest.raises(ValueError,match='selected light frame'):
        choose_reference(selected,'rejected.fits')


@pytest.mark.parametrize('name',['../outside.fits','/absolute.fits','a/../../x','a\\..\\x'])
def test_unsafe_paths(name):
    with pytest.raises(ValueError):archive.safe_relative(name)


def test_download_query_preserves_shared_key():
    u=archive.download_url('https://www.dropbox.com/scl/fo/example?rlkey=private&dl=0')
    assert 'rlkey=private' in u and 'dl=1' in u
    with pytest.raises(ValueError):archive.download_url('https://dropbox.com.evil.example/a')


class Response:
    def __init__(self,payload,content_type='application/binary'):
        self.payload=payload
        self.headers={'Content-Length':str(len(payload)),'Content-Type':content_type}
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def raise_for_status(self):pass
    def iter_content(self,n):yield self.payload


def payload():
    b=io.BytesIO();fits.PrimaryHDU(np.arange(256,dtype='uint16').reshape(16,16)).writeto(b)
    return b.getvalue()


def test_archive_resume_and_missing_receipt(tmp_path,monkeypatch):
    item={'path':'night/target/a.fits','url':'https://www.dropbox.com/scl/fi/example?dl=0'}
    monkeypatch.setattr(archive.requests,'get',lambda *a,**k:Response(payload()))
    a=archive.import_one(item,tmp_path,retries=1)
    assert a['status']=='downloaded' and a['fits_error'] is None
    assert archive.import_one(item,tmp_path,retries=1)['status']=='verified_existing'
    receipt=tmp_path/'receipts/night/target/a.fits.json';receipt.unlink()
    assert archive.import_one(item,tmp_path,retries=1)['sha256']==a['sha256']
    assert json.loads(receipt.read_text())['bytes']==len(payload())


def test_bad_sources_archived_but_html_rejected(tmp_path,monkeypatch):
    item={'path':'night/broken.fits','url':'https://www.dropbox.com/scl/fi/example'}
    monkeypatch.setattr(archive.requests,'get',lambda *a,**k:Response(b'broken-source'))
    r=archive.import_one(item,tmp_path,retries=1)
    assert r['status']=='downloaded' and r['fits_error']
    assert (tmp_path/'raw/night/broken.fits').read_bytes()==b'broken-source'
    item['path']='night/login.fits'
    monkeypatch.setattr(archive.requests,'get',lambda *a,**k:Response(b'<html>login</html>','text/html'))
    assert archive.import_one(item,tmp_path,retries=1)['status']=='failed'
    assert not (tmp_path/'raw/night/login.fits').exists()


def test_conflicting_original_never_overwritten(tmp_path,monkeypatch):
    p=tmp_path/'raw/night/a.fits';p.parent.mkdir(parents=True);p.write_bytes(b'original')
    monkeypatch.setattr(archive.requests,'get',lambda *a,**k:Response(payload()))
    r=archive.import_one({'path':'night/a.fits','url':'https://www.dropbox.com/a'},tmp_path,retries=1)
    assert r['status']=='failed' and p.read_bytes()==b'original'


def test_zip_traversal(tmp_path,monkeypatch):
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:z.writestr('../escape.fits',payload())
    monkeypatch.setattr(archive.requests,'get',lambda *a,**k:Response(b.getvalue()))
    with pytest.raises(ValueError):
        archive.import_folder_link('https://www.dropbox.com/folder',tmp_path/'drive','night',tmp_path/'stage')
    assert not (tmp_path/'escape.fits').exists()


def test_trails_are_measured_and_hot_pixels_excluded():
    rng=np.random.default_rng(12);yy,xx=np.mgrid[:1024,:1024]
    def field(sx,sy):
        a=rng.normal(500,4,(1024,1024)).astype('float32')
        for y in range(60,1000,85):
            for x in range(60,1000,85):
                a+=800*np.exp(-0.5*((xx-x)**2/sx**2+(yy-y)**2/sy**2))
        for y,x in rng.integers(0,1024,(120,2)):a[y,x]+=18000
        return a
    good,_=star_measurements(field(2.5,2.5))
    bad,_=star_measurements(field(11,2))
    assert good['sample_stars']>30 and good['axis_ratio']>0.8 and good['fwhm_px']>4
    assert bad['sample_stars']>30 and bad['axis_ratio']<0.4 and bad['trail_fraction']>0.8


def test_full_field_catches_trails_with_sparse_native_samples():
    yy,xx=np.mgrid[:512,:512];rng=np.random.default_rng(9)
    a=rng.normal(500,1,(512,512)).astype('float32')
    for y in range(25,500,50):
        for x in range(25,500,50):a+=80*np.exp(-.5*((xx-x)**2/7**2+(yy-y)**2/1.1**2))
    r=refine_field_metrics({'status':'review','reasons':[]},a)
    assert r['field_stars']>30 and r['status']=='reject'


def test_broad_sensor_step_flagged_even_with_round_stars():
    yy,xx=np.mgrid[:512,:512];rng=np.random.default_rng(19)
    a=rng.normal(500,1,(512,512)).astype('float32')
    for y in range(25,500,50):
        for x in range(25,500,50):a+=80*np.exp(-.5*((xx-x)**2+(yy-y)**2)/1.4**2)
    r=refine_field_metrics({'status':'candidate','reasons':[],'row_jump_adu':100,'noise_adu':5},a)
    assert r['status']=='review' and 'abrupt full-width row discontinuity' in r['reasons']


def test_metadata_only_sync_file_is_not_a_failed_exposure(tmp_path):
    root=tmp_path/'archive';(root/'raw').mkdir(parents=True)
    path=root/'raw/sync.fits';fits.PrimaryHDU().writeto(path)
    r=inspect_one(path,root)
    assert r['status']=='non_image' and r['night']=='root'


def test_repeated_folder_import_refreshes_updated_night(tmp_path,monkeypatch):
    def zipped(names):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w') as z:
            for name in names:z.writestr(name,payload())
        return b.getvalue()
    incoming=iter([zipped(['a.fits']),zipped(['a.fits','b.fits'])])
    monkeypatch.setattr(archive.requests,'get',lambda *a,**k:Response(next(incoming)))
    url='https://www.dropbox.com/folder'
    archive.import_folder_link(url,tmp_path/'drive','night',tmp_path/'stage',keep_staging=True)
    records=archive.import_folder_link(url,tmp_path/'drive','night',tmp_path/'stage')
    assert len(records)==2 and (tmp_path/'drive/raw/night/b.fits').exists()
    assert not list((tmp_path/'stage').glob('*.zip'))


def test_finishing_curve_preserves_order_without_hard_highlight_clipping():
    from finish import tone
    x=np.logspace(-6,6,1000)
    y=tone(x)
    assert np.all(np.diff(y)>0) and y.min()>.011 and y.max()<1


def test_gamut_compression_preserves_chroma_direction():
    from finish import compress_gamut
    rgb=np.array([[1.4,.3,-.2],[.1,.2,.3]],dtype=np.float32)
    gray=np.array([.4,.2],dtype=np.float32)
    out=compress_gamut(rgb,gray)
    assert out.min()>=.01199 and out.max()<=.99001
    np.testing.assert_allclose(out[1],rgb[1],atol=1e-7)
    before=rgb[0]-gray[0];after=out[0]-gray[0]
    np.testing.assert_allclose(after/np.linalg.norm(after),before/np.linalg.norm(before),atol=1e-6)


def test_common_crop_excludes_missing_coverage():
    from finish import common_crop
    a=np.ones((120,160),np.float32);a[:10]=np.nan;a[:,:12]=np.nan
    crop=common_crop([a,np.ones_like(a)])
    y0,y1,x0,x1=crop
    assert np.isfinite(a[y0:y1,x0:x1]).all()


@pytest.mark.parametrize('overrides',[None,{'saturation':1.3,'strength':7}])
def test_mixed_filters_do_not_change_individual_palette_exports(tmp_path,overrides):
    from finish import render
    import tifffile
    yy,xx=np.mgrid[:64,:80]
    channels=['SII','Ha','OIII','R','G','B']
    rng=np.random.default_rng(42)
    for i,c in enumerate(channels):
        a=(.05+(.3+.07*i)*np.exp(-((xx-30-i)**2+(yy-32)**2)/500)
           +rng.normal(0,.002,xx.shape)).astype('float32')
        fits.PrimaryHDU(a).writeto(tmp_path/f'master_{c}_bin4.fits')
    def run(chosen,label):
        report={'target':'test','binning':4,'output':str(tmp_path),
                'channels':{c:{'status':'integrated'} for c in chosen}}
        p={'channels':chosen,'crop_bin4':[8,56,8,72],
           'backgrounds':{c:0 for c in chosen},'slopes':{c:[0,0] for c in chosen},
           'scales':{c:1 for c in chosen},'rgb_gains':{c:1 for c in 'RGB'}}
        return render(report,parameters=p,settings=overrides,output_name=label)
    mixed=run(channels,'mixed')
    run(channels[:3],'narrowband');run(channels[3:],'broadband')
    for palette,group in [('SHO','narrowband'),('HOO','narrowband'),
                          ('HOO_Amber_Blue','narrowband'),('RGB','broadband')]:
        np.testing.assert_array_equal(tifffile.imread(tmp_path/'mixed'/f'{palette}_16bit.tif'),
                                      tifffile.imread(tmp_path/group/f'{palette}_16bit.tif'))
    if overrides:
        for s in mixed['palette_settings'].values():
            assert s['saturation']==overrides['saturation'] and s['strength']==overrides['strength']
