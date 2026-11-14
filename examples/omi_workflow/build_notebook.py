"""Build the portable Colab notebook without embedding private Dropbox links."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).parent
CORE_REF='23bbf079057d1546f55a0c158f74f4d38cd7ee05'


def build(destination=None):
    modules={name:(HERE/name).read_text() for name in ['archive.py','quality.py','process.py','finish.py','report.py','test_workflow.py']}
    cells=[]
    def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(True)})
    def code(text,form=False):cells.append({'cell_type':'code','metadata':{'cellView':'form'} if form else {},'execution_count':None,'outputs':[],'source':text.splitlines(True)})
    md('''# OMI → Colab → Google Drive

Archive original FITS, screen each exposure, register and integrate suitable lights on the GPU, and save linear masters plus finished PNG/TIFF presentations to Drive.

**Next session:** paste the shared Dropbox link for that observing-night folder, choose **New night folder**, enter its date label and the target name shown in the FITS header, then **Run all**. Your laptop does not download or process the images. A personal Dropbox account is not required.

Use **Runtime → Change runtime type → L4 GPU** (another CUDA GPU also works). Google may ask you to mount Drive. Keep this notebook private if you enter access-bearing shared links.

The initial archive can be resumed with **Saved manifest**. **Already archived** skips transfer. This is a run-on-demand notebook, not a permanent Dropbox watcher; Colab sessions have time and availability limits.''')
    code('''#@title 1. Choose source and processing target
IMPORT_MODE = "Saved manifest" #@param ["Saved manifest", "New night folder", "Already archived"]
NEW_NIGHT_LINK = "" #@param {type:"string"}
NEW_NIGHT_LABEL = "" #@param {type:"string"}
TARGET = "Heart Nebula" #@param {type:"string"}
OBSERVING_NIGHT = "2026-11-01" #@param {type:"string"}
REFERENCE_PATH = "" #@param {type:"string"}
BINNING = 2 #@param [1, 2, 4] {type:"raw"}
RUN_PROCESSING = True #@param {type:"boolean"}
FINISH_IMAGES = True #@param {type:"boolean"}
DISPLAY_IMAGES = False #@param {type:"boolean"}
ARCHIVE_PATH = "/content/drive/MyDrive/Astro/OMI_Archive" #@param {type:"string"}
''',True)
    code(f'''#@title 2. Connect Drive and install the pinned Alcor core
from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import sys, subprocess, json, hashlib, re
import importlib.metadata as metadata
ROOT=Path(ARCHIVE_PATH); ROOT.mkdir(parents=True,exist_ok=True)
WORK=Path('/content/omi_workflow'); WORK.mkdir(exist_ok=True)
subprocess.run([sys.executable,'-m','pip','-q','install','sep','astroalign','pytest',
               'git+https://github.com/AltairSo/alcor-imaging.git@{CORE_REF}'],check=True)
driver_report=subprocess.check_output(['nvidia-smi'],text=True)
cuda_major=int(re.search(r'CUDA Version:\\s*(\\d+)',driver_report).group(1))
cupy_package='cupy-cuda13x' if cuda_major>=13 else 'cupy-cuda12x'
installed=[d.metadata['Name'] for d in metadata.distributions() if d.metadata['Name'] in ['cupy','cupy-cuda11x','cupy-cuda12x','cupy-cuda13x']]
if installed!=[cupy_package]:
    if installed:subprocess.run([sys.executable,'-m','pip','-q','uninstall','-y',*installed],check=True)
    subprocess.run([sys.executable,'-m','pip','-q','install','--no-deps',cupy_package],check=True)
sys.path.insert(0,str(WORK))
from alcor_imaging import gpu_available
assert gpu_available(), 'Select a GPU runtime before processing.'
print(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True))
''',True)
    code('#@title 3. Load the versioned workflow (included in this notebook)\nMODULES = '+repr(modules)+'''
for name,source in MODULES.items():
    (WORK/name).write_text(source)
saved=ROOT/'workflow';saved.mkdir(exist_ok=True)
for name,source in MODULES.items():(saved/name).write_text(source)
(saved/'source_hashes.json').write_text(json.dumps({n:hashlib.sha256(s.encode()).hexdigest() for n,s in MODULES.items()},indent=2))
(saved/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
checks=subprocess.run([sys.executable,'-m','pytest','-q',str(WORK/'test_workflow.py')],cwd=WORK,capture_output=True,text=True)
print(checks.stdout)
assert checks.returncode==0, checks.stderr
import importlib
importlib.invalidate_caches()
for filename in MODULES:
    module=filename.removesuffix('.py')
    if module in sys.modules:importlib.reload(sys.modules[module])
''',True)
    md('''### Transfer and archive

Original paths, bytes and SHA-256 checksums are retained. A repeat run verifies existing files and retries incomplete ones. Conflicting originals are not overwritten. Damaged source FITS are preserved and reported. A future folder import downloads a ZIP to temporary Colab disk; use individual observing-night links so it fits available space.''')
    code('''#@title 4. Import originals directly into Drive
from archive import import_manifest, import_folder_link
if IMPORT_MODE=='New night folder':
    assert NEW_NIGHT_LINK and NEW_NIGHT_LABEL, 'Enter the Dropbox night-folder link and date label.'
    transfers=import_folder_link(NEW_NIGHT_LINK,ROOT,NEW_NIGHT_LABEL)
elif IMPORT_MODE=='Saved manifest':
    manifest=ROOT/'source_manifest.complete.private.json'
    if not manifest.exists():manifest=ROOT/'source_manifest.private.json'
    assert manifest.exists(), 'For a new archive, choose New night folder.'
    transfers=import_manifest(json.loads(manifest.read_text()),ROOT,workers=4)
    failures=[r for r in transfers if r['status']=='failed']
    assert not failures, f'{len(failures)} transfers need a retry; see transfer_report.json'
else:
    print('Using archived FITS; transfer skipped.')
print('Originals:',ROOT/'raw')
''',True)
    md('''### Quality screening

The report distinguishes candidate, review and rejected frames. The Moon's illumination is context, not a rejection criterion. Measurements include native star profiles, full-field trails, noise, background, gradients, saturation and row/column patterns. Review and rejected frames stay on Drive and are excluded from automatic processing.

The supplied data contain lights only. Without matching darks/flats, fixed-pattern noise and illumination errors may remain. Inspect `QUALITY_REPORT.md`, `quality/frame_report.csv`, and the per-frame JPG review sheets. Manual decisions can be recorded in `quality/overrides.json` by relative raw path, with `status` and `reason`.''')
    code('''#@title 5. Screen the archive and summarize observing sessions
from quality import scan
from report import summarize
rows=scan(ROOT,workers=4)
summary=summarize(ROOT)
for r in summary:
    print(f"{r['night']} | {r['object']} | {r['filter']}: {r['candidate']}/{r['total']} candidates, "
          f"{r['review']} review, {r['reject']} rejected, {r['planetary_or_test']} planetary/test")
print('Report:',ROOT/'QUALITY_REPORT.md')
''',True)
    md('''### Integrate and render

The default **2×2 binning** produces presentation masters. Use **1** for native resolution or **4** for faster review. Registration must pass star-match, residual and scale checks. Different camera/gain/geometry groups are not silently mixed. At least three registered frames per channel are required. Sharp stars alone do not confirm pointing: after inspecting a mismatched field, set **REFERENCE_PATH** to the accepted exposure's relative raw path. Leave it blank for automatic selection. A blank observing night combines compatible archived sessions; actual contributions per night are recorded for each filter.

The screening quicklook is saved separately. Finishing restores the earlier B20 method: common sky anchors, star-excluded diffuse scales, wavelet noise reduction, global colour matrices, soft highlights and modest luminance contrast. Linear masters remain unchanged.

HOO and its amber/blue alternative both use only Hα and OIII; SHO requires measured SII as well. A single filter stays monochrome. RGB uses empirical unsaturated-star balance, not catalogue photometric calibration. Global display choices are recorded in `finished/render_report.json`. No reference-photo pixels, generated structure, local hue patches or replacement stars are added.''')
    code('''#@title 6. Process the chosen target on the GPU and save results
from process import process_target
if RUN_PROCESSING:
    result=process_target(ROOT,TARGET,night=OBSERVING_NIGHT or None,binning=int(BINNING),backend='gpu',reference_path=REFERENCE_PATH or None)
    print(json.dumps(result,indent=2))
    print('Results:',result['output'])
    preview=Path(result['output'])/'colour_quicklook.png'
    if FINISH_IMAGES:
        from finish import render
        presentation=render(result)
        print('Finished presentations:',presentation['output'])
        available=presentation['exports']
        preferred=next((p for p in ['SHO','HOO','RGB'] if p in available),next(iter(available)))
        preview=Path(presentation['output'])/(preferred+'.png')
    if DISPLAY_IMAGES:
        from IPython.display import display,Image
        display(Image(filename=str(preview),width=1000))
else:
    print('Archive and quality report complete. Processing disabled in configuration.')
''',True)
    md('''Saved layout:

```
OMI_Archive/
  raw/                       original Dropbox paths and bytes
  receipts/                  per-file SHA-256 and transfer receipts
  quality/                   measurements, decisions, caches and review sheets
  processed/<target>/<run>/  linear FITS masters, PNG, 16-bit TIFF, coverage, provenance
    finished/                presentation PNG/TIFF, previews and rendering parameters
  workflow/                  exact scripts and environment used
  QUALITY_REPORT.md
```

Keep the shared-link manifests private. Colab is not an always-on scheduler. Background jobs stop if its runtime is reclaimed; completed originals and reports remain in Drive.

[Dropbox folder downloads](https://help.dropbox.com/sync/download-entire-folders) · [Colab runtime limits](https://research.google.com/colaboratory/faq.html)''')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'OMI Dropbox Pipeline.ipynb','provenance':[]},'kernelspec':{'display_name':'Python 3','name':'python3'},'accelerator':'GPU'},'cells':cells}
    for i,c in enumerate(cells):c['id']='omi-'+str(i)
    dest=Path(destination) if destination else HERE.parent/'OMI_Dropbox_Pipeline.ipynb'
    dest.write_text(json.dumps(nb,indent=1)+'\n')
    return dest


if __name__=='__main__':print(build())
