# OMI Dropbox → Colab → Drive

Open [`../OMI_Dropbox_Pipeline.ipynb`](../OMI_Dropbox_Pipeline.ipynb) in Colab with a CUDA GPU runtime. It contains its workflow sources and pins the Alcor core commit. Raw image work runs in Colab, not on the laptop.

For a new observing night, set **New night folder**, paste that folder's Dropbox shared link, enter a date label, choose the FITS `OBJECT` and observing night, and run all cells. A Dropbox account, app secret or personal copy of OMI's archive is unnecessary when the shared link permits downloads. Existing private per-file manifests can also be resumed. The notebook is on demand; it is not a permanent watcher or scheduler.

## What it saves

- Original folder structure and bytes under `raw/`, including unusable or damaged source files.
- SHA-256 receipts, retryable transfer results and private source manifests.
- Per-frame noise, sky, gradients, saturation, banding and stellar morphology; native-pixel review sheets and small linear caches.
- Candidate/review/reject decisions. Review and rejected frames are excluded by default and never deleted.
- GPU-registered, sigma-clipped linear FITS masters, colour PNG and 16-bit TIFF quicklooks, coverage masks, transforms, weights and processing provenance.
- Exact scripts and an environment snapshot in Drive.

The default review products use 4×4 binning. Select binning 1 for native-resolution masters. Objects are selected from FITS headers, and incompatible camera/gain/offset/binning/geometry groups are refused. Registration validates matched-star count, residual and scale, including across filters or nights. At least three registered exposures are needed for a channel master.

## Quality decisions

Five native sensor samples measure PSF shape without filtering elongated stars out first. A separate full-field detector catches long arcs and trails that sparse samples can miss. Hot pixels are excluded from stellar measurements using source footprint and integrated-flux criteria. Synthetic tests check both round stars mixed with hot pixels and severe tracking trails.

Thresholds are conservative screening heuristics, not universal scientific quality guarantees. Read the report and inspect diagnostic sheets before calling a result production quality. Moon illumination is computed from FITS timestamps for context and never rejects a frame by itself. Bright sky, focus, tracking, clouds, saturation and detector problems must be distinguished.

Manual decisions live in `quality/overrides.json`, keyed by relative raw path:

```json
{"2026-01-01/Target/LIGHT/Ha/frame.fits": {"status": "accepted", "reason": "Reviewed native star profiles and pattern test"}}
```

Use `reject` to exclude a frame explicitly. Preserve the explanation.

## Limits

The workflow does not invent calibration frames. Current supplied lights are not dark/flat calibrated, so fixed-pattern noise, halos and illumination errors can remain. It subtracts only scalar sky offsets, avoiding an automatic fit that might erase extended nebulae.

SHO/HOO are explicitly mapped colour palettes. RGB is an approximate preview until reference-star calibration is performed. Display gains are global and recorded. There is no external-image blending, generative detail, star replacement, or local hue patch mapping.

The folder route stages a ZIP on temporary Colab disk. Use individual night folders rather than a huge root archive. Repeated imports refresh the shared folder snapshot so newly arrived exposures are included. Completed ZIP staging files are removed after the originals and receipts are safely archived. An interrupted extraction can explicitly reuse its existing ZIP with `refresh=False`; normal runs always refresh. Per-file imports retry incomplete files and verify receipts on resume. Existing originals with different bytes are never overwritten. Shared-link manifests contain access-bearing URLs and must stay private.

The modules are deliberately examples/orchestration, so the Alcor core remains independent of Dropbox, Drive and credential management.

Rebuild the embedded notebook after changing sources:

```bash
python examples/omi_workflow/build_notebook.py
```

Validation (run in Colab):

```bash
python -m pytest -q examples/omi_workflow/test_workflow.py
```

References: [Dropbox direct downloads](https://help.dropbox.com/share/force-download), [folder-download limits](https://help.dropbox.com/sync/download-entire-folders), [Colab limits](https://research.google.com/colaboratory/faq.html), [Astropy ephemerides](https://docs.astropy.org/en/stable/coordinates/solarsystem.html).
