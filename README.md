# Alcor Imaging

Research-grade, array-first astronomical image processing for telescope data.
Alcor Imaging turns FITS/FIT/FTS frames into calibrated masters and high-resolution
mono or narrowband RGB output without owning your storage workflow.

The library contains no FTP client, Google Drive integration, downloader, directory
crawler, credential handling, or notebook state. You decide where files come from
and pass explicit paths or NumPy arrays.

## Install in Colab

Once this directory is pushed as its own GitHub repository:

```python
!pip install -q "alcor-imaging[notebook] @ git+https://github.com/iam-abbas/alcor-imaging.git"
```

For local development:

```bash
python -m pip install -e ".[dev,notebook]"
pytest
```

## Quick HOO workflow

```python
from pathlib import Path

import alcor_imaging as ai

# File discovery remains in your notebook, where it is visible and controllable.
ha_paths = sorted(Path("/content/data/Ha").glob("*.fits"))
oiii_paths = sorted(Path("/content/data/OIII").glob("*.fits"))

config = ai.NarrowbandConfig(
    registration=ai.RegistrationConfig(
        downsample=4,
        detection_sigma=3.0,
        max_control_points=150,
    ),
    stacking=ai.StackConfig(
        method="sigma_clip_median",
        sigma=3.0,
        max_iterations=5,
        normalization="multiplicative",
    ),
    stretch=ai.StretchConfig(
        black_percentile=0.8,
        white_percentile=99.93,
        asinh_strength=6.5,
        shadow_protection=0.014,
        gamma=0.9,
    ),
    palette="HOO",
    channel_boosts=(1.0, 1.2),
    background_box_size=128,
    denoise_strength=0.45,
    saturation=1.12,
)

result = ai.process_narrowband((ha_paths, oiii_paths), config=config)

# Explicit exports: scientific linear channels, 16-bit edit master, display preview.
ai.write_fits("Ha_master_linear.fits", result.linear_channels[0], overwrite=True)
ai.write_fits("OIII_master_linear.fits", result.linear_channels[1], overwrite=True)
ai.write_tiff("target_HOO_16bit.tif", result.rgb, bits=16)
ai.write_png("target_HOO_preview.png", result.rgb)

for channel_name, stack_result in zip(("Ha", "OIII"), result.stacks):
    print(channel_name, "accepted", stack_result.accepted_indices)
    print(channel_name, "rejected", stack_result.rejected_indices)
```

## Full-control API

Every pipeline stage is independently usable:

```python
import alcor_imaging as ai

frame = ai.read_fits("light_001.fts")
quality = ai.measure_frame(frame.data)

master_bias = ai.make_master(bias_arrays)
master_dark = ai.make_master(dark_arrays)
master_flat = ai.make_master(flat_arrays, normalize=True)

calibration = ai.CalibrationSet(
    bias=master_bias,
    dark=master_dark,
    flat=master_flat,
    dark_exposure=300.0,
)
calibrated = ai.calibrate(frame.data, calibration, exposure=300.0)

aligned, records = ai.register_many(calibrated_frames, reference_index=quality_reference)
master = ai.stack(aligned, ai.StackConfig(method="sigma_clip_mean"), weights=weights)
background = ai.estimate_background(master, box_size=128)
linear = ai.subtract_background(master, background)
display = ai.stretch(linear, ai.StretchConfig(asinh_strength=7.0))
```

Custom palettes are ordinary 3×N matrices, so any filter set can be mapped without
changing library code:

```python
matrix = [
    [1.00, 0.10, 0.00],  # red from SII, Ha, OIII
    [0.15, 0.85, 0.20],  # green
    [0.00, 0.10, 1.00],  # blue
]
rgb = ai.combine_channels((sii, ha, oiii), matrix)
```

## Scientific behavior

- Processing uses `float32`; no quantization occurs until an explicit export.
- FITS values are read with Astropy scaling and are otherwise unmodified.
- Invalid registration borders remain `NaN`, allowing NaN-aware stacking and safe
  common-overlap cropping.
- Integration is tiled by default, bounding the largest temporary data cube for
  high-resolution sensors; set `StackConfig(tile_size=None)` to integrate at once.
- Calibration supports bias subtraction, exposure-scaled dark subtraction, and
  normalized flat correction.
- Registration decisions and transform parameters are returned as data, never only
  printed to a notebook.
- Linear masters and stretched display images are separate outputs.
- PNG is intentionally 8-bit preview output. Use 16-bit/float TIFF or float FITS for
  downstream editing and measurement.

The automatic pipeline is a reproducible starting point, not a substitute for
instrument-specific calibration. Overscan correction, bad-pixel maps, mosaics,
deconvolution PSFs, photometric color calibration, WCS reprojection, and drizzle are
best added as explicit specialized modules once required by a dataset.

## Package layout

```text
src/alcor_imaging/
  fits.py          single-file FITS read/write
  calibration.py  bias, dark, flat calibration
  quality.py      frame statistics and reference-selection metrics
  registration.py star-based similarity registration
  stacking.py     robust frame integration
  background.py   background modeling and subtraction
  geometry.py     common-overlap handling
  stretch.py      normalization and nonlinear transfer functions
  color.py        palettes, channel mixing, luminance, saturation
  enhance.py      denoising, sharpening, local contrast
  export.py       explicit TIFF and PNG output
  pipeline.py     configurable HOO/SHO orchestration
```
