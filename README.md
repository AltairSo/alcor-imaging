# Alcor Imaging

Research-grade, array-first astronomical image processing for telescope data.
Alcor Imaging turns FITS/FIT/FTS frames into calibrated mono, broadband,
narrowband, or arbitrary multispectral products without owning your storage workflow.

The library contains no FTP client, Google Drive integration, downloader, directory
crawler, credential handling, or notebook state. You decide where files come from
and pass explicit paths or NumPy arrays.

## Install in Colab

```python
%pip install -q \
  "alcor-imaging[notebook] @ git+https://github.com/AltairSo/alcor-imaging.git@main"
```

For local development:

```bash
python -m pip install -e ".[dev,notebook]"
pytest
```

To update an existing Colab installation:

```python
%pip install -q --upgrade --no-cache-dir \
  "alcor-imaging[notebook] @ git+https://github.com/AltairSo/alcor-imaging.git@main"
```

Restart the Colab session after installation or upgrade.

## Generic channel primitives

The core has no built-in concept of Ha, OIII, LRGB, an object name, a telescope,
or a directory layout. The caller supplies already-grouped sources and all metadata
required by the chosen integration method.

```python
ha = ai.integrate_mono_channel(
    ha_frames,
    mode="stack",
    registration=ai.RegistrationConfig(downsample=4),
    stacking=ai.StackConfig(method="sigma_clip_median"),
)

oiii = ai.integrate_mono_channel(
    oiii_frames,
    mode="stack",
    registration=ai.RegistrationConfig(downsample=4),
    stacking=ai.StackConfig(method="sigma_clip_median"),
)

aligned = ai.align_mono_masters(
    {"hydrogen": ha.master, "oxygen": oiii.master},
    reference="hydrogen",
)

rgb = ai.combine_channels(
    [aligned.masters["hydrogen"], aligned.masters["oxygen"]],
    matrix=((1.0, 0.0), (0.22, 0.78), (0.0, 1.0)),
)
```

For mixed-exposure inputs, select `mode="hdr"` and explicitly supply `exposures`,
`weights`, and `saturation_levels`. Convenience APIs such as `process_narrowband`,
`process_lrgb`, and `process_osc` compose these lower-level operations but do not
replace them.

## Mixed-exposure LRGB workflow

Mono-camera L/R/G/B masters are not Bayer data. Group them explicitly by filter and
use `process_lrgb`. Mixed 60s/120s/300s masters are converted to flux rate and merged
as HDR data: saturated long-exposure pixels are excluded, shorter exposures recover
bright structure, and integration-time weights preserve the depth of larger stacks.

```python
result = ai.process_lrgb(
    {"L": luminance_paths, "R": red_paths, "G": green_paths, "B": blue_paths},
    weights={
        "L": luminance_integration_seconds,
        "R": red_integration_seconds,
        "G": green_integration_seconds,
        "B": blue_integration_seconds,
    },
    config=ai.LRGBConfig(
        white_balance=(1.04, 1.0, 1.08),
        luminance_weight=0.72,
        highlight_knee=0.74,
    ),
)
```

The complete 653-minute M42 example, including its subframe-count weighting table,
is in `examples/colab_lrgb_m42.py`.

## One-shot-color / Bayer workflow

Do not send raw Bayer FITS frames through the mono workflow: that produces a
grayscale mosaic stack. `process_osc` calibrates raw CFA data before demosaicing,
registers all three channels with one transform, integrates them, and applies a
linked color-preserving stretch.

```python
import alcor_imaging as ai

sample = ai.read_fits(light_paths[0])
print("Bayer pattern:", ai.infer_bayer_pattern(sample.header))

result = ai.process_osc(
    light_paths,
    config=ai.OSCConfig(
        registration=ai.RegistrationConfig(downsample=2),
        stacking=ai.StackConfig(
            method="sigma_clip_median",
            normalization="multiplicative",
        ),
        stretch=ai.StretchConfig(
            black_percentile=0.2,
            white_percentile=99.995,
            asinh_strength=12.0,
            shadow_protection=0.006,
            gamma=0.95,
        ),
        background_box_size=None,  # Preserve large extended nebulosity such as M42.
        white_balance=(1.25, 1.0, 1.15),
        saturation=1.15,
        highlight_knee=0.78,
    ),
)

ai.write_fits("linear_RGB.fits", result.linear_rgb, overwrite=True, rgb_axis=-1)
ai.write_tiff("display_16bit.tif", result.rgb, bits=16)
ai.write_png("preview.png", result.rgb)
```

The Bayer pattern is read from `BAYERPAT`, `BAYERPATN`, `COLORTYP`, or `CFA`.
If none is present, confirm the camera pattern and set `bayer_pattern` explicitly.
Never guess it: a wrong pattern produces incorrect color and interpolation artifacts.
The complete Colab example is in `examples/colab_osc_m42.py`.

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

Custom palettes are ordinary 3-by-N matrices, so any filter set can be mapped without
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
  channels.py     generic mono integration and arbitrary master alignment
  demosaic.py     Bayer detection and Malvar/bilinear demosaicing
  quality.py      frame statistics and reference-selection metrics
  registration.py star-based similarity registration
  stacking.py     robust frame integration
  background.py   background modeling and subtraction
  geometry.py     common-overlap handling
  hdr.py          saturation-aware mixed-exposure integration
  stretch.py      normalization and nonlinear transfer functions
  color.py        palettes, channel mixing, luminance, saturation
  enhance.py      denoising, sharpening, local contrast
  export.py       explicit TIFF and PNG output
  pipeline.py     configurable HOO/SHO orchestration
```
