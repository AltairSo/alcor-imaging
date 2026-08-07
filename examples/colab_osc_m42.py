"""One-shot-color M42 workflow. File acquisition remains in the notebook."""

from pathlib import Path

import alcor_imaging as ai

light_dir = Path("/content/drive/MyDrive/Astro/M42/lights")
light_paths = sorted(
    [
        path
        for path in light_dir.iterdir()
        if path.suffix.lower() in {".fits", ".fit", ".fts"}
    ]
)
if not light_paths:
    raise FileNotFoundError(f"No FITS/FIT/FTS files found in {light_dir}")

sample = ai.read_fits(light_paths[0])
detected_pattern = ai.infer_bayer_pattern(sample.header)
print("Frames:", len(light_paths))
print("Shape:", sample.data.shape)
print("Bayer pattern:", detected_pattern)
print("Filter:", sample.header.get("FILTER"))

if detected_pattern is None:
    raise ValueError(
        "No Bayer pattern is recorded. Confirm the camera CFA before setting "
        "OSCConfig(bayer_pattern=...). If this is a mono camera, use mono/LRGB processing."
    )

result = ai.process_osc(
    light_paths,
    config=ai.OSCConfig(
        registration=ai.RegistrationConfig(
            downsample=2,
            detection_sigma=3.0,
            min_area=5,
        ),
        stacking=ai.StackConfig(
            method="sigma_clip_median",
            sigma=3.0,
            normalization="multiplicative",
            tile_size=512,
        ),
        stretch=ai.StretchConfig(
            black_percentile=0.2,
            white_percentile=99.995,
            asinh_strength=12.0,
            shadow_protection=0.006,
            gamma=0.95,
        ),
        # Keep this disabled for large extended nebulosity unless you have inspected
        # the background model and chosen a mesh much larger than the target structure.
        background_box_size=None,
        white_balance=(1.25, 1.0, 1.15),
        denoise_strength=0.15,
        saturation=1.15,
        highlight_knee=0.78,
    ),
)

print("Accepted:", len(result.accepted_indices), result.accepted_indices)
print("Rejected:", len(result.rejected_indices), result.rejected_indices)
for record in result.registrations:
    if not record.accepted:
        print("Rejected frame", record.index, record.error)

ai.write_fits("M42_OSC_linear_RGB.fits", result.linear_rgb, overwrite=True, rgb_axis=-1)
ai.write_tiff("M42_OSC_16bit.tif", result.rgb, bits=16)
ai.write_png("M42_OSC_preview.png", result.rgb)

