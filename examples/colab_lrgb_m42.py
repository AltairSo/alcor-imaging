"""HDR LRGB processing for the 653-minute Gemini-C5 M42 master set."""

from pathlib import Path

import matplotlib.pyplot as plt

import alcor_imaging as ai

light_dir = Path("/content/drive/MyDrive/Astro/M42/lights")
output_dir = Path("/content/drive/MyDrive/Astro/M42/output")
output_dir.mkdir(parents=True, exist_ok=True)

light_paths = sorted(
    path
    for path in light_dir.iterdir()
    if path.suffix.lower() in {".fits", ".fit", ".fts"}
)
if not light_paths:
    raise FileNotFoundError(f"No FITS/FIT/FTS files found in {light_dir}")

filter_alias = {"Lum": "L", "Red": "R", "Green": "G", "Blue": "B"}
sub_counts = {
    ("L", 60): 60,
    ("L", 120): 29,
    ("L", 300): 53,
    ("R", 60): 20,
    ("R", 120): 10,
    ("R", 300): 24,
    ("G", 60): 18,
    ("G", 120): 9,
    ("G", 300): 4,
    ("B", 60): 18,
    ("B", 120): 8,
    ("B", 300): 4,
}

channels = {channel: [] for channel in "LRGB"}
weights = {channel: [] for channel in "LRGB"}
for path in light_paths:
    frame = ai.read_fits(path)
    channel = filter_alias.get(str(frame.header.get("FILTER")))
    exposure = round(float(frame.header.get("EXPTIME", 0)))
    if channel is None or (channel, exposure) not in sub_counts:
        raise ValueError(
            f"Unexpected FILTER/EXPTIME in {path.name}: "
            f"{frame.header.get('FILTER')!r}/{exposure}"
        )
    channels[channel].append(path)
    weights[channel].append(sub_counts[channel, exposure] * exposure)

print("Library:", ai.__version__)
print("Masters:", {channel: len(paths) for channel, paths in channels.items()})
print("Integration seconds:", weights)

result = ai.process_lrgb(
    channels,
    weights=weights,
    config=ai.LRGBConfig(
        registration=ai.RegistrationConfig(
            downsample=4,
            detection_sigma=3.0,
            min_area=5,
        ),
        # Explicit calibration for this filter/camera/data set. These values are
        # deliberately notebook inputs rather than guesses inside the library.
        white_balance=(0.58, 1.0, 1.35),
        render=ai.RenderConfig(
            background_percentile=20.0,
            white_percentile=99.9,
            faint_strength=35.0,
            highlight_strength=6.0,
            core_start=0.08,
            core_end=0.65,
            shadow_knee=0.0015,
            gamma=0.88,
            saturation=0.90,
            highlight_knee=0.82,
        ),
        luminance_weight=0.60,
        luminance_ratio_limits=(0.7, 1.45),
        luminance_highlight_range=(0.5, 0.82),
        denoise_strength=0.08,
        saturation=1.0,
    ),
)

print("Accepted:", result.accepted_indices)
print("Rejected:", result.rejected_indices)
print("Final shape:", result.rgb.shape)
print(
    "Pixels clipped in every luminance exposure:",
    int(result.luminance_unrecoverable_mask.sum()),
)

ai.write_fits(
    output_dir / "M42_LRGB_linear_RGB.fits",
    result.linear_rgb,
    overwrite=True,
    rgb_axis=-1,
)
ai.write_fits(
    output_dir / "M42_HDR_luminance.fits",
    result.linear_luminance,
    overwrite=True,
)
ai.write_tiff(output_dir / "M42_LRGB_16bit.tif", result.rgb, bits=16)
ai.write_png(output_dir / "M42_LRGB_preview.png", result.rgb)

plt.figure(figsize=(18, 14))
plt.imshow(result.rgb)
plt.axis("off")
plt.tight_layout()
plt.show()
