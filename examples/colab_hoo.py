"""Minimal Colab-style HOO example. File acquisition intentionally omitted."""

from pathlib import Path

import alcor_imaging as ai

ha_paths = sorted(Path("/content/data/Ha").glob("*.fits"))
oiii_paths = sorted(Path("/content/data/OIII").glob("*.fits"))

result = ai.process_narrowband(
    (ha_paths, oiii_paths),
    config=ai.NarrowbandConfig(
        registration=ai.RegistrationConfig(downsample=4),
        stacking=ai.StackConfig(method="sigma_clip_median", normalization="multiplicative"),
        stretch=ai.StretchConfig(asinh_strength=6.5),
        palette="HOO",
        channel_boosts=(1.0, 1.2),
        denoise_strength=0.45,
        saturation=1.12,
    ),
)

ai.write_tiff("target_HOO_16bit.tif", result.rgb)
ai.write_png("target_HOO_preview.png", result.rgb)

