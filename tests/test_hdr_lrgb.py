import numpy as np
from scipy.ndimage import gaussian_filter

from alcor_imaging import (
    LRGBConfig,
    RegistrationConfig,
    StretchConfig,
    hdr_combine,
    process_lrgb,
)


def test_hdr_combine_uses_short_exposure_for_saturated_core() -> None:
    flux = np.full((48, 48), 2.0, dtype=np.float32)
    flux[22:26, 22:26] = 500.0
    exposures = (60.0, 300.0)
    images = [
        np.clip(flux * exposure + 200.0, 0, 65535).astype(np.float32)
        for exposure in exposures
    ]
    combined = hdr_combine(
        images,
        exposures,
        weights=(3600.0, 15900.0),
        saturation_levels=(65535.0, 65535.0),
        saturation_fraction=0.98,
        saturation_dilation=0,
        background_percentile=20.0,
    )
    np.testing.assert_allclose(combined[10, 10], 0.0, atol=1e-5)
    # Background subtraction leaves the source contrast: 500 - 2 counts/s.
    np.testing.assert_allclose(combined[23, 23], 498.0, rtol=1e-4)


def _mono_star_field() -> np.ndarray:
    rng = np.random.default_rng(23)
    image = np.zeros((192, 192), dtype=np.float32)
    for y, x, amplitude in zip(
        rng.integers(15, 177, 35),
        rng.integers(15, 177, 35),
        rng.uniform(100, 900, 35),
        strict=True,
    ):
        image[y, x] = amplitude
    return gaussian_filter(image, 1.2).astype(np.float32)


def test_lrgb_pipeline_builds_color_and_injects_luminance() -> None:
    stars = _mono_star_field()
    channels = {
        "Lum": (stars * 1.2 + 100,),
        "Red": (stars * 0.9 + 100,),
        "Green": (stars * 0.65 + 100,),
        "Blue": (stars * 0.4 + 100,),
    }
    exposures = {channel: (60.0,) for channel in channels}
    result = process_lrgb(
        channels,
        exposures=exposures,
        config=LRGBConfig(
            registration=RegistrationConfig(downsample=1, min_area=3),
            rgb_stretch=StretchConfig(black_percentile=0, white_percentile=99.99),
            luminance_stretch=StretchConfig(
                black_percentile=0, white_percentile=99.99
            ),
            saturation_level=65535,
            denoise_strength=0,
        ),
    )
    assert result.rgb.ndim == 3 and result.rgb.shape[-1] == 3
    assert result.linear_luminance.shape == result.rgb.shape[:2]
    assert np.all(np.isfinite(result.rgb))
    assert np.min(result.rgb) >= 0 and np.max(result.rgb) <= 1
    assert not np.allclose(result.rgb[..., 0], result.rgb[..., 2])
    assert set(result.accepted_indices) == {"L", "R", "G", "B"}
    assert result.luminance_unrecoverable_mask.shape == result.linear_luminance.shape
