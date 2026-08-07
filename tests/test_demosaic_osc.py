import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, shift

from alcor_imaging import (
    OSCConfig,
    RegistrationConfig,
    StackConfig,
    StretchConfig,
    demosaic,
    infer_bayer_pattern,
    mosaic_rgb,
    process_osc,
    register_rgb_many,
)


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_malvar_demosaic_preserves_constant_rgb_interior(pattern: str) -> None:
    rgb = np.empty((32, 34, 3), dtype=np.float32)
    rgb[...] = (0.8, 0.4, 0.2)
    recovered = demosaic(mosaic_rgb(rgb, pattern), pattern)
    np.testing.assert_allclose(recovered[4:-4, 4:-4], rgb[4:-4, 4:-4], atol=1e-6)


def test_bayer_pattern_header_inference() -> None:
    assert infer_bayer_pattern({"BAYERPAT": "'rggb'"}) == "RGGB"
    assert infer_bayer_pattern({"FILTER": "L"}) is None


def _star_rgb() -> np.ndarray:
    rng = np.random.default_rng(9)
    rgb = np.zeros((192, 192, 3), dtype=np.float32)
    colors = np.asarray((1.0, 0.65, 0.35), dtype=np.float32)
    for y, x, amplitude in zip(
        rng.integers(15, 177, 35),
        rng.integers(15, 177, 35),
        rng.uniform(100, 1000, 35),
        strict=True,
    ):
        rgb[y, x] = amplitude * colors
    return gaussian_filter(rgb, sigma=(1.2, 1.2, 0)).astype(np.float32)


def test_rgb_registration_uses_one_transform_for_all_channels() -> None:
    reference = _star_rgb()
    source = shift(reference, (4, -6, 0), order=3, mode="constant", cval=0)
    aligned, records = register_rgb_many(
        (reference, source),
        config=RegistrationConfig(downsample=1, detection_sigma=3.0, min_area=3),
        on_error="raise",
    )
    assert len(aligned) == 2
    np.testing.assert_allclose(records[1].translation, (6, -4), atol=0.03)
    valid = np.all(np.isfinite(aligned[1]), axis=-1)
    assert np.mean(np.abs(aligned[1][valid] - reference[valid])) < 0.02


def test_osc_pipeline_produces_color_and_preserves_linear_master() -> None:
    rgb = _star_rgb()
    raw = mosaic_rgb(rgb, "RGGB")
    result = process_osc(
        (raw,),
        config=OSCConfig(
            bayer_pattern="RGGB",
            stacking=StackConfig(method="median", tile_size=64),
            stretch=StretchConfig(
                black_percentile=0,
                white_percentile=99.99,
                asinh_strength=8,
            ),
            white_balance=(1.1, 1.0, 0.9),
            denoise_strength=0,
            saturation=1.1,
        ),
    )
    assert result.bayer_pattern == "RGGB"
    assert result.linear_rgb.shape == rgb.shape
    assert result.rgb.shape == rgb.shape
    assert result.rgb.dtype == np.float32
    assert np.max(result.rgb) <= 1.0
    assert not np.allclose(result.rgb[..., 0], result.rgb[..., 2])


def test_osc_pipeline_requires_confirmed_bayer_pattern_for_arrays() -> None:
    with pytest.raises(ValueError, match="No Bayer pattern"):
        process_osc((np.ones((16, 16), dtype=np.float32),))

