import numpy as np
from scipy.ndimage import gaussian_filter, shift

from alcor_imaging import (
    NarrowbandConfig,
    RegistrationConfig,
    StackConfig,
    process_narrowband,
    register_image,
)


def synthetic_star_field(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((256, 256), dtype=np.float32)
    for y, x, amplitude in zip(
        rng.integers(20, 236, 40),
        rng.integers(20, 236, 40),
        rng.uniform(100, 1000, 40),
        strict=True,
    ):
        image[y, x] = amplitude
    return gaussian_filter(image, 1.2).astype(np.float32)


def test_registration_recovers_subpixel_translation() -> None:
    reference = synthetic_star_field()
    source = shift(reference, (5, -7), order=3, mode="constant", cval=0)
    aligned, transform, footprint = register_image(
        source,
        reference,
        RegistrationConfig(downsample=1, detection_sigma=3.0, min_area=3),
    )
    np.testing.assert_allclose(transform.translation, (7, -5), atol=0.02)
    assert abs(transform.rotation) < 1e-3
    assert np.mean(np.abs(aligned[~footprint] - reference[~footprint])) < 0.01


def test_array_first_hoo_pipeline_returns_linear_and_display_products() -> None:
    ha = synthetic_star_field()
    oiii = synthetic_star_field() * 0.65
    result = process_narrowband(
        ((ha,), (oiii,)),
        config=NarrowbandConfig(
            registration=RegistrationConfig(downsample=1, min_area=3),
            stacking=StackConfig(method="median", tile_size=64),
            palette="HOO",
            channel_boosts=(1.0, 1.1),
        ),
    )
    assert len(result.linear_channels) == 2
    assert result.rgb.shape[-1] == 3
    assert result.rgb.dtype == np.float32
    assert np.nanmin(result.rgb) >= 0.0
    assert np.nanmax(result.rgb) <= 1.0
