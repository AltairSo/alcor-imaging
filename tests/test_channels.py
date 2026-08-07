import numpy as np
from scipy.ndimage import gaussian_filter, shift

from alcor_imaging import (
    NarrowbandConfig,
    RegistrationConfig,
    StackConfig,
    StretchConfig,
    align_mono_masters,
    integrate_mono_channel,
    process_narrowband,
)


def _star_field() -> np.ndarray:
    rng = np.random.default_rng(101)
    image = np.zeros((192, 192), dtype=np.float32)
    for y, x, amplitude in zip(
        rng.integers(15, 177, 36),
        rng.integers(15, 177, 36),
        rng.uniform(100, 1000, 36),
        strict=True,
    ):
        image[y, x] = amplitude
    return gaussian_filter(image, 1.2).astype(np.float32)


def test_generic_mono_channel_integration_and_master_alignment() -> None:
    reference = _star_field()
    frames = (
        reference,
        shift(reference, (3, -5), order=3, mode="constant", cval=0),
        shift(reference, (-4, 2), order=3, mode="constant", cval=0),
    )
    registration = RegistrationConfig(downsample=1, detection_sigma=3, min_area=3)
    integrated = integrate_mono_channel(
        frames,
        registration=registration,
        stacking=StackConfig(method="sigma_clip_median", tile_size=64),
    )
    assert integrated.accepted_indices == [0, 1, 2]
    assert integrated.rejected_indices == []
    assert integrated.master.shape == reference.shape

    aligned = align_mono_masters(
        {
            "science-A": integrated.master,
            "science-B": shift(
                integrated.master, (2, -3), order=3, mode="constant", cval=0
            ),
        },
        reference="science-A",
        registration=registration,
        crop=True,
    )
    assert list(aligned.masters) == ["science-A", "science-B"]
    assert aligned.masters["science-A"].shape == aligned.masters["science-B"].shape
    np.testing.assert_allclose(
        aligned.registrations["science-B"].translation, (3, -2), atol=0.03
    )


def test_original_ha_oiii_individual_frame_workflow() -> None:
    base = _star_field()
    ha_frames = (
        base,
        shift(base, (3, -5), order=3, mode="constant", cval=0),
        shift(base, (-4, 2), order=3, mode="constant", cval=0),
    )
    oiii_base = base * 0.65
    oiii_frames = (
        oiii_base,
        shift(oiii_base, (2, -4), order=3, mode="constant", cval=0),
        shift(oiii_base, (-3, 3), order=3, mode="constant", cval=0),
    )
    result = process_narrowband(
        (ha_frames, oiii_frames),
        config=NarrowbandConfig(
            registration=RegistrationConfig(
                downsample=1, detection_sigma=3, min_area=3
            ),
            stacking=StackConfig(method="sigma_clip_median", tile_size=64),
            stretch=StretchConfig(black_percentile=0, white_percentile=99.99),
            palette="HOO",
            channel_boosts=(1.0, 1.2),
        ),
    )
    assert [stack.accepted_indices for stack in result.stacks] == [[0, 1, 2], [0, 1, 2]]
    assert result.rgb.shape[-1] == 3
    assert len(result.linear_channels) == 2


def test_narrowband_accepts_arbitrary_channel_count_and_mixing_matrix() -> None:
    base = _star_field()
    channels = tuple((base * scale,) for scale in (1.0, 0.8, 0.6, 0.4))
    result = process_narrowband(
        channels,
        config=NarrowbandConfig(
            registration=RegistrationConfig(downsample=1, min_area=3),
            stacking=StackConfig(method="median", tile_size=64),
            stretch=StretchConfig(black_percentile=0, white_percentile=99.99),
            mixing_matrix=(
                (1.0, 0.2, 0.0, 0.0),
                (0.0, 0.5, 0.5, 0.0),
                (0.0, 0.0, 0.2, 1.0),
            ),
        ),
    )
    assert len(result.linear_channels) == 4
    assert result.rgb.shape[-1] == 3

