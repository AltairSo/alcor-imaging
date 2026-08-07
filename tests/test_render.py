import numpy as np

from alcor_imaging import (
    RenderConfig,
    dual_asinh_stretch_rgb,
    estimate_background_offsets,
    neutralize_background,
    render_rgb,
)


def _linear_rgb() -> np.ndarray:
    y, x = np.mgrid[-1:1:96j, -1:1:112j]
    glow = np.exp(-4.0 * (x * x + y * y))
    core = 50.0 * np.exp(-150.0 * (x * x + y * y))
    rgb = np.stack(
        (0.12 + 1.3 * glow + core, 0.08 + 0.8 * glow + core, 0.2 + glow + core),
        axis=-1,
    )
    return rgb.astype(np.float32)


def test_background_offsets_and_neutralization_are_per_channel() -> None:
    rgb = np.full((20, 24, 3), (10.0, 20.0, 30.0), dtype=np.float32)
    rgb[10:, 12:] += (1.0, 2.0, 3.0)
    offsets = estimate_background_offsets(rgb, percentile=20)
    assert offsets == (10.0, 20.0, 30.0)
    neutral = neutralize_background(rgb, offsets=offsets)
    np.testing.assert_array_equal(neutral[0, 0], 0)
    np.testing.assert_array_equal(neutral[-1, -1], (1.0, 2.0, 3.0))


def test_dual_stretch_is_bounded_and_preserves_linear_channel_ratios() -> None:
    rgb = _linear_rgb() - (0.12, 0.08, 0.2)
    rendered = dual_asinh_stretch_rgb(rgb, white_percentile=99.5, gamma=1.0)
    assert rendered.dtype == np.float32
    assert np.min(rendered) >= 0 and np.max(rendered) <= 1
    source_ratio = rgb[30, 30, 0] / rgb[30, 30, 1]
    output_ratio = rendered[30, 30, 0] / rendered[30, 30, 1]
    np.testing.assert_allclose(output_ratio, source_ratio, rtol=1e-5)


def test_render_rgb_reveals_faint_signal_without_clipping_highlights() -> None:
    rgb = _linear_rgb()
    rendered = render_rgb(
        rgb,
        RenderConfig(
            channel_gains=(0.8, 1.0, 1.1),
            white_percentile=99.5,
            saturation=0.9,
        ),
    )
    assert np.all(np.isfinite(rendered))
    assert rendered[35, 35].mean() > rendered[0, 0].mean()
    assert rendered[48, 56].max() <= 1.0


def test_render_can_skip_automatic_background_estimation() -> None:
    rgb = _linear_rgb()
    rendered = render_rgb(rgb, RenderConfig(background_percentile=None))
    assert rendered.shape == rgb.shape
