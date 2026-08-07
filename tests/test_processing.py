import numpy as np
import pytest

from alcor_imaging import (
    StackConfig,
    StretchConfig,
    adjust_saturation,
    apply_palette,
    crop_to_overlap,
    normalize,
    quantize,
    stack,
    stretch,
)


def test_sigma_clipped_stack_is_nan_aware() -> None:
    frames = [
        np.asarray([[1.0, np.nan], [2.0, 2.0]]),
        np.asarray([[1.0, 3.0], [2.0, 2.0]]),
        np.asarray([[50.0, 3.0], [2.0, 2.0]]),
    ]
    result = stack(frames, StackConfig(method="sigma_clip_median"))
    np.testing.assert_allclose(result, [[1.0, 3.0], [2.0, 2.0]])


def test_overlap_crop_uses_pixels_valid_in_every_frame() -> None:
    first = np.ones((5, 6), dtype=np.float32)
    second = first.copy()
    first[0, :] = np.nan
    second[:, -1] = np.nan
    cropped = crop_to_overlap((first, second))
    assert [image.shape for image in cropped] == [(4, 5), (4, 5)]
    assert all(np.all(np.isfinite(image)) for image in cropped)


def test_hoo_palette_mapping() -> None:
    ha = np.full((2, 2), 0.8)
    oiii = np.full((2, 2), 0.4)
    rgb = apply_palette((ha, oiii), "HOO")
    np.testing.assert_allclose(rgb[..., 0], 0.8)
    np.testing.assert_allclose(rgb[..., 1], 0.22 * 0.8 + 0.78 * 0.4)
    np.testing.assert_allclose(rgb[..., 2], 0.4)


def test_stretch_is_bounded_and_non_destructive() -> None:
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    original = image.copy()
    result = stretch(image, StretchConfig(black_percentile=0, white_percentile=100))
    assert result.dtype == np.float32
    assert result.min() >= 0 and result.max() <= 1
    np.testing.assert_array_equal(image, original)


def test_validation_and_quantization() -> None:
    with pytest.raises(ValueError, match="below"):
        normalize(np.ones((2, 2)), black_percentile=90, white_percentile=10)
    np.testing.assert_array_equal(quantize([[0.0, 0.5, 1.0]], bits=8), [[0, 128, 255]])


def test_zero_saturation_produces_gray_rgb() -> None:
    rgb = np.asarray([[[1.0, 0.3, 0.1]]], dtype=np.float32)
    gray = adjust_saturation(rgb, 0.0)
    np.testing.assert_allclose(gray[..., 0], gray[..., 1])
    np.testing.assert_allclose(gray[..., 1], gray[..., 2])

