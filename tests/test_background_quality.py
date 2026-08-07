import numpy as np

from alcor_imaging import estimate_background, measure_frame, repair_nonfinite


def test_repair_nonfinite_uses_finite_median() -> None:
    image = np.asarray([[1.0, np.nan], [3.0, np.inf]], dtype=np.float32)
    repaired = repair_nonfinite(image)
    np.testing.assert_array_equal(repaired, [[1.0, 2.0], [3.0, 2.0]])


def test_background_model_tracks_smooth_gradient() -> None:
    y, x = np.indices((64, 64))
    gradient = (10 + 0.01 * x + 0.02 * y).astype(np.float32)
    model = estimate_background(gradient, box_size=16, smooth=1)
    assert model.shape == gradient.shape
    assert np.median(np.abs(model - gradient)) < 0.2


def test_frame_quality_detects_synthetic_stars() -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    for y, x in ((12, 15), (30, 40), (50, 20)):
        image[y, x] = 100.0
    metrics = measure_frame(image, detection_sigma=3.0)
    assert metrics.star_count >= 3
    assert metrics.sharpness > 0

