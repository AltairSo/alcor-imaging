import numpy as np
import pytest

from alcor_imaging import CalibrationSet, calibrate, make_master


def test_make_master_median_rejects_single_outlier() -> None:
    frames = [np.ones((4, 4)), np.ones((4, 4)), np.full((4, 4), 100.0)]
    np.testing.assert_allclose(make_master(frames), 1.0)


def test_calibrate_bias_dark_flat_and_exposure_scaling() -> None:
    shape = (3, 3)
    calibration = CalibrationSet(
        bias=np.full(shape, 10.0),
        dark=np.full(shape, 2.0),
        flat=np.full(shape, 0.5),
        dark_exposure=10.0,
    )
    result = calibrate(np.full(shape, 30.0), calibration, exposure=20.0)
    np.testing.assert_allclose(result, 16.0)


def test_calibrate_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        calibrate(np.ones((4, 4)), CalibrationSet(bias=np.ones((3, 3))))

