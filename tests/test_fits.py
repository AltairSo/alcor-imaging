import numpy as np

from alcor_imaging import read_fits, write_fits


def test_fits_round_trip_preserves_float_data_and_metadata(tmp_path) -> None:
    path = tmp_path / "science.fts"
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    write_fits(path, data, header={"FILTER": "Ha", "EXPTIME": 300.0})
    frame = read_fits(path)
    np.testing.assert_array_equal(frame.data, data)
    assert frame.data.dtype == np.float32
    assert frame.header["FILTER"] == "Ha"
    assert frame.header["EXPTIME"] == 300.0


def test_rgb_fits_axis_conversion(tmp_path) -> None:
    path = tmp_path / "rgb.fits"
    rgb = np.zeros((3, 4, 3), dtype=np.float32)
    rgb[..., 1] = 0.5
    write_fits(path, rgb, rgb_axis=-1)
    frame = read_fits(path, plane=1)
    assert frame.data.shape == (3, 4)
    np.testing.assert_allclose(frame.data, 0.5)

