"""Numerical safeguards for reference-star colour calibration."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from calibrate_nina_colour import (
    aperture_flux,
    fit_colour,
    response_values,
    spectral_photons,
)


def test_aperture_removes_sloping_sky_and_preserves_channel_ratio():
    yy, xx = np.indices((160, 160), dtype=float)
    star = 100 * np.exp(-((xx - 80.4) ** 2 + (yy - 79.7) ** 2) / (2 * 3**2))
    rng = np.random.default_rng(19)
    sky = 120 + 0.2 * xx - 0.1 * yy + rng.normal(0, 0.01, xx.shape)
    x, y = (np.array([80.4, 79.7]) - 1.5) / 4
    red = aperture_flux(star * 0.7 + sky, x, y, factor=4)
    green = aperture_flux(star + sky, x, y, factor=4)
    assert red[0] / green[0] == pytest.approx(0.7, abs=1e-4)
    assert green[0] == pytest.approx(100 * 2 * np.pi * 9, abs=2.0)
    assert green[1] == pytest.approx(1, abs=1e-3)


def test_robust_fit_recovers_white_point_despite_colour_biased_population():
    rng = np.random.default_rng(4)
    x = np.r_[np.linspace(-0.3, 0.3, 30), np.linspace(0.3, 1, 70)]
    y = 2.5 * np.log10(1.5) + 0.92 * x + rng.normal(0, 0.01, len(x))
    y[::13] += 0.4
    fitted = fit_colour(x, y)
    assert fitted["gain"] == pytest.approx(1.5, rel=0.015)
    assert fitted["slope"] == pytest.approx(0.92, abs=0.015)
    # Neutralizing the average red-biased field would produce the wrong gain.
    assert abs(10 ** (np.median(y) / 2.5) - 1.5) > 0.4


def test_fit_refuses_extrapolation_or_too_few_stars():
    with pytest.raises(ValueError, match="eight stars"):
        fit_colour(np.linspace(-0.4, 0.4, 7), np.zeros(7))
    with pytest.raises(ValueError, match="white-reference"):
        fit_colour(np.linspace(0.1, 0.8, 20), np.zeros(20))


def test_spectral_photometry_converts_energy_to_photons():
    wl = np.arange(400.0, 701.0)
    response = np.ones_like(wl)
    actual = spectral_photons(wl, np.ones_like(wl) * 3, response)
    assert actual == pytest.approx(1.5 * (700**2 - 400**2))
    record = {
        "wavelength": {"value": [4000, 7000], "units": "angstrom"},
        "values": {"value": [100, 100], "range": 100},
    }
    np.testing.assert_allclose(response_values(record, wl), 1.0)
