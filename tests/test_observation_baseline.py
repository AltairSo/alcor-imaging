"""Check fidelity invariants of the observation-only display export."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
spec = importlib.util.spec_from_file_location(
    "baseline", EXAMPLES / "render_observation_baseline.py"
)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def test_baseline_is_pointwise_monotonic_and_ignores_luminance():
    ramp = np.geomspace(0.001, 10000, 1200).astype("float32").reshape(30, 40)
    masters = {c: ramp * g for c, g in zip("RGB", [1, 0.7, 0.4], strict=True)}
    params = {
        "rgb_gains": [1, 1, 1],
        "backgrounds_adu_per_sec": dict.fromkeys("RGB", 0),
        "white": 200,
    }
    full = baseline.baseline_tile(masters, params)
    assert np.all(np.diff(full.reshape(-1, 3), axis=0) > 0)
    with_l = baseline.baseline_tile({**masters, "L": ramp * 1e8}, params)
    np.testing.assert_array_equal(full, with_l)
    tiled = np.concatenate(
        [
            baseline.baseline_tile({c: a[i : i + 3] for c, a in masters.items()}, params)
            for i in range(0, 30, 3)
        ]
    )
    np.testing.assert_array_equal(full, tiled)
    # Undo only the documented encoding/pedestal to confirm linear hue ratios.
    encoded = (full - 0.035) / 0.965
    linear = np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)
    np.testing.assert_allclose(linear[..., 1] / linear[..., 0], 0.7, atol=2e-5)
    np.testing.assert_allclose(linear[..., 2] / linear[..., 0], 0.4, atol=2e-5)


def test_background_noise_survives_instead_of_being_clipped_at_zero():
    signal = np.array([[-0.01, 0, 0.01]], dtype="float32")
    params = {
        "rgb_gains": [1, 1, 1],
        "backgrounds_adu_per_sec": dict.fromkeys("RGB", 0),
        "white": 200,
    }
    result = baseline.baseline_tile(dict.fromkeys("RGB", signal), params)
    assert 0 < result[0, 0, 0] < result[0, 1, 0] < result[0, 2, 0]
    np.testing.assert_allclose(result[0, 1], 0.035)
    np.testing.assert_allclose(
        baseline.encode_srgb(np.array([0.0, 0.0031308, 1.0])), [0.0, 0.040449936, 1.0]
    )
