"""Clipped observations must not erase other channels or produce invented flux."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "examples"))
from render_natural_enhanced import classify_availability, natural_tile, reject_clipped


def test_each_channel_retains_available_samples_and_marks_all_missing():
    counts = {
        "R": np.array([[3, 2, 0, 0]], dtype=np.uint8),
        "G": np.array([[3, 3, 2, 0]], dtype=np.uint8),
        "B": np.array([[3, 1, 3, 0]], dtype=np.uint8),
    }
    np.testing.assert_array_equal(
        classify_availability(counts, dict.fromkeys("RGB", 3)), [[0, 1, 2, 3]]
    )
    cube = np.array([[[10, 65535, 65535]], [[12, 14, 65535]], [[11, 16, 65535]]], dtype=np.float32)
    flags = cube >= 65535
    combined, count = reject_clipped(cube, flags, np.array([[99, 99, 50000]]))
    np.testing.assert_array_equal(count, [[3, 2, 0]])
    np.testing.assert_array_equal(combined, [[11, 15, 50000]])
    # A missing R sample must not alter intact G measurements.
    intact = np.ones_like(cube) * 123
    other, other_count = reject_clipped(intact, np.zeros_like(flags), np.zeros((1, 3)))
    np.testing.assert_array_equal(other, [[123, 123, 123]])
    np.testing.assert_array_equal(other_count, [[3, 3, 3]])


def test_saturation_weight_turns_off_luminance_boost_without_recolouring():
    ramp = np.geomspace(0.001, 1000, 400).astype(np.float32).reshape(20, 20)
    p = {
        "rgb_gains": [1, 1, 1],
        "backgrounds_adu_per_sec": dict.fromkeys("LRGB", 0),
        "white": 200,
        "l_scale": 0.5,
    }
    m = {c: ramp * g for c, g in zip("RGBL", [1, 0.7, 0.5, 10], strict=True)}
    missing = natural_tile(m, p, np.zeros_like(ramp))
    changed = natural_tile({**m, "L": np.zeros_like(ramp)}, p, np.zeros_like(ramp))
    np.testing.assert_array_equal(missing, changed)
    safe = natural_tile(m, p, np.ones_like(ramp))
    assert np.isfinite(safe).all() and safe.min() >= 0 and safe.max() <= 1
    assert np.max(abs(safe - missing)) > 0.01
    tiled = np.concatenate(
        [
            natural_tile({c: a[i : i + 2] for c, a in m.items()}, p, np.ones_like(ramp[i : i + 2]))
            for i in range(0, 20, 2)
        ]
    )
    np.testing.assert_array_equal(safe, tiled)


def test_signed_background_noise_does_not_become_bright_colour_spikes():
    rng = np.random.default_rng(42)
    params = {
        "rgb_gains": [1, 1, 1],
        "backgrounds_adu_per_sec": dict.fromkeys("LRGB", 0),
        "white": 200,
        "l_scale": 1,
    }
    masters = {c: rng.normal(0, 0.4, (200, 200)).astype("float32") for c in "LRGB"}
    weight = np.ones((200, 200))
    baseline = natural_tile(masters, params, weight, mix=0)
    enhanced = natural_tile(masters, params, weight)
    faint = baseline.max(-1) < 0.2
    assert enhanced[faint].max() < 0.25
