"""Regression checks for the runnable Colab/NINA workflow."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from skimage.transform import SimilarityTransform

spec = importlib.util.spec_from_file_location(
    "nina_workflow", Path(__file__).parents[1] / "examples/process_nina_lrgb.py"
)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def test_filter_metadata_conflicts_fail_loudly():
    assert workflow.group_filter({"FILTER": "Red"}, "frame.fits") == "R"
    assert workflow.group_filter({}, "nina_G_20s.fits") == "G"
    with pytest.raises(ValueError, match="disagreement"):
        workflow.group_filter({"FILTER": "L"}, "nina_B_20s.fits")
    with pytest.raises(ValueError, match="Unsupported"):
        workflow.group_filter({"FILTER": "Ha"}, "nina_L_20s.fits")


def test_binned_transform_preserves_pixel_centres_with_rotation():
    transform = SimilarityTransform(rotation=0.03, translation=(3.7, -2.2))
    points = np.array([[12.0, 24.0], [100.0, 60.0]])
    scale = 4
    binned = SimilarityTransform(matrix=workflow.change_pixel_scale(transform.params, scale))
    np.testing.assert_allclose(
        binned(points) * scale + (scale - 1) / 2,
        transform(points * scale + (scale - 1) / 2),
        atol=1e-10,
    )


def test_linked_render_is_monotonic_and_tiling_preserves_colours():
    levels = np.geomspace(0.001, 1000, 1024).astype("float32").reshape(32, 32)
    masters = {c: levels * gain for c, gain in zip("RGBL", [1.0, 0.8, 0.6, 1.0], strict=True)}
    params = {
        "backgrounds_adu_per_sec": dict.fromkeys("RGBL", 0.0),
        "rgb_gains": [1.0, 1.0, 1.0],
        "white": 200.0,
        "l_scale": 0.8,
    }
    full = workflow.render_tile(masters, params)
    tiled = np.concatenate(
        [
            workflow.render_tile({c: a[i : i + 4] for c, a in masters.items()}, params)
            for i in range(0, 32, 4)
        ]
    )
    np.testing.assert_array_equal(full, tiled)
    assert np.all(np.diff(full[..., 0].ravel()) >= 0)
    np.testing.assert_allclose(full[..., 1] / full[..., 0], 0.8, atol=2e-7)
    assert np.isfinite(full).all() and full.min() >= 0 and full.max() <= 1
