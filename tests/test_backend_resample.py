import sys
import types

import numpy as np
import pytest
from scipy.ndimage import affine_transform, gaussian_filter

from alcor_imaging import (
    GPUUnavailableError,
    RenderConfig,
    backend_info,
    compose_mosaic,
    gpu_available,
    render_rgb,
    warp_affine,
)


@pytest.fixture
def simulated_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise CUDA orchestration with NumPy/SciPy standing in for CuPy kernels."""

    cupy = types.ModuleType("cupy")
    for name in (
        "abs",
        "arange",
        "arcsinh",
        "asarray",
        "clip",
        "divide",
        "einsum",
        "exp",
        "float32",
        "full_like",
        "isfinite",
        "max",
        "maximum",
        "minimum",
        "ndarray",
        "ones",
        "percentile",
        "power",
        "stack",
        "where",
        "zeros",
        "zeros_like",
    ):
        setattr(cupy, name, getattr(np, name))
    cupy.asnumpy = np.asarray

    class Pool:
        def free_all_blocks(self) -> None:
            pass

    cupy.get_default_memory_pool = Pool
    cupy.get_default_pinned_memory_pool = Pool
    runtime = types.SimpleNamespace(
        getDeviceCount=lambda: 1,
        getDeviceProperties=lambda _: {"name": b"Simulated CUDA"},
        memGetInfo=lambda: (4_000_000_000, 8_000_000_000),
    )
    cupy.cuda = types.SimpleNamespace(
        runtime=runtime,
        Device=lambda: types.SimpleNamespace(id=0),
    )

    cupyx = types.ModuleType("cupyx")
    cupyx_scipy = types.ModuleType("cupyx.scipy")
    cupyx_ndimage = types.ModuleType("cupyx.scipy.ndimage")

    def fake_affine(*args: object, **kwargs: object) -> np.ndarray:
        kwargs.pop("texture_memory", None)
        return affine_transform(*args, **kwargs)

    cupyx_ndimage.affine_transform = fake_affine
    cupyx_ndimage.gaussian_filter = gaussian_filter
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    monkeypatch.setitem(sys.modules, "cupyx", cupyx)
    monkeypatch.setitem(sys.modules, "cupyx.scipy", cupyx_scipy)
    monkeypatch.setitem(sys.modules, "cupyx.scipy.ndimage", cupyx_ndimage)


def test_cpu_backend_info_is_explicit() -> None:
    info = backend_info("cpu")
    assert info.selected == "cpu"
    assert info.requested == "cpu"


def test_explicit_gpu_reports_actionable_error_when_unavailable() -> None:
    if gpu_available():
        pytest.skip("CUDA is available on this test host.")
    with pytest.raises(GPUUnavailableError, match=r"GPU|CUDA|gpu"):
        backend_info("gpu")


def test_warp_affine_uses_cartesian_source_to_output_transform() -> None:
    source = np.zeros((7, 8), dtype=np.float32)
    source[2, 3] = 5.0
    transform = np.asarray(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    warped = warp_affine(
        source,
        transform,
        (9, 11),
        backend="cpu",
        order=0,
        fill_value=0.0,
    )
    assert warped[3, 5] == 5.0
    assert np.count_nonzero(warped) == 1


def test_compose_mosaic_is_tiled_and_supports_explicit_photometry() -> None:
    left = np.ones((4, 4), dtype=np.float32)
    right = np.ones((4, 4), dtype=np.float32)
    identity = np.eye(3, dtype=np.float32)
    translate = identity.copy()
    translate[0, 2] = 4
    mosaic, coverage = compose_mosaic(
        (left, right),
        (identity, translate),
        (4, 8),
        backend="cpu",
        tile_size=3,
        feather_width=0,
        panel_gains=(1.0, 2.0),
        return_coverage=True,
    )
    np.testing.assert_allclose(mosaic[:, :4], 1.0)
    np.testing.assert_allclose(mosaic[:, 4:], 2.0)
    np.testing.assert_allclose(coverage, 1.0)


def test_render_config_can_select_cpu_backend() -> None:
    rgb = np.ones((12, 13, 3), dtype=np.float32)
    rgb[4:8, 5:9] = (2.0, 1.5, 1.2)
    result = render_rgb(
        rgb,
        RenderConfig(background_percentile=None, backend="cpu", tile_size=8),
    )
    assert result.shape == rgb.shape
    assert result.dtype == np.float32


def test_gpu_warp_and_tiled_render_match_cpu_under_simulation(simulated_cuda: None) -> None:
    source = np.arange(12 * 13, dtype=np.float32).reshape(12, 13)
    transform = np.asarray(
        [[1.0, 0.0, 1.25], [0.0, 1.0, -0.75], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    cpu_warp = warp_affine(source, transform, (12, 13), backend="cpu")
    gpu_warp = warp_affine(source, transform, (12, 13), backend="gpu")
    np.testing.assert_allclose(gpu_warp, cpu_warp, equal_nan=True)

    identity = np.eye(3, dtype=np.float32)
    shift_right = identity.copy()
    shift_right[0, 2] = 8
    panels = (np.ones((10, 10), dtype=np.float32), np.full((10, 10), 2.0, np.float32))
    cpu_mosaic = compose_mosaic(
        panels, (identity, shift_right), (10, 18), backend="cpu", tile_size=7
    )
    gpu_mosaic = compose_mosaic(
        panels, (identity, shift_right), (10, 18), backend="gpu", tile_size=7
    )
    np.testing.assert_allclose(gpu_mosaic, cpu_mosaic, equal_nan=True)

    y, x = np.mgrid[-1:1:40j, -1:1:48j]
    glow = np.exp(-4 * (x * x + y * y)).astype(np.float32)
    rgb = np.stack((glow, glow * 0.8, glow * 0.6), axis=-1)
    config = RenderConfig(background_percentile=None, mask_blur_sigma=1.5)
    cpu_render = render_rgb(rgb, config, backend="cpu")
    gpu_render = render_rgb(rgb, config, backend="gpu", tile_size=17)
    np.testing.assert_allclose(gpu_render, cpu_render, atol=2e-5)
