from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import affine_transform as scipy_affine_transform

from .backend import Backend, get_array_module, resolve_backend, to_host


def _validate_matrix(matrix: ArrayLike) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float32)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("Every transform must be a finite 3-by-3 homogeneous matrix.")
    if abs(float(np.linalg.det(result))) < 1e-12:
        raise ValueError("Transform matrix is singular.")
    return result


def _ndimage_inverse_matrix(
    forward_xy: np.ndarray, *, x_offset: int = 0, y_offset: int = 0
) -> np.ndarray:
    """Convert source-XY -> canvas-XY into tile-row/col -> source-row/col."""
    tile_to_canvas = np.asarray(
        [[1.0, 0.0, x_offset], [0.0, 1.0, y_offset], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    inverse_xy = np.linalg.inv(forward_xy).astype(np.float32) @ tile_to_canvas
    swap = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    return (swap @ inverse_xy @ swap).astype(np.float32)


def _affine_device(
    image: Any,
    matrix: np.ndarray,
    output_shape: tuple[int, int],
    *,
    backend: str,
    order: int,
    cval: float,
    x_offset: int = 0,
    y_offset: int = 0,
) -> Any:
    inverse = _ndimage_inverse_matrix(matrix, x_offset=x_offset, y_offset=y_offset)
    if backend == "gpu":
        import cupy as cp
        from cupyx.scipy.ndimage import affine_transform as cupy_affine_transform

        inverse_device = cp.asarray(inverse)
        kwargs = {"texture_memory": True} if order in {0, 1} else {}
        return cupy_affine_transform(
            image,
            inverse_device,
            output_shape=output_shape,
            order=order,
            mode="constant",
            cval=cval,
            prefilter=order > 1,
            **kwargs,
        )
    return scipy_affine_transform(
        image,
        inverse,
        output_shape=output_shape,
        order=order,
        mode="constant",
        cval=cval,
        prefilter=order > 1,
    )


def _warp_device(
    image: Any,
    matrix: np.ndarray,
    output_shape: tuple[int, int],
    *,
    backend: str,
    order: int,
    cval: float,
    x_offset: int = 0,
    y_offset: int = 0,
) -> Any:
    if image.ndim == 2:
        return _affine_device(
            image,
            matrix,
            output_shape,
            backend=backend,
            order=order,
            cval=cval,
            x_offset=x_offset,
            y_offset=y_offset,
        )
    channels = [
        _affine_device(
            image[..., index],
            matrix,
            output_shape,
            backend=backend,
            order=order,
            cval=cval,
            x_offset=x_offset,
            y_offset=y_offset,
        )
        for index in range(image.shape[-1])
    ]
    xp = get_array_module(backend)  # already resolved; does not transfer data
    return xp.stack(channels, axis=-1)


def warp_affine(
    image: ArrayLike,
    transform: ArrayLike,
    output_shape: tuple[int, int],
    *,
    backend: Backend = "auto",
    order: int = 1,
    fill_value: float = np.nan,
    return_footprint: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Warp a mono or channel-last image with a source-to-output transform.

    ``transform`` follows the conventional Cartesian ``(x, y)`` homogeneous
    convention used by ``skimage.transform.SimilarityTransform.params``. GPU
    execution uses CUDA texture-memory interpolation for float32 order 0/1 warps.
    Results are returned as NumPy arrays so the public API remains backend-neutral.
    """
    if len(output_shape) != 2 or min(output_shape) < 1:
        raise ValueError("output_shape must contain two positive integers.")
    if order not in {0, 1, 2, 3, 4, 5}:
        raise ValueError("order must be between 0 and 5.")
    selected = resolve_backend(backend)
    xp = get_array_module(selected)
    source = np.asarray(image, dtype=np.float32)
    if source.ndim not in {2, 3} or (source.ndim == 3 and source.shape[-1] < 1):
        raise ValueError("image must be mono or channel-last.")
    matrix = _validate_matrix(transform)
    finite = np.all(np.isfinite(source), axis=-1) if source.ndim == 3 else np.isfinite(source)
    clean = (
        np.where(finite[..., None], source, 0.0)
        if source.ndim == 3
        else np.where(finite, source, 0.0)
    )
    device = xp.asarray(clean, dtype=xp.float32)
    warped = _warp_device(
        device,
        matrix,
        tuple(int(value) for value in output_shape),
        backend=selected,
        order=order,
        cval=0.0,
    )
    validity = _affine_device(
        xp.asarray(finite, dtype=xp.float32),
        matrix,
        tuple(int(value) for value in output_shape),
        backend=selected,
        order=1,
        cval=0.0,
    )
    footprint_device = validity < (1.0 - 1e-5)
    if warped.ndim == 3:
        warped = xp.where(footprint_device[..., None], fill_value, warped)
    else:
        warped = xp.where(footprint_device, fill_value, warped)
    result = np.asarray(to_host(warped), dtype=np.float32)
    if not return_footprint:
        return result
    footprint = np.asarray(to_host(footprint_device), dtype=bool)
    return result, footprint


def _edge_weight(shape: tuple[int, int], feather_width: float, xp: Any) -> Any:
    if feather_width <= 0:
        return xp.ones(shape, dtype=xp.float32)
    rows = xp.arange(shape[0], dtype=xp.float32)
    cols = xp.arange(shape[1], dtype=xp.float32)
    row_distance = xp.minimum(rows + 1, shape[0] - rows)[:, None]
    col_distance = xp.minimum(cols + 1, shape[1] - cols)[None, :]
    weight = xp.clip(xp.minimum(row_distance, col_distance) / feather_width, 0.0, 1.0)
    return weight * weight * (3.0 - 2.0 * weight)


def compose_mosaic(
    images: Sequence[ArrayLike],
    transforms: Sequence[ArrayLike],
    output_shape: tuple[int, int],
    *,
    backend: Backend = "auto",
    tile_size: int = 1024,
    feather_width: float = 256.0,
    panel_weights: Sequence[float] | None = None,
    panel_gains: Sequence[float] | None = None,
    panel_offsets: Sequence[float] | None = None,
    fill_value: float = np.nan,
    return_coverage: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compose caller-positioned panels with bounded host and GPU memory.

    Each transform maps its source panel into the requested output canvas in
    Cartesian ``(x, y)`` coordinates. Photometric gains and offsets are explicit:
    the library does not infer project-specific exposure or color corrections.
    Source panels remain on the GPU for throughput; output and blend working
    buffers are limited to one tile at a time.
    """
    if not images or len(images) != len(transforms):
        raise ValueError("images and transforms must have the same non-zero length.")
    if len(output_shape) != 2 or min(output_shape) < 1:
        raise ValueError("output_shape must contain two positive integers.")
    if tile_size < 1:
        raise ValueError("tile_size must be positive.")
    if feather_width < 0:
        raise ValueError("feather_width cannot be negative.")
    count = len(images)
    weights = (
        np.ones(count, dtype=np.float32)
        if panel_weights is None
        else np.asarray(panel_weights, dtype=np.float32)
    )
    gains = (
        np.ones(count, dtype=np.float32)
        if panel_gains is None
        else np.asarray(panel_gains, dtype=np.float32)
    )
    offsets = (
        np.zeros(count, dtype=np.float32)
        if panel_offsets is None
        else np.asarray(panel_offsets, dtype=np.float32)
    )
    parameter_arrays = (
        ("panel_weights", weights),
        ("panel_gains", gains),
        ("panel_offsets", offsets),
    )
    for name, values in parameter_arrays:
        if values.shape != (count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain one finite scalar per image.")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("panel_weights must be non-negative with at least one positive value.")

    prepared = [np.asarray(image, dtype=np.float32) for image in images]
    ndim = prepared[0].ndim
    channels = prepared[0].shape[-1] if ndim == 3 else None
    if ndim not in {2, 3}:
        raise ValueError("Mosaic panels must be mono or channel-last images.")
    if any(image.ndim != ndim or (ndim == 3 and image.shape[-1] != channels) for image in prepared):
        raise ValueError("All mosaic panels must have matching channel geometry.")
    matrices = [_validate_matrix(transform) for transform in transforms]
    selected = resolve_backend(backend)
    xp = get_array_module(selected)

    device_images = []
    device_validity = []
    device_feathers = []
    for image in prepared:
        finite = np.all(np.isfinite(image), axis=-1) if ndim == 3 else np.isfinite(image)
        clean = (
            np.where(finite[..., None], image, 0.0) if ndim == 3 else np.where(finite, image, 0.0)
        )
        device_images.append(xp.asarray(clean, dtype=xp.float32))
        device_validity.append(xp.asarray(finite, dtype=xp.float32))
        device_feathers.append(_edge_weight(image.shape[:2], feather_width, xp))

    spatial = tuple(int(value) for value in output_shape)
    result_shape = spatial + ((int(channels),) if channels is not None else ())
    result = np.full(result_shape, fill_value, dtype=np.float32)
    coverage = np.zeros(spatial, dtype=np.float32) if return_coverage else None
    tiles_x = (spatial[1] + tile_size - 1) // tile_size
    tiles_y = (spatial[0] + tile_size - 1) // tile_size
    total_tiles = tiles_x * tiles_y
    tile_index = 0

    for y0 in range(0, spatial[0], tile_size):
        for x0 in range(0, spatial[1], tile_size):
            y1 = min(y0 + tile_size, spatial[0])
            x1 = min(x0 + tile_size, spatial[1])
            tile_shape = (y1 - y0, x1 - x0)
            accumulator_shape = tile_shape + (() if ndim == 2 else (channels,))
            accumulator = xp.zeros(accumulator_shape, dtype=xp.float32)
            denominator = xp.zeros(tile_shape, dtype=xp.float32)
            for index, matrix in enumerate(matrices):
                warped = _warp_device(
                    device_images[index],
                    matrix,
                    tile_shape,
                    backend=selected,
                    order=1,
                    cval=0.0,
                    x_offset=x0,
                    y_offset=y0,
                )
                valid = _affine_device(
                    device_validity[index],
                    matrix,
                    tile_shape,
                    backend=selected,
                    order=1,
                    cval=0.0,
                    x_offset=x0,
                    y_offset=y0,
                )
                feather = _affine_device(
                    device_feathers[index],
                    matrix,
                    tile_shape,
                    backend=selected,
                    order=1,
                    cval=0.0,
                    x_offset=x0,
                    y_offset=y0,
                )
                blend = xp.where(valid >= (1.0 - 1e-5), xp.clip(feather, 0.0, 1.0), 0.0)
                blend *= float(weights[index])
                corrected = warped * float(gains[index]) + float(offsets[index])
                accumulator += corrected * (blend[..., None] if ndim == 3 else blend)
                denominator += blend
            valid_output = denominator > 1e-12
            if ndim == 3:
                safe_denominator = xp.where(valid_output, denominator, 1.0)
                tile = accumulator / safe_denominator[..., None]
                tile = xp.where(
                    valid_output[..., None],
                    tile,
                    xp.asarray(fill_value, dtype=xp.float32),
                )
            else:
                safe_denominator = xp.where(valid_output, denominator, 1.0)
                tile = accumulator / safe_denominator
                tile = xp.where(
                    valid_output,
                    tile,
                    xp.asarray(fill_value, dtype=xp.float32),
                )
            result[y0:y1, x0:x1] = np.asarray(to_host(tile), dtype=np.float32)
            if coverage is not None:
                coverage[y0:y1, x0:x1] = np.asarray(to_host(denominator), dtype=np.float32)
            tile_index += 1
            if progress is not None:
                progress(tile_index, total_tiles)
    return (result, coverage) if coverage is not None else result
