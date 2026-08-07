from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import median_filter, zoom

from ._validation import FloatImage, as_float_image, finite_values


def repair_nonfinite(image: ArrayLike, *, value: float | None = None) -> FloatImage:
    """Replace NaN/inf pixels with a caller value or the finite-pixel median."""
    result = as_float_image(image, copy=True)
    finite = np.isfinite(result)
    if not np.any(finite):
        raise ValueError("Image contains no finite pixels.")
    fill = float(np.median(result[finite])) if value is None else float(value)
    result[~finite] = fill
    return result


def estimate_background(image: ArrayLike, *, box_size: int = 64, smooth: int = 3) -> FloatImage:
    """Estimate large-scale background using a robust median mesh.

    This intentionally avoids star-model assumptions. Callers processing crowded
    fields can increase ``box_size`` or provide their own background model.
    """
    data = repair_nonfinite(image)
    if box_size < 4:
        raise ValueError("box_size must be at least 4 pixels.")
    height, width = data.shape
    rows = max(1, int(np.ceil(height / box_size)))
    cols = max(1, int(np.ceil(width / box_size)))
    mesh = np.empty((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            tile = data[
                row * box_size : min((row + 1) * box_size, height),
                col * box_size : min((col + 1) * box_size, width),
            ]
            mesh[row, col] = np.median(tile)
    if smooth > 1 and min(mesh.shape) > 1:
        mesh = median_filter(mesh, size=min(smooth, min(mesh.shape)), mode="nearest")
    model = zoom(mesh, (height / rows, width / cols), order=3 if min(mesh.shape) > 3 else 1)
    return model[:height, :width].astype(np.float32)


def subtract_background(
    image: ArrayLike,
    background: ArrayLike | None = None,
    *,
    box_size: int = 64,
    preserve_level: bool = False,
) -> FloatImage:
    data = as_float_image(image)
    model = (
        estimate_background(data, box_size=box_size)
        if background is None
        else as_float_image(background)
    )
    if model.shape != data.shape:
        raise ValueError("Background model must match image shape.")
    level = float(np.median(finite_values(model))) if preserve_level else 0.0
    return (data - model + level).astype(np.float32)
