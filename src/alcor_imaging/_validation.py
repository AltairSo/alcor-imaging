from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatImage = NDArray[np.float32]


def as_float_image(image: ArrayLike, *, ndim: int | None = 2, copy: bool = False) -> FloatImage:
    """Return an image as float32, rejecting empty and dimensionally invalid arrays."""
    result = np.asarray(image, dtype=np.float32)
    if copy:
        result = result.copy()
    if result.size == 0:
        raise ValueError("Image must not be empty.")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"Expected a {ndim}D image, received shape {result.shape}.")
    return result


def as_image_sequence(images: Sequence[ArrayLike]) -> list[FloatImage]:
    if not images:
        raise ValueError("At least one image is required.")
    result = [as_float_image(image) for image in images]
    shape = result[0].shape
    if any(image.shape != shape for image in result[1:]):
        shapes = sorted({image.shape for image in result})
        raise ValueError(f"All images must have the same shape; received {shapes}.")
    return result


def finite_values(image: ArrayLike) -> NDArray[np.float32]:
    array = as_float_image(image, ndim=None)
    values = array[np.isfinite(array)]
    if values.size == 0:
        raise ValueError("Image contains no finite pixels.")
    return values


def validate_percentile(value: float, name: str) -> None:
    if not 0 <= value <= 100:
        raise ValueError(f"{name} must be between 0 and 100, received {value}.")

