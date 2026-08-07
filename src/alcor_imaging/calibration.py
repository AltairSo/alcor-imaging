from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatImage, as_float_image, as_image_sequence, finite_values
from .models import CalibrationSet


def make_master(
    frames: Sequence[ArrayLike],
    *,
    method: str = "median",
    normalize: bool = False,
) -> FloatImage:
    """Combine calibration frames into a master bias, dark, or flat."""
    images = as_image_sequence(frames)
    cube = np.stack(images)
    if normalize:
        medians = np.nanmedian(cube, axis=(1, 2))
        if np.any(~np.isfinite(medians) | (np.abs(medians) < 1e-12)):
            raise ValueError("Cannot normalize a calibration frame with zero/invalid median.")
        cube = cube / medians[:, None, None]
    if method == "median":
        result = np.nanmedian(cube, axis=0)
    elif method == "mean":
        result = np.nanmean(cube, axis=0)
    else:
        raise ValueError("method must be 'median' or 'mean'.")
    return result.astype(np.float32)


def calibrate(
    light: ArrayLike,
    calibration: CalibrationSet,
    *,
    exposure: float | None = None,
    clip_flat_floor: float = 1e-3,
) -> FloatImage:
    """Apply master bias, exposure-scaled dark, and normalized flat correction."""
    result = as_float_image(light, copy=True)
    bias = as_float_image(calibration.bias) if calibration.bias is not None else None
    dark = as_float_image(calibration.dark) if calibration.dark is not None else None
    flat = as_float_image(calibration.flat) if calibration.flat is not None else None

    for name, image in (("bias", bias), ("dark", dark), ("flat", flat)):
        if image is not None and image.shape != result.shape:
            raise ValueError(
                f"{name} shape {image.shape} does not match light shape {result.shape}."
            )

    if bias is not None:
        result -= bias
    if dark is not None:
        scale = 1.0
        if calibration.dark_exposure is not None:
            if exposure is None or exposure <= 0 or calibration.dark_exposure <= 0:
                raise ValueError("Positive light and dark exposures are required for dark scaling.")
            scale = exposure / calibration.dark_exposure
        result -= dark * scale
    if flat is not None:
        normalized_flat = flat / np.median(finite_values(flat))
        valid = np.isfinite(normalized_flat) & (normalized_flat > clip_flat_floor)
        result = np.divide(result, normalized_flat, out=np.full_like(result, np.nan), where=valid)
    return result.astype(np.float32, copy=False)


def calibrate_many(
    lights: Sequence[ArrayLike],
    calibration: CalibrationSet,
    *,
    exposures: Sequence[float] | None = None,
) -> list[FloatImage]:
    if exposures is not None and len(exposures) != len(lights):
        raise ValueError("exposures must contain one value per light frame.")
    return [
        calibrate(light, calibration, exposure=None if exposures is None else exposures[index])
        for index, light in enumerate(lights)
    ]
