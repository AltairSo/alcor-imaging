from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import binary_dilation

from ._validation import FloatImage, as_image_sequence, validate_percentile


def _hdr_combine(
    images: Sequence[ArrayLike],
    exposures: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    saturation_levels: Sequence[float] | None = None,
    saturation_fraction: float = 0.98,
    saturation_dilation: int = 2,
    background_percentile: float = 20.0,
) -> tuple[FloatImage, np.ndarray]:
    """Combine aligned mixed-exposure masters into a linear flux-rate image.

    Each image has a robust scalar background removed and is divided by its
    individual-subframe exposure. Saturated pixels are excluded before weighting.
    ``weights`` should describe effective integration time when inputs are themselves
    master stacks; otherwise individual exposure times are a reasonable default.
    """
    prepared = as_image_sequence(images)
    exposure_array = np.asarray(exposures, dtype=np.float64)
    if exposure_array.shape != (len(prepared),) or np.any(exposure_array <= 0):
        raise ValueError("exposures must contain one positive value per image.")
    if weights is None:
        weight_array = exposure_array.copy()
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != (len(prepared),) or np.any(weight_array < 0):
            raise ValueError("weights must contain one non-negative value per image.")
        if not np.any(weight_array > 0):
            raise ValueError("At least one weight must be positive.")
    if saturation_levels is None:
        saturation_array = np.asarray(
            [float(np.nanmax(image)) for image in prepared], dtype=np.float64
        )
    else:
        saturation_array = np.asarray(saturation_levels, dtype=np.float64)
        if saturation_array.shape != (len(prepared),) or np.any(saturation_array <= 0):
            raise ValueError("saturation_levels must contain one positive value per image.")
    if not 0 < saturation_fraction <= 1:
        raise ValueError("saturation_fraction must lie in (0, 1].")
    if saturation_dilation < 0:
        raise ValueError("saturation_dilation cannot be negative.")
    validate_percentile(background_percentile, "background_percentile")

    weighted_sum = np.zeros(prepared[0].shape, dtype=np.float64)
    weight_sum = np.zeros(prepared[0].shape, dtype=np.float64)
    flux_images: list[FloatImage] = []
    for index, image in enumerate(prepared):
        finite_values = image[np.isfinite(image)]
        if finite_values.size == 0:
            flux_images.append(np.full_like(image, np.nan))
            continue
        background = float(np.percentile(finite_values, background_percentile))
        flux = ((image - background) / exposure_array[index]).astype(np.float32)
        flux_images.append(flux)
        saturated = image >= saturation_fraction * saturation_array[index]
        if saturation_dilation:
            saturated = binary_dilation(saturated, iterations=saturation_dilation)
        valid = np.isfinite(flux) & ~saturated
        weighted_sum[valid] += flux[valid] * weight_array[index]
        weight_sum[valid] += weight_array[index]

    combined = np.divide(
        weighted_sum,
        weight_sum,
        out=np.full(prepared[0].shape, np.nan, dtype=np.float64),
        where=weight_sum > 0,
    )
    # If every exposure is saturated at a pixel, retain the shortest exposure's
    # value. It cannot restore clipped information, but it avoids inventing a hole.
    shortest_first = np.argsort(exposure_array)
    unrecoverable = ~np.isfinite(combined)
    missing = unrecoverable.copy()
    for index in shortest_first:
        fallback = flux_images[int(index)]
        usable = missing & np.isfinite(fallback)
        combined[usable] = fallback[usable]
        missing &= ~usable
        if not np.any(missing):
            break
    return combined.astype(np.float32), unrecoverable


def hdr_combine(
    images: Sequence[ArrayLike],
    exposures: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    saturation_levels: Sequence[float] | None = None,
    saturation_fraction: float = 0.98,
    saturation_dilation: int = 2,
    background_percentile: float = 20.0,
) -> FloatImage:
    """Combine aligned mixed-exposure masters into a linear flux-rate image."""
    result, _ = _hdr_combine(
        images,
        exposures,
        weights=weights,
        saturation_levels=saturation_levels,
        saturation_fraction=saturation_fraction,
        saturation_dilation=saturation_dilation,
        background_percentile=background_percentile,
    )
    return result


def hdr_combine_with_mask(
    images: Sequence[ArrayLike],
    exposures: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    saturation_levels: Sequence[float] | None = None,
    saturation_fraction: float = 0.98,
    saturation_dilation: int = 2,
    background_percentile: float = 20.0,
) -> tuple[FloatImage, np.ndarray]:
    """Return an HDR master and pixels clipped in every supplied exposure."""
    return _hdr_combine(
        images,
        exposures,
        weights=weights,
        saturation_levels=saturation_levels,
        saturation_fraction=saturation_fraction,
        saturation_dilation=saturation_dilation,
        background_percentile=background_percentile,
    )
