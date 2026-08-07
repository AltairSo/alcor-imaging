from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from astropy.stats import sigma_clip
from astropy.utils.exceptions import AstropyUserWarning
from numpy.typing import ArrayLike

from ._validation import FloatImage, as_image_sequence
from .models import RegistrationConfig, StackConfig, StackResult
from .registration import register_many


def _normalization_coefficients(
    images: Sequence[FloatImage], mode: str
) -> tuple[np.ndarray, np.ndarray]:
    count = len(images)
    if mode == "none":
        return np.ones(count, dtype=np.float32), np.zeros(count, dtype=np.float32)
    medians = np.asarray([np.nanmedian(image) for image in images], dtype=np.float32)
    if np.any(~np.isfinite(medians)):
        raise ValueError("Frame normalization failed because a median is not finite.")
    target = float(np.median(medians))
    if mode == "median":
        return np.ones(count, dtype=np.float32), target - medians
    if mode == "multiplicative":
        if np.any(np.abs(medians) < 1e-12):
            raise ValueError("Multiplicative normalization requires non-zero medians.")
        return target / medians, np.zeros(count, dtype=np.float32)
    raise ValueError(f"Unknown normalization mode {mode!r}.")


def _combine_cube(
    cube: np.ndarray,
    config: StackConfig,
    weights: np.ndarray,
) -> FloatImage:
    if config.method.startswith("sigma_clip"):
        # NaNs are expected at registered borders and already have defined semantics.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Input data contains invalid values*",
                category=AstropyUserWarning,
            )
            clipped = sigma_clip(
                cube,
                sigma=config.sigma,
                maxiters=config.max_iterations,
                axis=0,
                cenfunc="median",
                stdfunc="mad_std",
                masked=True,
                copy=False,
            )
        data = clipped.filled(np.nan)
    else:
        data = cube

    if config.method.endswith("median") or config.method == "median":
        result = np.nanmedian(data, axis=0)
    elif config.method in {"mean", "sigma_clip_mean"}:
        valid = np.isfinite(data)
        weighted = np.where(valid, data, 0.0) * weights[:, None, None]
        denominator = np.sum(valid * weights[:, None, None], axis=0)
        result = np.divide(
            np.sum(weighted, axis=0),
            denominator,
            out=np.full(cube.shape[1:], np.nan, dtype=np.float32),
            where=denominator > 0,
        )
    else:
        raise ValueError(f"Unknown stack method {config.method!r}.")
    return np.asarray(result, dtype=np.float32)


def stack(
    images: Sequence[ArrayLike],
    config: StackConfig | None = None,
    *,
    weights: Sequence[float] | None = None,
) -> FloatImage:
    """Combine aligned frames with NaN-aware or sigma-clipped estimators."""
    config = config or StackConfig()
    prepared = as_image_sequence(images)
    scales, offsets = _normalization_coefficients(prepared, config.normalization)
    if weights is not None:
        if config.method not in {"mean", "sigma_clip_mean"}:
            raise ValueError("weights are supported only by mean-based stack methods.")
        weight_array = np.asarray(weights, dtype=np.float32)
        if weight_array.shape != (len(prepared),) or np.any(weight_array < 0):
            raise ValueError("weights must be one non-negative value per image.")
        if not np.any(weight_array > 0):
            raise ValueError("At least one weight must be positive.")
    else:
        weight_array = np.ones(len(prepared), dtype=np.float32)

    height, width = prepared[0].shape
    tile_size = config.tile_size or max(height, width)
    if tile_size < 1:
        raise ValueError("tile_size must be positive or None.")
    result = np.empty((height, width), dtype=np.float32)
    for y0 in range(0, height, tile_size):
        for x0 in range(0, width, tile_size):
            y1, x1 = min(y0 + tile_size, height), min(x0 + tile_size, width)
            cube = np.stack(
                [
                    image[y0:y1, x0:x1] * scales[index] + offsets[index]
                    for index, image in enumerate(prepared)
                ]
            ).astype(np.float32, copy=False)
            result[y0:y1, x0:x1] = _combine_cube(cube, config, weight_array)
    return result


def stack_rgb(
    images: Sequence[ArrayLike],
    config: StackConfig | None = None,
    *,
    weights: Sequence[float] | None = None,
) -> FloatImage:
    """Stack registered RGB frames independently without changing their geometry."""
    if not images:
        raise ValueError("At least one RGB image is required.")
    prepared = [np.asarray(image, dtype=np.float32) for image in images]
    shape = prepared[0].shape
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError("RGB frames must have shape (height, width, 3).")
    if any(image.shape != shape for image in prepared[1:]):
        raise ValueError("All RGB frames must have the same shape.")
    return np.stack(
        [
            stack([image[..., channel] for image in prepared], config, weights=weights)
            for channel in range(3)
        ],
        axis=-1,
    ).astype(np.float32)


def register_and_stack(
    images: Sequence[ArrayLike],
    *,
    reference_index: int = 0,
    registration: RegistrationConfig | None = None,
    stacking: StackConfig | None = None,
    minimum_accepted: int = 2,
) -> StackResult:
    registration = registration or RegistrationConfig()
    stacking = stacking or StackConfig()
    aligned, records = register_many(
        images, reference_index=reference_index, config=registration, on_error="reject"
    )
    accepted = [record.index for record in records if record.accepted]
    rejected = [record.index for record in records if not record.accepted]
    if len(accepted) < minimum_accepted:
        raise RuntimeError(
            f"Only {len(accepted)} frames registered; at least {minimum_accepted} are required."
        )
    return StackResult(
        image=stack(aligned, stacking),
        accepted_indices=accepted,
        rejected_indices=rejected,
        registrations=records,
    )
