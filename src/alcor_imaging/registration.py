from __future__ import annotations

from collections.abc import Sequence

import astroalign
import numpy as np
from numpy.typing import ArrayLike
from skimage.transform import SimilarityTransform

from ._validation import FloatImage, as_float_image, finite_values
from .models import RegistrationConfig, RegistrationRecord


def registration_stretch(
    image: ArrayLike,
    *,
    low_percentile: float = 20.0,
    high_percentile: float = 99.7,
    strength: float = 12.0,
) -> FloatImage:
    """Create a star-enhancing view used only for registration detection."""
    data = as_float_image(image)
    finite = finite_values(data)
    low, high = np.percentile(finite, (low_percentile, high_percentile))
    normalized = np.clip((data - low) / max(float(high - low), 1e-12), 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    return (np.arcsinh(strength * normalized) / np.arcsinh(strength)).astype(np.float32)


def estimate_transform(
    source: ArrayLike,
    reference: ArrayLike,
    config: RegistrationConfig | None = None,
) -> SimilarityTransform:
    """Estimate a full-resolution similarity transform using star correspondences."""
    config = config or RegistrationConfig()
    source_data = as_float_image(source)
    reference_data = as_float_image(reference)
    if source_data.shape != reference_data.shape:
        raise ValueError("Source and reference must have matching shapes.")
    if config.downsample < 1:
        raise ValueError("downsample must be at least 1.")

    step = config.downsample
    source_small = registration_stretch(source_data)[::step, ::step]
    reference_small = registration_stretch(reference_data)[::step, ::step]
    transform, _ = astroalign.find_transform(
        source_small,
        reference_small,
        max_control_points=config.max_control_points,
        detection_sigma=config.detection_sigma,
        min_area=config.min_area,
    )
    return SimilarityTransform(
        scale=transform.scale,
        rotation=transform.rotation,
        translation=np.asarray(transform.translation) * step,
    )


def apply_transform(
    source: ArrayLike,
    reference: ArrayLike,
    transform: SimilarityTransform,
    *,
    fill_value: float = np.nan,
) -> tuple[FloatImage, np.ndarray]:
    """Warp source onto reference and return the image plus invalid-pixel footprint."""
    source_data = as_float_image(source)
    reference_data = as_float_image(reference)
    aligned, footprint = astroalign.apply_transform(
        transform,
        source_data,
        reference_data,
        fill_value=fill_value,
        propagate_mask=True,
    )
    result = np.asarray(aligned, dtype=np.float32)
    invalid = np.asarray(footprint, dtype=bool) | ~np.isfinite(result)
    result[invalid] = fill_value
    return result, invalid


def register_image(
    source: ArrayLike,
    reference: ArrayLike,
    config: RegistrationConfig | None = None,
) -> tuple[FloatImage, SimilarityTransform, np.ndarray]:
    config = config or RegistrationConfig()
    transform = estimate_transform(source, reference, config)
    aligned, footprint = apply_transform(
        source, reference, transform, fill_value=config.fill_value
    )
    return aligned, transform, footprint


def register_many(
    images: Sequence[ArrayLike],
    *,
    reference_index: int = 0,
    config: RegistrationConfig | None = None,
    on_error: str = "reject",
) -> tuple[list[FloatImage], list[RegistrationRecord]]:
    """Register frames, retaining a structured record of every acceptance/rejection."""
    config = config or RegistrationConfig()
    if not images:
        raise ValueError("At least one image is required.")
    if not 0 <= reference_index < len(images):
        raise IndexError("reference_index is outside the image sequence.")
    if on_error not in {"reject", "raise"}:
        raise ValueError("on_error must be 'reject' or 'raise'.")
    prepared = [as_float_image(image) for image in images]
    reference = prepared[reference_index]
    aligned: list[FloatImage] = []
    records: list[RegistrationRecord] = []
    for index, image in enumerate(prepared):
        if image.shape != reference.shape:
            error = f"shape {image.shape} does not match reference {reference.shape}"
            if on_error == "raise":
                raise ValueError(error)
            records.append(RegistrationRecord(index=index, accepted=False, error=error))
            continue
        if index == reference_index:
            aligned.append(image.copy())
            records.append(
                RegistrationRecord(
                    index=index, accepted=True, rotation_degrees=0.0, translation=(0.0, 0.0)
                )
            )
            continue
        try:
            registered, transform, _ = register_image(image, reference, config)
        except Exception as error:  # astroalign exposes several backend exception types
            if on_error == "raise":
                raise
            records.append(RegistrationRecord(index=index, accepted=False, error=str(error)))
            continue
        translation = tuple(float(value) for value in transform.translation)
        aligned.append(registered)
        records.append(
            RegistrationRecord(
                index=index,
                accepted=True,
                rotation_degrees=float(np.degrees(transform.rotation)),
                translation=translation,
            )
        )
    return aligned, records
