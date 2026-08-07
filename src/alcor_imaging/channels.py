from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike

import numpy as np

from ._validation import FloatImage, as_float_image
from .calibration import calibrate
from .fits import read_fits
from .geometry import overlap_bounds
from .hdr import hdr_combine_with_mask
from .models import (
    CalibrationSet,
    ChannelIntegrationResult,
    ImageSource,
    MasterAlignmentResult,
    RegistrationConfig,
    RegistrationRecord,
    StackConfig,
)
from .registration import register_image, register_many
from .stacking import stack


def _load_mono(source: ImageSource) -> FloatImage:
    if isinstance(source, (str, PathLike)):
        return read_fits(source).data
    return as_float_image(source)


def integrate_mono_channel(
    sources: Sequence[ImageSource],
    *,
    mode: str = "stack",
    registration: RegistrationConfig | None = None,
    stacking: StackConfig | None = None,
    reference_index: int = 0,
    calibration: CalibrationSet | None = None,
    exposures: Sequence[float] | None = None,
    weights: Sequence[float] | None = None,
    saturation_levels: Sequence[float] | None = None,
    saturation_fraction: float = 0.98,
    saturation_dilation: int = 2,
    background_percentile: float = 20.0,
    minimum_accepted: int | None = None,
) -> ChannelIntegrationResult:
    """Register and integrate one caller-defined mono channel.

    ``mode='stack'`` performs ordinary robust integration using ``StackConfig``.
    ``mode='hdr'`` converts mixed exposures to flux rate, rejects saturated samples,
    and requires explicit ``exposures`` and ``saturation_levels``. This function does
    not infer filters, group files, parse filenames, or choose scientific metadata.
    """
    if not sources:
        raise ValueError("At least one source is required.")
    if mode not in {"stack", "hdr"}:
        raise ValueError("mode must be 'stack' or 'hdr'.")
    registration = registration or RegistrationConfig()
    stacking = stacking or StackConfig()
    images = [_load_mono(source) for source in sources]
    if exposures is not None and len(exposures) != len(images):
        raise ValueError("exposures must contain one value per source.")
    if weights is not None and len(weights) != len(images):
        raise ValueError("weights must contain one value per source.")
    if saturation_levels is not None and len(saturation_levels) != len(images):
        raise ValueError("saturation_levels must contain one value per source.")
    if calibration is not None:
        images = [
            calibrate(
                image,
                calibration,
                exposure=None if exposures is None else exposures[index],
            )
            for index, image in enumerate(images)
        ]

    aligned, records = register_many(
        images,
        reference_index=reference_index,
        config=registration,
        on_error="reject",
    )
    accepted = [record.index for record in records if record.accepted]
    rejected = [record.index for record in records if not record.accepted]
    required = minimum_accepted
    if required is None:
        required = 1 if len(images) == 1 else 2
    if len(accepted) < required:
        details = [(record.index, record.error) for record in records if not record.accepted]
        raise RuntimeError(
            f"Only {len(accepted)} sources registered; at least {required} are required. "
            f"Rejections: {details}"
        )

    if mode == "stack":
        master = stack(
            aligned,
            stacking,
            weights=None if weights is None else [weights[index] for index in accepted],
        )
        unrecoverable = None
    else:
        if exposures is None or saturation_levels is None:
            raise ValueError(
                "HDR integration requires explicit exposures and saturation_levels."
            )
        master, unrecoverable = hdr_combine_with_mask(
            aligned,
            [exposures[index] for index in accepted],
            weights=None if weights is None else [weights[index] for index in accepted],
            saturation_levels=[saturation_levels[index] for index in accepted],
            saturation_fraction=saturation_fraction,
            saturation_dilation=saturation_dilation,
            background_percentile=background_percentile,
        )
    return ChannelIntegrationResult(
        master=master,
        accepted_indices=accepted,
        rejected_indices=rejected,
        registrations=records,
        unrecoverable_mask=unrecoverable,
    )


def align_mono_masters(
    masters: Mapping[str, ImageSource],
    *,
    reference: str,
    registration: RegistrationConfig | None = None,
    crop: bool = True,
    repair_holes: bool = False,
) -> MasterAlignmentResult:
    """Align arbitrary named mono masters to a caller-selected reference channel."""
    if not masters:
        raise ValueError("At least one master is required.")
    if reference not in masters:
        raise KeyError(f"Reference channel {reference!r} is not present in masters.")
    registration = registration or RegistrationConfig()
    prepared = {name: _load_mono(source) for name, source in masters.items()}
    reference_image = prepared[reference]
    aligned: dict[str, FloatImage] = {reference: reference_image.copy()}
    records = {
        reference: RegistrationRecord(
            index=0,
            accepted=True,
            rotation_degrees=0.0,
            translation=(0.0, 0.0),
        )
    }
    for name, image in prepared.items():
        if name == reference:
            continue
        registered, transform, _ = register_image(image, reference_image, registration)
        aligned[name] = registered
        records[name] = RegistrationRecord(
            index=0,
            accepted=True,
            rotation_degrees=float(np.degrees(transform.rotation)),
            translation=tuple(float(value) for value in transform.translation),
        )
    # Restore caller insertion order because it often encodes matrix channel order.
    aligned = {name: aligned[name] for name in masters}
    if crop:
        row_slice, col_slice = overlap_bounds(list(aligned.values()))
        aligned = {
            name: image[row_slice, col_slice].copy() for name, image in aligned.items()
        }
    if repair_holes:
        for image in aligned.values():
            finite = np.isfinite(image)
            if not np.all(finite):
                image[~finite] = np.nanmedian(image)
    return MasterAlignmentResult(
        masters=aligned,
        registrations=records,
        reference=reference,
    )

