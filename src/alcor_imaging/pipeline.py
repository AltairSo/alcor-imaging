from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import numpy as np

from ._validation import FloatImage, as_float_image
from .background import subtract_background
from .calibration import calibrate
from .color import adjust_saturation, apply_palette
from .enhance import wavelet_denoise
from .fits import read_fits
from .geometry import crop_to_overlap
from .models import (
    CalibrationSet,
    ImageSource,
    NarrowbandConfig,
    NarrowbandResult,
)
from .registration import register_image
from .stacking import register_and_stack
from .stretch import stretch


def _load_source(source: ImageSource) -> FloatImage:
    if isinstance(source, (str, PathLike)):
        return read_fits(source).data
    return as_float_image(source)


def process_narrowband(
    channels: Sequence[Sequence[ImageSource]],
    *,
    config: NarrowbandConfig | None = None,
    calibrations: Sequence[CalibrationSet | None] | None = None,
    exposures: Sequence[Sequence[float] | None] | None = None,
    reference_indices: Sequence[int] | None = None,
) -> NarrowbandResult:
    """Run a reproducible narrowband workflow over explicit frame sequences.

    HOO expects channels ``(Ha, OIII)``. SHO expects ``(SII, Ha, OIII)``.
    Inputs may be arrays or individual FITS paths; discovery and file management
    remain the caller's responsibility.
    """
    config = config or NarrowbandConfig()
    palette = config.palette.upper()
    if palette not in {"HOO", "SHO"}:
        raise ValueError("palette must be 'HOO' or 'SHO'.")
    required_channels = 2 if palette == "HOO" else 3
    if len(channels) != required_channels:
        raise ValueError(f"{palette} requires {required_channels} channel sequences.")
    if any(not channel for channel in channels):
        raise ValueError("Every channel must contain at least one light frame.")
    if calibrations is None:
        calibrations = [None] * len(channels)
    if len(calibrations) != len(channels):
        raise ValueError("calibrations must contain one entry per channel.")
    if exposures is None:
        exposures = [None] * len(channels)
    if len(exposures) != len(channels):
        raise ValueError("exposures must contain one sequence per channel.")
    if reference_indices is None:
        reference_indices = [0] * len(channels)
    if len(reference_indices) != len(channels):
        raise ValueError("reference_indices must contain one index per channel.")

    stacks = []
    for channel_index, sources in enumerate(channels):
        images = [_load_source(source) for source in sources]
        channel_exposures = exposures[channel_index]
        if channel_exposures is not None and len(channel_exposures) != len(images):
            raise ValueError(f"Exposure count does not match channel {channel_index} frame count.")
        calibration = calibrations[channel_index]
        if calibration is not None:
            images = [
                calibrate(
                    image,
                    calibration,
                    exposure=None if channel_exposures is None else channel_exposures[index],
                )
                for index, image in enumerate(images)
            ]
        stacks.append(
            register_and_stack(
                images,
                reference_index=reference_indices[channel_index],
                registration=config.registration,
                stacking=config.stacking,
                minimum_accepted=1 if len(images) == 1 else 2,
            )
        )

    masters = [stacks[0].image]
    for channel_stack in stacks[1:]:
        aligned, _, _ = register_image(
            channel_stack.image, stacks[0].image, config.registration
        )
        masters.append(aligned)
    if config.crop_to_overlap:
        masters = crop_to_overlap(masters, repair_holes=True)

    linear_channels: list[FloatImage] = []
    for master in masters:
        channel = master
        if config.background_box_size is not None:
            channel = subtract_background(channel, box_size=config.background_box_size)
        linear_channels.append(channel.astype(np.float32))

    display_channels = [stretch(channel, config.stretch) for channel in linear_channels]
    if config.denoise_strength:
        display_channels = [
            wavelet_denoise(channel, strength=config.denoise_strength)
            for channel in display_channels
        ]
    if config.channel_boosts:
        if len(config.channel_boosts) != len(display_channels):
            raise ValueError("channel_boosts must contain one gain per channel.")
        display_channels = [
            np.clip(channel * boost, 0.0, 1.0).astype(np.float32)
            for channel, boost in zip(display_channels, config.channel_boosts, strict=True)
        ]
    rgb = apply_palette(display_channels, palette)
    rgb = adjust_saturation(rgb, config.saturation)
    return NarrowbandResult(
        linear_channels=tuple(linear_channels),
        masters=tuple(masters),
        rgb=rgb,
        stacks=tuple(stacks),
    )
