from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import numpy as np

from ._validation import FloatImage, as_float_image
from .background import subtract_background
from .calibration import calibrate
from .color import adjust_saturation, apply_palette, channel_balance
from .demosaic import demosaic, infer_bayer_pattern, normalize_bayer_pattern
from .enhance import wavelet_denoise
from .fits import read_fits
from .geometry import crop_to_overlap, overlap_bounds
from .models import (
    CalibrationSet,
    Frame,
    ImageSource,
    NarrowbandConfig,
    NarrowbandResult,
    OSCConfig,
    OSCResult,
)
from .registration import register_image, register_rgb_many
from .stacking import register_and_stack, stack_rgb
from .stretch import stretch, stretch_rgb


def _load_source(source: ImageSource) -> FloatImage:
    if isinstance(source, (str, PathLike)):
        return read_fits(source).data
    return as_float_image(source)


def _load_frame(source: ImageSource) -> Frame:
    if isinstance(source, (str, PathLike)):
        return read_fits(source)
    return Frame(data=as_float_image(source), header={})


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


def process_osc(
    sources: Sequence[ImageSource],
    *,
    config: OSCConfig | None = None,
    calibration: CalibrationSet | None = None,
    exposures: Sequence[float] | None = None,
) -> OSCResult:
    """Process explicit one-shot-color Bayer FITS frames into linear and display RGB.

    Detector calibration occurs on the untouched CFA data before demosaicing.
    Registration is estimated from luminance and then applied identically to RGB channels.
    """
    config = config or OSCConfig()
    if not sources:
        raise ValueError("At least one OSC light frame is required.")
    if exposures is not None and len(exposures) != len(sources):
        raise ValueError("exposures must contain one value per light frame.")
    frames = [_load_frame(source) for source in sources]

    detected_patterns = {
        pattern
        for frame in frames
        if (pattern := infer_bayer_pattern(frame.header)) is not None
    }
    if config.bayer_pattern is not None:
        pattern = normalize_bayer_pattern(config.bayer_pattern)
    elif len(detected_patterns) == 1:
        pattern = detected_patterns.pop()
    elif len(detected_patterns) > 1:
        raise ValueError(f"Input FITS files disagree on Bayer pattern: {detected_patterns}.")
    else:
        raise ValueError(
            "No Bayer pattern was found in FITS headers. Set OSCConfig(bayer_pattern='RGGB') "
            "only after confirming the camera's actual CFA pattern."
        )

    rgb_frames: list[FloatImage] = []
    for index, frame in enumerate(frames):
        raw = frame.data
        if calibration is not None:
            raw = calibrate(
                raw,
                calibration,
                exposure=None if exposures is None else exposures[index],
            )
        rgb_frames.append(demosaic(raw, pattern, method=config.demosaic_method))

    aligned, records = register_rgb_many(
        rgb_frames,
        reference_index=config.reference_index,
        config=config.registration,
        on_error="reject",
    )
    accepted = [record.index for record in records if record.accepted]
    rejected = [record.index for record in records if not record.accepted]
    minimum = 1 if len(rgb_frames) == 1 else 2
    if len(aligned) < minimum:
        rejection_details = [
            (record.index, record.error) for record in records if not record.accepted
        ]
        raise RuntimeError(
            f"Only {len(aligned)} OSC frames registered; at least {minimum} are required. "
            f"Rejections: {rejection_details}"
        )

    if config.crop_to_overlap:
        row_slice, col_slice = overlap_bounds([image[..., 0] for image in aligned])
        aligned = [image[row_slice, col_slice].copy() for image in aligned]
    master = stack_rgb(aligned, config.stacking)

    linear_channels = []
    for channel in range(3):
        channel_data = master[..., channel]
        if config.background_box_size is not None:
            channel_data = subtract_background(
                channel_data, box_size=config.background_box_size
            )
        linear_channels.append(channel_data)
    linear_rgb = np.stack(linear_channels, axis=-1).astype(np.float32)
    linear_rgb = channel_balance(linear_rgb, config.white_balance, clip=False)

    display_rgb = stretch_rgb(
        linear_rgb,
        config.stretch,
        highlight_knee=config.highlight_knee,
    )
    if config.denoise_strength:
        display_rgb = wavelet_denoise(display_rgb, strength=config.denoise_strength)
    display_rgb = adjust_saturation(display_rgb, config.saturation)
    return OSCResult(
        linear_rgb=linear_rgb,
        rgb=display_rgb,
        accepted_indices=accepted,
        rejected_indices=rejected,
        registrations=records,
        bayer_pattern=pattern,
    )
