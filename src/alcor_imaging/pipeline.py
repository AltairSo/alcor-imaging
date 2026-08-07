from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import TypeVar

import numpy as np
from scipy.ndimage import gaussian_filter

from ._validation import FloatImage, as_float_image
from .background import subtract_background
from .calibration import calibrate
from .color import adjust_saturation, apply_luminance, apply_palette, channel_balance
from .demosaic import demosaic, infer_bayer_pattern, normalize_bayer_pattern
from .enhance import wavelet_denoise
from .fits import read_fits
from .geometry import crop_to_overlap, overlap_bounds
from .hdr import hdr_combine, hdr_combine_with_mask
from .models import (
    CalibrationSet,
    Frame,
    ImageSource,
    LRGBConfig,
    LRGBResult,
    NarrowbandConfig,
    NarrowbandResult,
    OSCConfig,
    OSCResult,
    RegistrationRecord,
)
from .registration import register_image, register_many, register_rgb_many
from .stacking import register_and_stack, stack_rgb
from .stretch import stretch, stretch_rgb

T = TypeVar("T")


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


def _normalize_lrgb_key(value: str) -> str:
    normalized = str(value).strip().upper()
    aliases = {
        "L": "L",
        "LUM": "L",
        "LUMINANCE": "L",
        "R": "R",
        "RED": "R",
        "G": "G",
        "GREEN": "G",
        "B": "B",
        "BLUE": "B",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown LRGB channel {value!r}.")
    return aliases[normalized]


def _normalize_lrgb_mapping(mapping: Mapping[str, T]) -> dict[str, T]:
    result: dict[str, T] = {}
    for key, value in mapping.items():
        normalized = _normalize_lrgb_key(key)
        if normalized in result:
            raise ValueError(f"More than one value was supplied for LRGB channel {normalized}.")
        result[normalized] = value
    return result


def _saturation_level(frame: Frame) -> float:
    for key in ("SATURATE", "SATLEVEL", "DATAMAX"):
        value = frame.header.get(key)
        if value is not None and float(value) > 0:
            return float(value)
    bitpix = int(frame.header.get("BITPIX", 0))
    bzero = float(frame.header.get("BZERO", 0))
    bscale = float(frame.header.get("BSCALE", 1))
    if bitpix > 0:
        if bzero == 2 ** (bitpix - 1):
            return float(((2**bitpix) - 1) * bscale)
        return float(((2 ** (bitpix - 1)) - 1) * bscale + bzero)
    return float(np.nanmax(frame.data))


def process_lrgb(
    channels: Mapping[str, Sequence[ImageSource]],
    *,
    config: LRGBConfig | None = None,
    exposures: Mapping[str, Sequence[float]] | None = None,
    weights: Mapping[str, Sequence[float]] | None = None,
) -> LRGBResult:
    """Process mixed-exposure mono L/R/G/B masters into an HDR LRGB image.

    Inputs must be grouped explicitly by filter. FITS ``EXPTIME``/``EXPOSURE`` values
    are used unless ``exposures`` is supplied. For pre-stacked masters, pass effective
    integration-time ``weights`` so deeper masters contribute proportionally.
    """
    config = config or LRGBConfig()
    normalized_channels = _normalize_lrgb_mapping(channels)
    if set(normalized_channels) != {"L", "R", "G", "B"}:
        missing = sorted({"L", "R", "G", "B"} - set(normalized_channels))
        raise ValueError(f"LRGB processing requires L, R, G, and B; missing {missing}.")
    normalized_exposures = _normalize_lrgb_mapping(exposures) if exposures else {}
    normalized_weights = _normalize_lrgb_mapping(weights) if weights else {}

    masters: dict[str, FloatImage] = {}
    registrations: dict[str, list[RegistrationRecord]] = {}
    accepted_indices: dict[str, list[int]] = {}
    rejected_indices: dict[str, list[int]] = {}
    luminance_unrecoverable: np.ndarray | None = None

    for channel in ("L", "R", "G", "B"):
        sources = normalized_channels[channel]
        if not sources:
            raise ValueError(f"LRGB channel {channel} contains no images.")
        frames = [_load_frame(source) for source in sources]
        if channel in normalized_exposures:
            channel_exposures = list(normalized_exposures[channel])
        else:
            channel_exposures = [
                float(frame.header.get("EXPTIME", frame.header.get("EXPOSURE", 0)))
                for frame in frames
            ]
        if len(channel_exposures) != len(frames) or any(
            exposure <= 0 for exposure in channel_exposures
        ):
            raise ValueError(
                f"Channel {channel} needs one positive exposure per image; "
                "supply the exposures mapping when FITS metadata is incomplete."
            )
        channel_weights = (
            list(normalized_weights[channel])
            if channel in normalized_weights
            else channel_exposures
        )
        if len(channel_weights) != len(frames):
            raise ValueError(f"Channel {channel} needs one weight per image.")
        reference_index = int(np.argmax(channel_exposures))
        aligned, records = register_many(
            [frame.data for frame in frames],
            reference_index=reference_index,
            config=config.registration,
            on_error="reject",
        )
        accepted = [record.index for record in records if record.accepted]
        rejected = [record.index for record in records if not record.accepted]
        if not accepted:
            raise RuntimeError(f"No {channel} images registered successfully.")
        combine_kwargs = {
            "weights": [channel_weights[index] for index in accepted],
            "saturation_levels": [
                config.saturation_level or _saturation_level(frames[index])
                for index in accepted
            ],
            "saturation_fraction": config.saturation_fraction,
            "saturation_dilation": config.saturation_dilation,
            "background_percentile": config.background_percentile,
        }
        accepted_exposures = [channel_exposures[index] for index in accepted]
        if channel == "L":
            masters[channel], luminance_unrecoverable = hdr_combine_with_mask(
                aligned,
                accepted_exposures,
                **combine_kwargs,
            )
        else:
            masters[channel] = hdr_combine(
                aligned,
                accepted_exposures,
                **combine_kwargs,
            )
        registrations[channel] = records
        accepted_indices[channel] = accepted
        rejected_indices[channel] = rejected

    aligned_masters = [masters["L"]]
    for channel in ("R", "G", "B"):
        aligned, transform, _ = register_image(
            masters[channel], masters["L"], config.registration
        )
        aligned_masters.append(aligned)
        registrations[f"{channel}_to_L"] = [
            RegistrationRecord(
                index=0,
                accepted=True,
                rotation_degrees=float(np.degrees(transform.rotation)),
                translation=tuple(float(value) for value in transform.translation),
            )
        ]
    if config.crop_to_overlap:
        row_slice, col_slice = overlap_bounds(aligned_masters)
        aligned_masters = [
            image[row_slice, col_slice].copy() for image in aligned_masters
        ]
        for image in aligned_masters:
            finite = np.isfinite(image)
            if not np.all(finite):
                image[~finite] = np.nanmedian(image)
        if luminance_unrecoverable is not None:
            luminance_unrecoverable = luminance_unrecoverable[row_slice, col_slice]
    linear_luminance, red, green, blue = aligned_masters
    linear_rgb = np.stack((red, green, blue), axis=-1).astype(np.float32)
    linear_rgb = channel_balance(linear_rgb, config.white_balance, clip=False)

    display_rgb = stretch_rgb(
        linear_rgb, config.rgb_stretch, highlight_knee=config.highlight_knee
    )
    if config.denoise_strength:
        display_rgb = wavelet_denoise(display_rgb, strength=config.denoise_strength)
    luminance_rgb = np.repeat(linear_luminance[..., None], 3, axis=-1)
    display_luminance = stretch_rgb(
        luminance_rgb,
        config.luminance_stretch,
        highlight_knee=config.highlight_knee,
    )[..., 0]
    if not 0 <= config.luminance_weight <= 1:
        raise ValueError("luminance_weight must lie between 0 and 1.")
    rgb_luminance = np.einsum(
        "...c,c->...", display_rgb, (0.2126, 0.7152, 0.0722)
    )
    if luminance_unrecoverable is None:
        luminance_reliability = np.ones_like(display_luminance)
    else:
        luminance_reliability = 1.0 - gaussian_filter(
            luminance_unrecoverable.astype(np.float32), sigma=3.0
        )
    effective_luminance_weight = config.luminance_weight * luminance_reliability
    target_luminance = (
        effective_luminance_weight * display_luminance
        + (1.0 - effective_luminance_weight) * rgb_luminance
    )
    display_rgb = apply_luminance(
        display_rgb, target_luminance, ratio_limits=(0.5, 2.5)
    )
    display_rgb = adjust_saturation(display_rgb, config.saturation)
    if luminance_unrecoverable is None:
        luminance_unrecoverable = np.zeros_like(linear_luminance, dtype=bool)
    return LRGBResult(
        linear_luminance=linear_luminance,
        linear_rgb=linear_rgb,
        rgb=display_rgb,
        channel_masters={
            "L": linear_luminance,
            "R": red,
            "G": green,
            "B": blue,
        },
        registrations=registrations,
        accepted_indices=accepted_indices,
        rejected_indices=rejected_indices,
        luminance_unrecoverable_mask=luminance_unrecoverable,
    )
