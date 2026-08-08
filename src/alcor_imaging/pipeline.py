from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import TypeVar

import numpy as np
from scipy.ndimage import gaussian_filter

from ._validation import FloatImage, as_float_image
from .background import subtract_background
from .calibration import calibrate
from .channels import align_mono_masters, integrate_mono_channel
from .color import (
    adjust_saturation,
    apply_luminance,
    apply_palette,
    channel_balance,
    combine_channels,
)
from .demosaic import demosaic, infer_bayer_pattern, normalize_bayer_pattern
from .enhance import wavelet_denoise
from .fits import read_fits
from .geometry import overlap_bounds
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
    RenderConfig,
    StackResult,
)
from .registration import register_image, register_rgb_many
from .render import render_rgb
from .stacking import stack_rgb
from .stretch import stretch, stretch_rgb

T = TypeVar("T")


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
    weights: Sequence[Sequence[float] | None] | None = None,
    reference_indices: Sequence[int] | None = None,
) -> NarrowbandResult:
    """Run a reproducible narrowband workflow over explicit frame sequences.

    Channel order is caller-defined and must match the selected palette or the columns
    of ``mixing_matrix``. Inputs may be arrays or individual FITS paths; discovery,
    grouping, filter identification, and file management remain the caller's responsibility.
    """
    config = config or NarrowbandConfig()
    if not channels:
        raise ValueError("At least one channel is required.")
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
    if weights is None:
        weights = [None] * len(channels)
    if len(weights) != len(channels):
        raise ValueError("weights must contain one sequence per channel.")
    if reference_indices is None:
        reference_indices = [0] * len(channels)
    if len(reference_indices) != len(channels):
        raise ValueError("reference_indices must contain one index per channel.")

    integrations = []
    for channel_index, sources in enumerate(channels):
        channel_exposures = exposures[channel_index]
        integrations.append(
            integrate_mono_channel(
                sources,
                mode="stack",
                reference_index=reference_indices[channel_index],
                registration=config.registration,
                stacking=config.stacking,
                calibration=calibrations[channel_index],
                exposures=channel_exposures,
                weights=weights[channel_index],
            )
        )

    master_names = [str(index) for index in range(len(integrations))]
    aligned_result = align_mono_masters(
        {
            name: integration.master
            for name, integration in zip(master_names, integrations, strict=True)
        },
        reference=master_names[0],
        registration=config.registration,
        crop=config.crop_to_overlap,
        repair_holes=True,
    )
    masters = list(aligned_result.masters.values())

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
    if config.mixing_matrix is not None:
        rgb = combine_channels(display_channels, config.mixing_matrix)
    else:
        rgb = apply_palette(display_channels, config.palette)
    rgb = adjust_saturation(rgb, config.saturation)
    stacks = tuple(
        StackResult(
            image=integration.master,
            accepted_indices=integration.accepted_indices,
            rejected_indices=integration.rejected_indices,
            registrations=integration.registrations,
        )
        for integration in integrations
    )
    return NarrowbandResult(
        linear_channels=tuple(linear_channels),
        masters=tuple(masters),
        rgb=rgb,
        stacks=stacks,
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
        integration = integrate_mono_channel(
            [frame.data for frame in frames],
            mode="hdr",
            registration=config.registration,
            reference_index=reference_index,
            exposures=channel_exposures,
            weights=channel_weights,
            saturation_levels=[
                config.saturation_level or _saturation_level(frame) for frame in frames
            ],
            saturation_fraction=config.saturation_fraction,
            saturation_dilation=config.saturation_dilation,
            background_percentile=config.background_percentile,
        )
        masters[channel] = integration.master
        if channel == "L":
            luminance_unrecoverable = integration.unrecoverable_mask
        registrations[channel] = integration.registrations
        accepted_indices[channel] = integration.accepted_indices
        rejected_indices[channel] = integration.rejected_indices

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

    if config.render is None:
        display_rgb = stretch_rgb(
            linear_rgb, config.rgb_stretch, highlight_knee=config.highlight_knee
        )
    else:
        display_rgb = render_rgb(linear_rgb, config.render)
    if config.denoise_strength:
        display_rgb = wavelet_denoise(display_rgb, strength=config.denoise_strength)
    luminance_rgb = np.repeat(linear_luminance[..., None], 3, axis=-1)
    if config.render is None:
        display_luminance = stretch_rgb(
            luminance_rgb,
            config.luminance_stretch,
            highlight_knee=config.highlight_knee,
        )[..., 0]
    else:
        luminance_render = RenderConfig(
            background_percentile=config.render.background_percentile,
            white_percentile=config.render.white_percentile,
            faint_strength=config.render.faint_strength,
            highlight_strength=config.render.highlight_strength,
            core_start=config.render.core_start,
            core_end=config.render.core_end,
            mask_blur_sigma=config.render.mask_blur_sigma,
            shadow_knee=config.render.shadow_knee,
            gamma=config.render.gamma,
            saturation=1.0,
            highlight_knee=config.render.highlight_knee,
            backend=config.render.backend,
            tile_size=config.render.tile_size,
        )
        display_luminance = render_rgb(luminance_rgb, luminance_render)[..., 0]
    if not 0 <= config.luminance_weight <= 1:
        raise ValueError("luminance_weight must lie between 0 and 1.")
    rgb_luminance = np.einsum(
        "...c,c->...", display_rgb, (0.2126, 0.7152, 0.0722)
    )
    highlight_start, highlight_end = config.luminance_highlight_range
    if not 0 <= highlight_start < highlight_end <= 1:
        raise ValueError(
            "luminance_highlight_range must increase from zero to at most one."
        )
    highlight_mask = np.clip(
        (rgb_luminance - highlight_start) / (highlight_end - highlight_start),
        0.0,
        1.0,
    )
    highlight_mask = highlight_mask * highlight_mask * (3.0 - 2.0 * highlight_mask)
    highlight_reliability = 1.0 - highlight_mask
    if luminance_unrecoverable is None:
        luminance_reliability = np.ones_like(display_luminance)
    else:
        luminance_reliability = 1.0 - gaussian_filter(
            luminance_unrecoverable.astype(np.float32), sigma=3.0
        )
    effective_luminance_weight = (
        config.luminance_weight * luminance_reliability * highlight_reliability
    )
    target_luminance = (
        effective_luminance_weight * display_luminance
        + (1.0 - effective_luminance_weight) * rgb_luminance
    )
    display_rgb = apply_luminance(
        display_rgb,
        target_luminance,
        ratio_limits=config.luminance_ratio_limits,
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
