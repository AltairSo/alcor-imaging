from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter

from ._validation import FloatImage, as_float_image, validate_percentile
from .color import adjust_saturation, channel_balance, luminance
from .models import RenderConfig
from .stretch import soft_clip


def estimate_background_offsets(
    rgb: ArrayLike, *, percentile: float = 20.0
) -> tuple[float, float, float]:
    """Estimate one robust additive background offset per RGB channel."""
    validate_percentile(percentile, "percentile")
    data = as_float_image(rgb, ndim=3)
    if data.shape[-1] != 3:
        raise ValueError("RGB input must have three channels on the last axis.")
    offsets = []
    for channel in np.moveaxis(data, -1, 0):
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            raise ValueError("Every RGB channel must contain finite pixels.")
        offsets.append(float(np.percentile(finite, percentile)))
    return tuple(offsets)  # type: ignore[return-value]


def neutralize_background(
    rgb: ArrayLike,
    *,
    percentile: float = 20.0,
    offsets: tuple[float, float, float] | None = None,
    clip: bool = True,
) -> FloatImage:
    """Remove independently estimated or caller-supplied channel backgrounds."""
    data = as_float_image(rgb, ndim=3)
    if data.shape[-1] != 3:
        raise ValueError("RGB input must have three channels on the last axis.")
    selected = offsets or estimate_background_offsets(data, percentile=percentile)
    if len(selected) != 3 or not np.all(np.isfinite(selected)):
        raise ValueError("offsets must contain three finite values.")
    result = data - np.asarray(selected, dtype=np.float32)
    if clip:
        result = np.clip(result, 0.0, None)
    return result.astype(np.float32)


def dual_asinh_stretch_rgb(
    rgb: ArrayLike,
    *,
    white_percentile: float = 99.9,
    faint_strength: float = 35.0,
    highlight_strength: float = 6.0,
    core_start: float = 0.08,
    core_end: float = 0.65,
    mask_blur_sigma: float = 2.0,
    shadow_knee: float = 0.0015,
    gamma: float = 0.88,
    highlight_knee: float = 0.82,
) -> FloatImage:
    """Stretch faint and bright structures separately while preserving RGB ratios.

    The faint asinh curve reveals low-surface-brightness signal. A smoothed mask
    progressively substitutes the gentler highlight curve around bright stars and
    cores. All curve operations are linked through luminance, so hues do not shift.
    """
    validate_percentile(white_percentile, "white_percentile")
    if faint_strength <= 0 or highlight_strength <= 0:
        raise ValueError("Stretch strengths must be positive.")
    if not 0 <= core_start < core_end:
        raise ValueError("core_start must be non-negative and below core_end.")
    if mask_blur_sigma < 0 or shadow_knee < 0:
        raise ValueError("mask_blur_sigma and shadow_knee cannot be negative.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    if not 0 < highlight_knee < 1:
        raise ValueError("highlight_knee must lie strictly between 0 and 1.")

    data = np.clip(as_float_image(rgb, ndim=3), 0.0, None)
    if data.shape[-1] != 3:
        raise ValueError("RGB input must have three channels on the last axis.")
    light = luminance(data)
    finite = light[np.isfinite(light)]
    if finite.size == 0:
        raise ValueError("RGB image contains no finite pixels.")
    white = float(np.percentile(finite, white_percentile))
    if white <= 0:
        raise ValueError("The selected RGB white point must be positive.")
    normalized = data / white
    light = light / white

    faint = np.arcsinh(faint_strength * light) / np.arcsinh(faint_strength)
    gentle = np.arcsinh(highlight_strength * light) / np.arcsinh(highlight_strength)
    core_mask = np.clip((light - core_start) / (core_end - core_start), 0.0, 1.0)
    if mask_blur_sigma:
        core_mask = gaussian_filter(core_mask, sigma=mask_blur_sigma)
    core_mask = core_mask * core_mask * (3.0 - 2.0 * core_mask)
    target = faint * (1.0 - core_mask) + gentle * core_mask

    if shadow_knee:
        shadow_mask = np.clip(light / shadow_knee, 0.0, 1.0)
        shadow_mask = shadow_mask * shadow_mask * (3.0 - 2.0 * shadow_mask)
        target *= shadow_mask
    if gamma != 1.0:
        target = np.power(np.clip(target, 0.0, None), gamma)

    ratio = np.divide(target, light, out=np.zeros_like(light), where=light > 1e-12)
    result = normalized * ratio[..., None]
    peak = np.max(result, axis=-1)
    compressed_peak = soft_clip(peak, knee=highlight_knee)
    compression = np.divide(
        compressed_peak, peak, out=np.zeros_like(peak), where=peak > 1e-12
    )
    return np.clip(result * compression[..., None], 0.0, 1.0).astype(np.float32)


def render_rgb(rgb: ArrayLike, config: RenderConfig | None = None) -> FloatImage:
    """Render a linear RGB array into a display-ready, bounded float image."""
    config = config or RenderConfig()
    data = channel_balance(rgb, config.channel_gains, clip=False)
    if config.background_offsets is not None:
        data = neutralize_background(data, offsets=config.background_offsets)
    elif config.background_percentile is not None:
        data = neutralize_background(data, percentile=config.background_percentile)
    else:
        data = np.clip(data, 0.0, None).astype(np.float32)
    result = dual_asinh_stretch_rgb(
        data,
        white_percentile=config.white_percentile,
        faint_strength=config.faint_strength,
        highlight_strength=config.highlight_strength,
        core_start=config.core_start,
        core_end=config.core_end,
        mask_blur_sigma=config.mask_blur_sigma,
        shadow_knee=config.shadow_knee,
        gamma=config.gamma,
        highlight_knee=config.highlight_knee,
    )
    return adjust_saturation(result, config.saturation)
