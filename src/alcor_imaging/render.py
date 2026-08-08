from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter

from ._validation import FloatImage, as_float_image, validate_percentile
from .backend import Backend, get_array_module, resolve_backend, to_host
from .color import adjust_saturation, channel_balance
from .models import RenderConfig


def _validate_dual_parameters(
    *,
    white_percentile: float,
    faint_strength: float,
    highlight_strength: float,
    core_start: float,
    core_end: float,
    mask_blur_sigma: float,
    shadow_knee: float,
    gamma: float,
    highlight_knee: float,
) -> None:
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


def estimate_background_offsets(
    rgb: ArrayLike, *, percentile: float = 20.0, backend: Backend = "cpu"
) -> tuple[float, float, float]:
    """Estimate one robust additive background offset per RGB channel."""
    validate_percentile(percentile, "percentile")
    data = as_float_image(rgb, ndim=3)
    if data.shape[-1] != 3:
        raise ValueError("RGB input must have three channels on the last axis.")
    selected_backend = resolve_backend(backend)
    xp = get_array_module(selected_backend)
    offsets: list[float] = []
    for channel in np.moveaxis(data, -1, 0):
        device_channel = xp.asarray(channel)
        finite = device_channel[xp.isfinite(device_channel)]
        if finite.size == 0:
            raise ValueError("Every RGB channel must contain finite pixels.")
        offsets.append(float(xp.percentile(finite, percentile)))
    return tuple(offsets)  # type: ignore[return-value]


def neutralize_background(
    rgb: ArrayLike,
    *,
    percentile: float = 20.0,
    offsets: tuple[float, float, float] | None = None,
    clip: bool = True,
    backend: Backend = "cpu",
) -> FloatImage:
    """Remove independently estimated or caller-supplied channel backgrounds."""
    data = as_float_image(rgb, ndim=3)
    if data.shape[-1] != 3:
        raise ValueError("RGB input must have three channels on the last axis.")
    selected = offsets or estimate_background_offsets(data, percentile=percentile, backend=backend)
    if len(selected) != 3 or not np.all(np.isfinite(selected)):
        raise ValueError("offsets must contain three finite values.")
    selected_backend = resolve_backend(backend)
    xp = get_array_module(selected_backend)
    result = xp.asarray(data) - xp.asarray(selected, dtype=xp.float32)
    if clip:
        result = xp.clip(result, 0.0, None)
    return np.asarray(to_host(result), dtype=np.float32)


def _soft_clip_device(image: object, knee: float, xp: object) -> object:
    data = xp.clip(image, 0.0, None)
    excess = xp.maximum(data - knee, 0.0)
    compressed = knee + (1.0 - knee) * (1.0 - xp.exp(-excess / max(1.0 - knee, 1e-12)))
    return xp.where(data <= knee, data, compressed)


def _dual_asinh_device(
    data: object,
    *,
    white: float,
    faint_strength: float,
    highlight_strength: float,
    core_start: float,
    core_end: float,
    mask_blur_sigma: float,
    shadow_knee: float,
    gamma: float,
    highlight_knee: float,
    xp: object,
    selected_backend: str,
) -> object:
    normalized = data / white
    light = xp.einsum("...c,c->...", normalized, (0.2126, 0.7152, 0.0722))
    faint = xp.arcsinh(faint_strength * light) / np.arcsinh(faint_strength)
    gentle = xp.arcsinh(highlight_strength * light) / np.arcsinh(highlight_strength)
    core_mask = xp.clip((light - core_start) / (core_end - core_start), 0.0, 1.0)
    if mask_blur_sigma:
        if selected_backend == "gpu":
            from cupyx.scipy.ndimage import gaussian_filter as device_gaussian_filter
        else:
            device_gaussian_filter = gaussian_filter
        core_mask = device_gaussian_filter(core_mask, sigma=mask_blur_sigma)
    core_mask = core_mask * core_mask * (3.0 - 2.0 * core_mask)
    target = faint * (1.0 - core_mask) + gentle * core_mask
    if shadow_knee:
        shadow_mask = xp.clip(light / shadow_knee, 0.0, 1.0)
        shadow_mask = shadow_mask * shadow_mask * (3.0 - 2.0 * shadow_mask)
        target *= shadow_mask
    if gamma != 1.0:
        target = xp.power(xp.clip(target, 0.0, None), gamma)
    ratio = xp.divide(target, light, out=xp.zeros_like(light), where=light > 1e-12)
    result = normalized * ratio[..., None]
    peak = xp.max(result, axis=-1)
    compressed_peak = _soft_clip_device(peak, highlight_knee, xp)
    compression = xp.divide(compressed_peak, peak, out=xp.zeros_like(peak), where=peak > 1e-12)
    return xp.clip(result * compression[..., None], 0.0, 1.0).astype(xp.float32)


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
    backend: Backend = "cpu",
) -> FloatImage:
    """Stretch faint and bright structures separately while preserving RGB ratios.

    The faint asinh curve reveals low-surface-brightness signal. A smoothed mask
    progressively substitutes the gentler highlight curve around bright stars and
    cores. All curve operations are linked through luminance, so hues do not shift.
    """
    _validate_dual_parameters(
        white_percentile=white_percentile,
        faint_strength=faint_strength,
        highlight_strength=highlight_strength,
        core_start=core_start,
        core_end=core_end,
        mask_blur_sigma=mask_blur_sigma,
        shadow_knee=shadow_knee,
        gamma=gamma,
        highlight_knee=highlight_knee,
    )

    data = np.clip(as_float_image(rgb, ndim=3), 0.0, None)
    if data.shape[-1] != 3:
        raise ValueError("RGB input must have three channels on the last axis.")
    selected_backend = resolve_backend(backend)
    xp = get_array_module(selected_backend)
    device = xp.asarray(data)
    light = xp.einsum("...c,c->...", device, (0.2126, 0.7152, 0.0722))
    finite = light[xp.isfinite(light)]
    if finite.size == 0:
        raise ValueError("RGB image contains no finite pixels.")
    white = float(xp.percentile(finite, white_percentile))
    if white <= 0:
        raise ValueError("The selected RGB white point must be positive.")
    result = _dual_asinh_device(
        device,
        white=white,
        faint_strength=faint_strength,
        highlight_strength=highlight_strength,
        core_start=core_start,
        core_end=core_end,
        mask_blur_sigma=mask_blur_sigma,
        shadow_knee=shadow_knee,
        gamma=gamma,
        highlight_knee=highlight_knee,
        xp=xp,
        selected_backend=selected_backend,
    )
    return np.asarray(to_host(result), dtype=np.float32)


def _gpu_render_rgb_tiled(
    rgb: FloatImage,
    config: RenderConfig,
    *,
    tile_size: int,
) -> FloatImage:
    import cupy as cp

    if tile_size < 1:
        raise ValueError("tile_size must be positive.")
    _validate_dual_parameters(
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
    gains = np.asarray(config.channel_gains, dtype=np.float32)
    if gains.shape != (3,) or np.any(~np.isfinite(gains)) or np.any(gains < 0):
        raise ValueError("channel_gains must contain three finite non-negative values.")
    if config.background_offsets is not None:
        offsets = np.asarray(config.background_offsets, dtype=np.float32)
    elif config.background_percentile is not None:
        validate_percentile(config.background_percentile, "background_percentile")
        estimated = []
        for channel in range(3):
            plane = cp.asarray(rgb[..., channel]) * float(gains[channel])
            finite_plane = plane[cp.isfinite(plane)]
            if finite_plane.size == 0:
                raise ValueError("Every RGB channel must contain finite pixels.")
            estimated.append(float(cp.percentile(finite_plane, config.background_percentile)))
            del plane, finite_plane
        offsets = np.asarray(estimated, dtype=np.float32)
    else:
        offsets = np.zeros(3, dtype=np.float32)
    if offsets.shape != (3,) or np.any(~np.isfinite(offsets)):
        raise ValueError("background_offsets must contain three finite values.")

    # One full-resolution luminance plane is sufficient for the global white point.
    light = cp.zeros(rgb.shape[:2], dtype=cp.float32)
    for channel, coefficient in enumerate((0.2126, 0.7152, 0.0722)):
        plane = cp.asarray(rgb[..., channel], dtype=cp.float32)
        plane = cp.maximum(plane * float(gains[channel]) - float(offsets[channel]), 0.0)
        light += plane * coefficient
        del plane
    finite = light[cp.isfinite(light)]
    if finite.size == 0:
        raise ValueError("RGB image contains no finite pixels.")
    white = float(cp.percentile(finite, config.white_percentile))
    del finite, light
    cp.get_default_memory_pool().free_all_blocks()
    if white <= 0:
        raise ValueError("The selected RGB white point must be positive.")

    result = np.empty_like(rgb, dtype=np.float32)
    halo = int(np.ceil(4 * config.mask_blur_sigma))
    for y0 in range(0, rgb.shape[0], tile_size):
        for x0 in range(0, rgb.shape[1], tile_size):
            y1 = min(y0 + tile_size, rgb.shape[0])
            x1 = min(x0 + tile_size, rgb.shape[1])
            hy0, hx0 = max(0, y0 - halo), max(0, x0 - halo)
            hy1, hx1 = min(rgb.shape[0], y1 + halo), min(rgb.shape[1], x1 + halo)
            device = cp.asarray(rgb[hy0:hy1, hx0:hx1], dtype=cp.float32)
            device = cp.maximum(device * cp.asarray(gains) - cp.asarray(offsets), 0.0)
            tile = _dual_asinh_device(
                device,
                white=white,
                faint_strength=config.faint_strength,
                highlight_strength=config.highlight_strength,
                core_start=config.core_start,
                core_end=config.core_end,
                mask_blur_sigma=config.mask_blur_sigma,
                shadow_knee=config.shadow_knee,
                gamma=config.gamma,
                highlight_knee=config.highlight_knee,
                xp=cp,
                selected_backend="gpu",
            )
            lightness = cp.einsum("...c,c->...", tile, (0.2126, 0.7152, 0.0722))
            tile = cp.clip(
                lightness[..., None] + config.saturation * (tile - lightness[..., None]),
                0.0,
                1.0,
            )
            sy0, sx0 = y0 - hy0, x0 - hx0
            result[y0:y1, x0:x1] = cp.asnumpy(tile[sy0 : sy0 + (y1 - y0), sx0 : sx0 + (x1 - x0)])
            del device, tile, lightness
    return result


def render_rgb(
    rgb: ArrayLike,
    config: RenderConfig | None = None,
    *,
    backend: Backend | None = None,
    tile_size: int | None = None,
) -> FloatImage:
    """Render a linear RGB array into a display-ready, bounded float image."""
    config = config or RenderConfig()
    selected_backend = config.backend if backend is None else backend
    selected_tile_size = config.tile_size if tile_size is None else tile_size
    input_data = as_float_image(rgb, ndim=3)
    if input_data.shape[-1] != 3:
        raise ValueError("RGB input must have three channels on the last axis.")
    if resolve_backend(selected_backend) == "gpu":
        return _gpu_render_rgb_tiled(input_data, config, tile_size=selected_tile_size)
    data = channel_balance(input_data, config.channel_gains, clip=False)
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
