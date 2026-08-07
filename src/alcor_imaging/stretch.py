from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatImage, as_float_image, finite_values, validate_percentile
from .models import StretchConfig


def normalize(
    image: ArrayLike,
    *,
    black_percentile: float = 0.8,
    white_percentile: float = 99.9,
    clip: bool = True,
) -> FloatImage:
    """Map robust black/white percentile points to zero and one."""
    validate_percentile(black_percentile, "black_percentile")
    validate_percentile(white_percentile, "white_percentile")
    if black_percentile >= white_percentile:
        raise ValueError("black_percentile must be below white_percentile.")
    data = as_float_image(image)
    black, white = np.percentile(finite_values(data), (black_percentile, white_percentile))
    result = (data - black) / max(float(white - black), 1e-12)
    if clip:
        result = np.clip(result, 0.0, 1.0)
    return result.astype(np.float32)


def asinh_stretch(image: ArrayLike, *, strength: float = 7.0) -> FloatImage:
    """Apply a normalized inverse-hyperbolic-sine stretch to linear data."""
    if strength <= 0:
        raise ValueError("strength must be positive.")
    data = as_float_image(image, ndim=None)
    return (np.arcsinh(strength * np.clip(data, 0.0, None)) / np.arcsinh(strength)).astype(
        np.float32
    )


def masked_asinh_stretch(
    image: ArrayLike,
    *,
    strength: float = 7.0,
    shadow_protection: float = 0.015,
) -> FloatImage:
    """Stretch signal while blending shadows back toward the linear input."""
    if shadow_protection < 0:
        raise ValueError("shadow_protection cannot be negative.")
    data = np.clip(as_float_image(image, ndim=None), 0.0, None)
    stretched = asinh_stretch(data, strength=strength)
    protection = np.divide(
        data,
        data + shadow_protection,
        out=np.ones_like(data),
        where=(data + shadow_protection) != 0,
    )
    return np.clip(protection * stretched + (1.0 - protection) * data, 0.0, 1.0).astype(
        np.float32
    )


def midtone_transfer(image: ArrayLike, *, midtone: float = 0.5) -> FloatImage:
    """Apply PixInsight-style midtones transfer; 0.5 is the identity."""
    if not 0 < midtone < 1:
        raise ValueError("midtone must lie strictly between 0 and 1.")
    data = np.clip(as_float_image(image, ndim=None), 0.0, 1.0)
    denominator = (2 * midtone - 1) * data - midtone
    result = np.divide(
        (midtone - 1) * data,
        denominator,
        out=np.zeros_like(data),
        where=np.abs(denominator) > 1e-12,
    )
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def stretch(image: ArrayLike, config: StretchConfig | None = None) -> FloatImage:
    config = config or StretchConfig()
    result = normalize(
        image,
        black_percentile=config.black_percentile,
        white_percentile=config.white_percentile,
    )
    result = masked_asinh_stretch(
        result,
        strength=config.asinh_strength,
        shadow_protection=config.shadow_protection,
    )
    if config.gamma <= 0:
        raise ValueError("gamma must be positive.")
    if config.gamma != 1.0:
        result = np.power(result, config.gamma).astype(np.float32)
    return np.clip(result, 0.0, 1.0)


def soft_clip(image: ArrayLike, *, knee: float = 0.85) -> FloatImage:
    """Compress values above a knee smoothly toward one instead of hard clipping."""
    if not 0 < knee < 1:
        raise ValueError("knee must lie strictly between 0 and 1.")
    data = np.clip(as_float_image(image, ndim=None), 0.0, None)
    excess = np.maximum(data - knee, 0.0)
    compressed = knee + (1.0 - knee) * (
        1.0 - np.exp(-excess / max(1.0 - knee, 1e-12))
    )
    return np.where(data <= knee, data, compressed).astype(np.float32)


def stretch_rgb(
    rgb: ArrayLike,
    config: StretchConfig | None = None,
    *,
    highlight_knee: float = 0.85,
) -> FloatImage:
    """Apply a linked luminance stretch that preserves RGB channel ratios."""
    config = config or StretchConfig()
    data = as_float_image(rgb, ndim=3)
    if data.shape[-1] != 3:
        raise ValueError("RGB input must have three channels on the last axis.")
    validate_percentile(config.black_percentile, "black_percentile")
    validate_percentile(config.white_percentile, "white_percentile")
    if config.black_percentile >= config.white_percentile:
        raise ValueError("black_percentile must be below white_percentile.")
    luminance = np.einsum("...c,c->...", data, (0.2126, 0.7152, 0.0722))
    finite = luminance[np.isfinite(luminance)]
    if finite.size == 0:
        raise ValueError("RGB image contains no finite pixels.")
    black = float(np.percentile(finite, config.black_percentile))
    shifted = np.clip(data - black, 0.0, None)
    shifted_luminance = np.einsum(
        "...c,c->...", shifted, (0.2126, 0.7152, 0.0722)
    )
    white = float(
        np.percentile(
            shifted_luminance[np.isfinite(shifted_luminance)], config.white_percentile
        )
    )
    normalized = shifted / max(white, 1e-12)
    light = np.einsum("...c,c->...", normalized, (0.2126, 0.7152, 0.0722))
    target = np.arcsinh(config.asinh_strength * light) / np.arcsinh(config.asinh_strength)
    protection = np.divide(
        light,
        light + config.shadow_protection,
        out=np.ones_like(light),
        where=(light + config.shadow_protection) != 0,
    )
    target = protection * target + (1.0 - protection) * light
    ratio = np.divide(target, light, out=np.zeros_like(light), where=light > 1e-12)
    result = normalized * ratio[..., None]
    if config.gamma <= 0:
        raise ValueError("gamma must be positive.")
    if config.gamma != 1.0:
        result = np.power(np.clip(result, 0.0, None), config.gamma)
    return soft_clip(result, knee=highlight_knee)
