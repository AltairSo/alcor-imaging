from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_wavelet

from ._validation import FloatImage, as_float_image
from .color import apply_luminance, luminance


def wavelet_denoise(
    image: ArrayLike,
    *,
    strength: float = 0.5,
    wavelet: str = "db2",
    levels: int = 2,
) -> FloatImage:
    """Blend BayesShrink wavelet denoising with the original image."""
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1.")
    data = as_float_image(image, ndim=None)
    if data.ndim not in (2, 3):
        raise ValueError("wavelet_denoise expects a mono or RGB image.")
    denoised = denoise_wavelet(
        data,
        method="BayesShrink",
        mode="soft",
        wavelet=wavelet,
        wavelet_levels=levels,
        rescale_sigma=True,
        channel_axis=-1 if data.ndim == 3 else None,
    )
    return (strength * denoised + (1.0 - strength) * data).astype(np.float32)


def unsharp_mask(
    image: ArrayLike,
    *,
    radius: float = 2.0,
    amount: float = 0.2,
    threshold: float = 0.0,
) -> FloatImage:
    if radius <= 0 or amount < 0 or threshold < 0:
        raise ValueError("radius must be positive; amount and threshold must be non-negative.")
    data = as_float_image(image, ndim=None)
    sigma = (radius, radius, 0) if data.ndim == 3 else radius
    detail = data - gaussian_filter(data, sigma=sigma)
    if threshold:
        detail = np.where(np.abs(detail) >= threshold, detail, 0.0)
    return np.clip(data + amount * detail, 0.0, 1.0).astype(np.float32)


def local_contrast(
    image: ArrayLike,
    *,
    radius: float = 12.0,
    amount: float = 0.1,
    signal_floor: float = 0.025,
) -> FloatImage:
    """Enhance large-scale local detail with a signal-dependent mask."""
    data = as_float_image(image, ndim=None)
    mono = luminance(data) if data.ndim == 3 else data
    detail = mono - gaussian_filter(mono, sigma=radius)
    mask = np.clip((mono - signal_floor) / max(0.2, signal_floor), 0.0, 1.0)
    target = np.clip(mono + amount * detail * mask, 0.0, 1.0)
    if data.ndim == 3:
        return apply_luminance(data, target, ratio_limits=(0.8, 1.25))
    return target.astype(np.float32)

