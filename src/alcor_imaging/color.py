from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatImage, as_float_image, as_image_sequence

PALETTES: dict[str, np.ndarray] = {
    # Input order: Ha, OIII
    "HOO": np.asarray([[1.0, 0.0], [0.22, 0.78], [0.0, 1.0]], dtype=np.float32),
    # Input order: SII, Ha, OIII (Hubble palette)
    "SHO": np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
}


def combine_channels(
    channels: Sequence[ArrayLike],
    matrix: ArrayLike,
    *,
    clip: bool = True,
) -> FloatImage:
    """Map N monochrome channels to RGB with an explicit 3-by-N mixing matrix."""
    prepared = as_image_sequence(channels)
    mixing = np.asarray(matrix, dtype=np.float32)
    if mixing.shape != (3, len(prepared)):
        raise ValueError(f"matrix must have shape (3, {len(prepared)}), received {mixing.shape}.")
    cube = np.stack(prepared, axis=-1)
    rgb = np.einsum("...c,rc->...r", cube, mixing, optimize=True)
    if clip:
        rgb = np.clip(rgb, 0.0, 1.0)
    return rgb.astype(np.float32)


def apply_palette(channels: Sequence[ArrayLike], palette: str | ArrayLike = "HOO") -> FloatImage:
    matrix = PALETTES.get(palette.upper()) if isinstance(palette, str) else palette
    if matrix is None:
        raise ValueError(f"Unknown palette {palette!r}; available palettes: {sorted(PALETTES)}.")
    return combine_channels(channels, matrix)


def luminance(rgb: ArrayLike) -> FloatImage:
    data = as_float_image(rgb, ndim=3)
    if data.shape[-1] != 3:
        raise ValueError("RGB input must have three channels on the last axis.")
    return np.einsum("...c,c->...", data, (0.2126, 0.7152, 0.0722)).astype(np.float32)


def adjust_saturation(rgb: ArrayLike, factor: float = 1.0) -> FloatImage:
    if factor < 0:
        raise ValueError("factor cannot be negative.")
    data = as_float_image(rgb, ndim=3)
    lightness = luminance(data)[..., None]
    return np.clip(lightness + factor * (data - lightness), 0.0, 1.0).astype(np.float32)


def apply_luminance(
    rgb: ArrayLike,
    target: ArrayLike,
    *,
    minimum: float = 0.01,
    ratio_limits: tuple[float, float] = (0.5, 2.0),
) -> FloatImage:
    """Inject a luminance image while preserving chromatic ratios."""
    data = as_float_image(rgb, ndim=3)
    target_data = as_float_image(target)
    if target_data.shape != data.shape[:2]:
        raise ValueError("Target luminance must match RGB spatial dimensions.")
    current = luminance(data)
    ratio = target_data / np.maximum(current, minimum)
    ratio = np.clip(ratio, *ratio_limits)
    return np.clip(data * ratio[..., None], 0.0, 1.0).astype(np.float32)


def channel_balance(
    rgb: ArrayLike,
    gains: tuple[float, float, float],
    *,
    clip: bool = True,
) -> FloatImage:
    data = as_float_image(rgb, ndim=3)
    if any(gain < 0 for gain in gains):
        raise ValueError("Channel gains cannot be negative.")
    result = data * np.asarray(gains, dtype=np.float32)
    if clip:
        result = np.clip(result, 0.0, 1.0)
    return result.astype(np.float32)
