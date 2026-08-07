from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import convolve

from ._validation import FloatImage, as_float_image

BayerPattern = Literal["RGGB", "BGGR", "GRBG", "GBRG"]
VALID_BAYER_PATTERNS = frozenset({"RGGB", "BGGR", "GRBG", "GBRG"})


def normalize_bayer_pattern(pattern: str) -> BayerPattern:
    normalized = "".join(character for character in str(pattern).upper() if character in "RGB")
    if normalized not in VALID_BAYER_PATTERNS:
        raise ValueError(
            f"Unsupported Bayer pattern {pattern!r}; expected RGGB, BGGR, GRBG, or GBRG."
        )
    return normalized  # type: ignore[return-value]


def infer_bayer_pattern(header: Mapping[str, Any]) -> BayerPattern | None:
    """Read a standard or commonly used Bayer-pattern FITS header value."""
    for key in ("BAYERPAT", "BAYERPATN", "COLORTYP", "CFA"):
        value = header.get(key)
        if value is None:
            continue
        try:
            return normalize_bayer_pattern(str(value))
        except ValueError:
            continue
    return None


def bayer_masks(shape: tuple[int, int], pattern: str) -> tuple[np.ndarray, ...]:
    """Return boolean red, green, and blue masks for a Bayer mosaic shape."""
    normalized = normalize_bayer_pattern(pattern)
    masks = {channel: np.zeros(shape, dtype=bool) for channel in "RGB"}
    for index, channel in enumerate(normalized):
        row, col = divmod(index, 2)
        masks[channel][row::2, col::2] = True
    return masks["R"], masks["G"], masks["B"]


def mosaic_rgb(rgb: ArrayLike, pattern: str = "RGGB") -> FloatImage:
    """Sample an RGB image through a Bayer CFA, primarily for simulation and testing."""
    data = as_float_image(rgb, ndim=3)
    if data.shape[-1] != 3:
        raise ValueError("RGB input must have three channels on the last axis.")
    red, green, blue = bayer_masks(data.shape[:2], pattern)
    return (
        data[..., 0] * red + data[..., 1] * green + data[..., 2] * blue
    ).astype(np.float32)


def _bilinear(cfa: FloatImage, pattern: str) -> FloatImage:
    masks = bayer_masks(cfa.shape, pattern)
    kernel = np.asarray([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    channels = []
    for mask in masks:
        numerator = convolve(cfa * mask, kernel, mode="mirror")
        denominator = convolve(mask.astype(np.float32), kernel, mode="mirror")
        channels.append(
            np.divide(
                numerator,
                denominator,
                out=np.zeros_like(cfa),
                where=denominator > 0,
            )
        )
    return np.stack(channels, axis=-1).astype(np.float32)


def _malvar(cfa: FloatImage, pattern: str) -> FloatImage:
    """Malvar-He-Cutler linear demosaicing with 5-by-5 gradient correction."""
    red_mask, green_mask, blue_mask = bayer_masks(cfa.shape, pattern)

    green_at_red_blue = np.asarray(
        [
            [0, 0, -1, 0, 0],
            [0, 0, 2, 0, 0],
            [-1, 2, 4, 2, -1],
            [0, 0, 2, 0, 0],
            [0, 0, -1, 0, 0],
        ],
        dtype=np.float32,
    ) / 8
    red_blue_at_green_h = np.asarray(
        [
            [0, 0, 0.5, 0, 0],
            [0, -1, 0, -1, 0],
            [-1, 4, 5, 4, -1],
            [0, -1, 0, -1, 0],
            [0, 0, 0.5, 0, 0],
        ],
        dtype=np.float32,
    ) / 8
    red_blue_at_green_v = red_blue_at_green_h.T
    red_at_blue = np.asarray(
        [
            [0, 0, -1.5, 0, 0],
            [0, 2, 0, 2, 0],
            [-1.5, 0, 6, 0, -1.5],
            [0, 2, 0, 2, 0],
            [0, 0, -1.5, 0, 0],
        ],
        dtype=np.float32,
    ) / 8

    green_interp = convolve(cfa, green_at_red_blue, mode="mirror")
    horizontal_interp = convolve(cfa, red_blue_at_green_h, mode="mirror")
    vertical_interp = convolve(cfa, red_blue_at_green_v, mode="mirror")
    opposite_interp = convolve(cfa, red_at_blue, mode="mirror")

    red = np.where(red_mask, cfa, 0.0)
    green = np.where(green_mask, cfa, green_interp)
    blue = np.where(blue_mask, cfa, 0.0)

    red_rows = np.any(red_mask, axis=1)[:, None]
    red_cols = np.any(red_mask, axis=0)[None, :]
    blue_rows = np.any(blue_mask, axis=1)[:, None]
    blue_cols = np.any(blue_mask, axis=0)[None, :]

    red = np.where(green_mask & red_rows, horizontal_interp, red)
    red = np.where(green_mask & red_cols, vertical_interp, red)
    blue = np.where(green_mask & blue_rows, horizontal_interp, blue)
    blue = np.where(green_mask & blue_cols, vertical_interp, blue)
    red = np.where(blue_mask, opposite_interp, red)
    blue = np.where(red_mask, opposite_interp, blue)
    return np.stack((red, green, blue), axis=-1).astype(np.float32)


def demosaic(
    cfa: ArrayLike,
    pattern: str,
    *,
    method: Literal["malvar", "bilinear"] = "malvar",
) -> FloatImage:
    """Convert a 2D Bayer CFA image to full-resolution linear RGB."""
    data = as_float_image(cfa)
    if method == "malvar":
        return _malvar(data, pattern)
    if method == "bilinear":
        return _bilinear(data, pattern)
    raise ValueError("method must be 'malvar' or 'bilinear'.")

