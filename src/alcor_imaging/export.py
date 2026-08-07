from __future__ import annotations

from os import PathLike

import numpy as np
import tifffile
from numpy.typing import ArrayLike
from PIL import Image

from ._validation import as_float_image


def quantize(image: ArrayLike, *, bits: int = 16) -> np.ndarray:
    """Quantize normalized data without silently rescaling its dynamic range."""
    if bits not in (8, 16):
        raise ValueError("bits must be 8 or 16.")
    data = np.clip(as_float_image(image, ndim=None), 0.0, 1.0)
    maximum = (1 << bits) - 1
    dtype = np.uint8 if bits == 8 else np.uint16
    return np.round(data * maximum).astype(dtype)


def write_tiff(
    destination: str | PathLike[str],
    image: ArrayLike,
    *,
    bits: int | str = 16,
) -> None:
    """Write an 8/16-bit display TIFF or a 32-bit floating-point linear TIFF."""
    data = as_float_image(image, ndim=None)
    if data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[-1] not in (3, 4)):
        raise ValueError("TIFF input must be mono, RGB, or RGBA.")
    if bits == "float32":
        output = data.astype(np.float32)
    elif bits in (8, 16):
        output = quantize(data, bits=int(bits))
    else:
        raise ValueError("bits must be 8, 16, or 'float32'.")
    tifffile.imwrite(
        destination,
        output,
        photometric="rgb" if data.ndim == 3 else "minisblack",
        metadata={"axes": "YXS" if data.ndim == 3 else "YX"},
    )


def write_png(destination: str | PathLike[str], image: ArrayLike) -> None:
    """Write an 8-bit display PNG. Use TIFF or FITS for scientific/linear output."""
    data = as_float_image(image, ndim=None)
    if data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[-1] not in (3, 4)):
        raise ValueError("PNG input must be mono, RGB, or RGBA.")
    Image.fromarray(quantize(data, bits=8)).save(destination)

