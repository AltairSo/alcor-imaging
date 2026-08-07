from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import ArrayLike

from ._validation import as_float_image
from .models import Frame


def read_fits(
    source: str | PathLike[str],
    *,
    hdu: int | str = 0,
    plane: int | None = None,
    memmap: bool = False,
) -> Frame:
    """Read one 2D science plane from FITS/FIT/FTS without altering its values."""
    path = Path(source)
    with fits.open(path, memmap=memmap, do_not_scale_image_data=False) as hdul:
        selected = hdul[hdu]
        if selected.data is None:
            raise ValueError(f"HDU {hdu!r} in {path} has no image data.")
        data = np.asarray(selected.data)
        header = dict(selected.header)

    data = np.squeeze(data)
    if plane is not None:
        if data.ndim != 3:
            raise ValueError(f"plane was provided but FITS data has shape {data.shape}.")
        data = data[plane]
    return Frame(data=as_float_image(data), header=header, source=str(path))


def write_fits(
    destination: str | PathLike[str],
    image: ArrayLike,
    *,
    header: Mapping[str, Any] | None = None,
    overwrite: bool = False,
    rgb_axis: int | None = None,
) -> None:
    """Write a mono or RGB float32 image while preserving supplied FITS metadata."""
    data = as_float_image(image, ndim=None)
    if data.ndim not in (2, 3):
        raise ValueError(f"FITS output must be 2D or 3D, received shape {data.shape}.")
    if rgb_axis is not None and data.ndim == 3:
        data = np.moveaxis(data, rgb_axis, 0)

    fits_header = fits.Header()
    if header:
        for key, value in header.items():
            try:
                fits_header[key] = value
            except (ValueError, TypeError):
                continue
    fits.writeto(destination, data.astype(np.float32), header=fits_header, overwrite=overwrite)

