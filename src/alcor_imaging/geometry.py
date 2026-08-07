from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatImage, as_image_sequence


def overlap_bounds(images: Sequence[ArrayLike]) -> tuple[slice, slice]:
    """Return the bounding rectangle of pixels finite in every input image."""
    prepared = as_image_sequence(images)
    valid = np.logical_and.reduce([np.isfinite(image) for image in prepared])
    rows, cols = np.where(valid)
    if rows.size == 0:
        raise ValueError("Images have no common finite overlap.")
    return slice(int(rows.min()), int(rows.max()) + 1), slice(int(cols.min()), int(cols.max()) + 1)


def crop_to_overlap(
    images: Sequence[ArrayLike], *, repair_holes: bool = False
) -> list[FloatImage]:
    prepared = as_image_sequence(images)
    row_slice, col_slice = overlap_bounds(prepared)
    result = [image[row_slice, col_slice].copy() for image in prepared]
    if repair_holes:
        for image in result:
            finite = np.isfinite(image)
            if not np.all(finite):
                image[~finite] = np.nanmedian(image)
    return result

