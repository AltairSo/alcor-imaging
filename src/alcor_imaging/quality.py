from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.stats import mad_std, sigma_clipped_stats
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter, label, maximum_filter

from ._validation import as_float_image


@dataclass(frozen=True, slots=True)
class FrameQuality:
    background: float
    noise: float
    star_count: int
    median_fwhm: float | None
    sharpness: float


def measure_frame(image: ArrayLike, *, detection_sigma: float = 5.0) -> FrameQuality:
    """Estimate background, robust noise, stars, FWHM, and Laplacian sharpness."""
    data = as_float_image(image)
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError("Image contains no finite pixels.")
    clean = data.copy()
    clean[~finite] = np.median(clean[finite])
    _, median, std = sigma_clipped_stats(clean, sigma=3.0, maxiters=5)
    noise = float(mad_std(clean, ignore_nan=True))
    smooth = gaussian_filter(clean, 1.0)
    threshold = median + detection_sigma * max(std, 1e-12)
    peaks = (smooth == maximum_filter(smooth, size=5)) & (smooth > threshold)
    labeled, star_count = label(peaks)
    fwhms: list[float] = []
    for object_index in range(1, star_count + 1):
        y, x = np.where(labeled == object_index)
        if y.size != 1:
            continue
        y0, x0 = int(y[0]), int(x[0])
        if y0 < 4 or x0 < 4 or y0 + 5 > clean.shape[0] or x0 + 5 > clean.shape[1]:
            continue
        patch = clean[y0 - 4 : y0 + 5, x0 - 4 : x0 + 5] - median
        patch = np.clip(patch, 0, None)
        total = float(patch.sum())
        if total <= 0:
            continue
        yy, xx = np.indices(patch.shape)
        cx = float((xx * patch).sum() / total)
        cy = float((yy * patch).sum() / total)
        variance = float((((xx - cx) ** 2 + (yy - cy) ** 2) * patch).sum() / (2 * total))
        if variance > 0:
            fwhms.append(2.3548 * np.sqrt(variance))
    laplacian = (
        -4 * clean
        + np.roll(clean, 1, 0)
        + np.roll(clean, -1, 0)
        + np.roll(clean, 1, 1)
        + np.roll(clean, -1, 1)
    )
    return FrameQuality(
        background=float(median),
        noise=noise,
        star_count=int(star_count),
        median_fwhm=float(np.median(fwhms)) if fwhms else None,
        sharpness=float(np.var(laplacian[2:-2, 2:-2])),
    )

