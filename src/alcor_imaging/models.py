from __future__ import annotations

from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

ImageSource: TypeAlias = str | PathLike[str] | NDArray[np.number[Any]]


@dataclass(slots=True)
class Frame:
    """A floating-point image and its metadata.

    Data is always a two-dimensional float32 array. The header is deliberately
    a plain mapping so callers are not coupled to an Astropy header object.
    """

    data: NDArray[np.float32]
    header: dict[str, Any] = field(default_factory=dict)
    source: str | None = None


@dataclass(frozen=True, slots=True)
class CalibrationSet:
    bias: NDArray[np.floating[Any]] | None = None
    dark: NDArray[np.floating[Any]] | None = None
    flat: NDArray[np.floating[Any]] | None = None
    dark_exposure: float | None = None


@dataclass(frozen=True, slots=True)
class RegistrationConfig:
    downsample: int = 2
    max_control_points: int = 100
    detection_sigma: float = 3.0
    min_area: int = 5
    transform: Literal["similarity"] = "similarity"
    fill_value: float = np.nan


@dataclass(frozen=True, slots=True)
class StackConfig:
    method: Literal["mean", "median", "sigma_clip_mean", "sigma_clip_median"] = (
        "sigma_clip_median"
    )
    sigma: float = 3.0
    max_iterations: int = 5
    normalization: Literal["none", "median", "multiplicative"] = "none"
    tile_size: int | None = 512


@dataclass(frozen=True, slots=True)
class StretchConfig:
    black_percentile: float = 0.8
    white_percentile: float = 99.9
    asinh_strength: float = 7.0
    shadow_protection: float = 0.015
    gamma: float = 1.0


@dataclass(frozen=True, slots=True)
class NarrowbandConfig:
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    stacking: StackConfig = field(default_factory=StackConfig)
    stretch: StretchConfig = field(default_factory=StretchConfig)
    palette: Literal["HOO", "SHO"] = "HOO"
    channel_boosts: tuple[float, ...] = ()
    crop_to_overlap: bool = True
    background_box_size: int | None = None
    denoise_strength: float = 0.0
    saturation: float = 1.0


@dataclass(slots=True)
class RegistrationRecord:
    index: int
    accepted: bool
    rotation_degrees: float | None = None
    translation: tuple[float, float] | None = None
    error: str | None = None


@dataclass(slots=True)
class StackResult:
    image: NDArray[np.float32]
    accepted_indices: list[int]
    rejected_indices: list[int]
    registrations: list[RegistrationRecord]


@dataclass(slots=True)
class NarrowbandResult:
    linear_channels: tuple[NDArray[np.float32], ...]
    masters: tuple[NDArray[np.float32], ...]
    rgb: NDArray[np.float32]
    stacks: tuple[StackResult, ...]
