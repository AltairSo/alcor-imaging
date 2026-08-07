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
class RenderConfig:
    """Configuration for color-preserving nonlinear RGB rendering.

    Percentiles are measured from the supplied image. Channel gains are explicit
    because filter response, atmosphere, and scientific/creative intent cannot be
    inferred reliably from image pixels alone.
    """

    background_percentile: float | None = 20.0
    background_offsets: tuple[float, float, float] | None = None
    channel_gains: tuple[float, float, float] = (1.0, 1.0, 1.0)
    white_percentile: float = 99.9
    faint_strength: float = 35.0
    highlight_strength: float = 6.0
    core_start: float = 0.08
    core_end: float = 0.65
    mask_blur_sigma: float = 2.0
    shadow_knee: float = 0.0015
    gamma: float = 0.88
    saturation: float = 0.9
    highlight_knee: float = 0.82


@dataclass(frozen=True, slots=True)
class NarrowbandConfig:
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    stacking: StackConfig = field(default_factory=StackConfig)
    stretch: StretchConfig = field(default_factory=StretchConfig)
    palette: str = "HOO"
    mixing_matrix: tuple[tuple[float, ...], ...] | None = None
    channel_boosts: tuple[float, ...] = ()
    crop_to_overlap: bool = True
    background_box_size: int | None = None
    denoise_strength: float = 0.0
    saturation: float = 1.0


@dataclass(frozen=True, slots=True)
class OSCConfig:
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    stacking: StackConfig = field(default_factory=StackConfig)
    stretch: StretchConfig = field(
        default_factory=lambda: StretchConfig(
            black_percentile=0.2,
            white_percentile=99.99,
            asinh_strength=10.0,
            shadow_protection=0.008,
            gamma=0.95,
        )
    )
    bayer_pattern: Literal["RGGB", "BGGR", "GRBG", "GBRG"] | None = None
    demosaic_method: Literal["malvar", "bilinear"] = "malvar"
    reference_index: int = 0
    crop_to_overlap: bool = True
    background_box_size: int | None = None
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
    denoise_strength: float = 0.2
    saturation: float = 1.1
    highlight_knee: float = 0.82


@dataclass(frozen=True, slots=True)
class LRGBConfig:
    registration: RegistrationConfig = field(
        default_factory=lambda: RegistrationConfig(downsample=4)
    )
    rgb_stretch: StretchConfig = field(
        default_factory=lambda: StretchConfig(
            black_percentile=0.2,
            white_percentile=99.995,
            asinh_strength=18.0,
            shadow_protection=0.006,
            gamma=0.88,
        )
    )
    luminance_stretch: StretchConfig = field(
        default_factory=lambda: StretchConfig(
            black_percentile=0.2,
            white_percentile=99.997,
            asinh_strength=20.0,
            shadow_protection=0.005,
            gamma=0.88,
        )
    )
    saturation_fraction: float = 0.98
    saturation_level: float | None = None
    saturation_dilation: int = 2
    background_percentile: float = 20.0
    crop_to_overlap: bool = True
    white_balance: tuple[float, float, float] = (1.0, 1.0, 1.0)
    luminance_weight: float = 0.75
    luminance_ratio_limits: tuple[float, float] = (0.65, 1.6)
    luminance_highlight_range: tuple[float, float] = (0.55, 0.9)
    denoise_strength: float = 0.1
    saturation: float = 1.12
    highlight_knee: float = 0.78
    render: RenderConfig | None = field(default_factory=RenderConfig)


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
class ChannelIntegrationResult:
    master: NDArray[np.float32]
    accepted_indices: list[int]
    rejected_indices: list[int]
    registrations: list[RegistrationRecord]
    unrecoverable_mask: NDArray[np.bool_] | None = None


@dataclass(slots=True)
class MasterAlignmentResult:
    masters: dict[str, NDArray[np.float32]]
    registrations: dict[str, RegistrationRecord]
    reference: str


@dataclass(slots=True)
class NarrowbandResult:
    linear_channels: tuple[NDArray[np.float32], ...]
    masters: tuple[NDArray[np.float32], ...]
    rgb: NDArray[np.float32]
    stacks: tuple[StackResult, ...]


@dataclass(slots=True)
class OSCResult:
    linear_rgb: NDArray[np.float32]
    rgb: NDArray[np.float32]
    accepted_indices: list[int]
    rejected_indices: list[int]
    registrations: list[RegistrationRecord]
    bayer_pattern: str


@dataclass(slots=True)
class LRGBResult:
    linear_luminance: NDArray[np.float32]
    linear_rgb: NDArray[np.float32]
    rgb: NDArray[np.float32]
    channel_masters: dict[str, NDArray[np.float32]]
    registrations: dict[str, list[RegistrationRecord]]
    accepted_indices: dict[str, list[int]]
    rejected_indices: dict[str, list[int]]
    luminance_unrecoverable_mask: NDArray[np.bool_]
