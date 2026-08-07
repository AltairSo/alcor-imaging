"""Alcor Imaging: array-first astronomical image processing."""

from .background import estimate_background, repair_nonfinite, subtract_background
from .calibration import calibrate, calibrate_many, make_master
from .color import (
    PALETTES,
    adjust_saturation,
    apply_luminance,
    apply_palette,
    channel_balance,
    combine_channels,
    luminance,
)
from .demosaic import (
    VALID_BAYER_PATTERNS,
    bayer_masks,
    demosaic,
    infer_bayer_pattern,
    mosaic_rgb,
    normalize_bayer_pattern,
)
from .enhance import local_contrast, unsharp_mask, wavelet_denoise
from .export import quantize, write_png, write_tiff
from .fits import read_fits, write_fits
from .geometry import crop_to_overlap, overlap_bounds
from .models import (
    CalibrationSet,
    Frame,
    NarrowbandConfig,
    NarrowbandResult,
    OSCConfig,
    OSCResult,
    RegistrationConfig,
    RegistrationRecord,
    StackConfig,
    StackResult,
    StretchConfig,
)
from .pipeline import process_narrowband, process_osc
from .quality import FrameQuality, measure_frame
from .registration import (
    apply_transform,
    estimate_transform,
    register_image,
    register_many,
    register_rgb_many,
    registration_stretch,
)
from .stacking import register_and_stack, stack, stack_rgb
from .stretch import (
    asinh_stretch,
    masked_asinh_stretch,
    midtone_transfer,
    normalize,
    soft_clip,
    stretch,
    stretch_rgb,
)

__all__ = [
    "PALETTES",
    "VALID_BAYER_PATTERNS",
    "CalibrationSet",
    "Frame",
    "FrameQuality",
    "NarrowbandConfig",
    "NarrowbandResult",
    "OSCConfig",
    "OSCResult",
    "RegistrationConfig",
    "RegistrationRecord",
    "StackConfig",
    "StackResult",
    "StretchConfig",
    "adjust_saturation",
    "apply_luminance",
    "apply_palette",
    "apply_transform",
    "asinh_stretch",
    "bayer_masks",
    "calibrate",
    "calibrate_many",
    "channel_balance",
    "combine_channels",
    "crop_to_overlap",
    "demosaic",
    "estimate_background",
    "estimate_transform",
    "infer_bayer_pattern",
    "local_contrast",
    "luminance",
    "make_master",
    "masked_asinh_stretch",
    "measure_frame",
    "midtone_transfer",
    "mosaic_rgb",
    "normalize",
    "normalize_bayer_pattern",
    "overlap_bounds",
    "process_narrowband",
    "process_osc",
    "quantize",
    "read_fits",
    "register_and_stack",
    "register_image",
    "register_many",
    "register_rgb_many",
    "registration_stretch",
    "repair_nonfinite",
    "soft_clip",
    "stack",
    "stack_rgb",
    "stretch",
    "stretch_rgb",
    "subtract_background",
    "unsharp_mask",
    "wavelet_denoise",
    "write_fits",
    "write_png",
    "write_tiff",
]

__version__ = "0.2.1"
