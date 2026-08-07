"""Alcor Imaging: array-first astronomical image processing."""

from .background import estimate_background, repair_nonfinite, subtract_background
from .calibration import calibrate, calibrate_many, make_master
from .channels import align_mono_masters, integrate_mono_channel
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
from .hdr import hdr_combine, hdr_combine_with_mask
from .models import (
    CalibrationSet,
    ChannelIntegrationResult,
    Frame,
    LRGBConfig,
    LRGBResult,
    MasterAlignmentResult,
    NarrowbandConfig,
    NarrowbandResult,
    OSCConfig,
    OSCResult,
    RegistrationConfig,
    RegistrationRecord,
    RenderConfig,
    StackConfig,
    StackResult,
    StretchConfig,
)
from .pipeline import process_lrgb, process_narrowband, process_osc
from .quality import FrameQuality, measure_frame
from .registration import (
    apply_transform,
    estimate_transform,
    register_image,
    register_many,
    register_rgb_many,
    registration_stretch,
)
from .render import (
    dual_asinh_stretch_rgb,
    estimate_background_offsets,
    neutralize_background,
    render_rgb,
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
    "ChannelIntegrationResult",
    "Frame",
    "FrameQuality",
    "LRGBConfig",
    "LRGBResult",
    "MasterAlignmentResult",
    "NarrowbandConfig",
    "NarrowbandResult",
    "OSCConfig",
    "OSCResult",
    "RegistrationConfig",
    "RegistrationRecord",
    "RenderConfig",
    "StackConfig",
    "StackResult",
    "StretchConfig",
    "adjust_saturation",
    "align_mono_masters",
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
    "dual_asinh_stretch_rgb",
    "estimate_background",
    "estimate_background_offsets",
    "estimate_transform",
    "hdr_combine",
    "hdr_combine_with_mask",
    "infer_bayer_pattern",
    "integrate_mono_channel",
    "local_contrast",
    "luminance",
    "make_master",
    "masked_asinh_stretch",
    "measure_frame",
    "midtone_transfer",
    "mosaic_rgb",
    "neutralize_background",
    "normalize",
    "normalize_bayer_pattern",
    "overlap_bounds",
    "process_lrgb",
    "process_narrowband",
    "process_osc",
    "quantize",
    "read_fits",
    "register_and_stack",
    "register_image",
    "register_many",
    "register_rgb_many",
    "registration_stretch",
    "render_rgb",
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

__version__ = "0.3.1"
