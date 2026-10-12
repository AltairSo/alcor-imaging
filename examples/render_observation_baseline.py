#!/usr/bin/env python3
"""Render an auditable RGB-only baseline from this observation's linear masters.

Reference-star gains are retained. No external image, spatial enhancement, L
injection, saturation repair, or local colour adjustment enters this renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile
from astropy.io import fits
from PIL import Image, ImageCms, ImageDraw
from process_nina_lrgb import bin_mean, render_tile, save_json
from scipy.ndimage import binary_dilation, map_coordinates


def encode_srgb(linear):
    """IEC sRGB transfer, retaining negative noise until final display clipping."""
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055,
    )


def baseline_tile(masters, params, strength=15.0, pedestal=0.035):
    if strength <= 0 or not 0 <= pedestal < 1 or params["white"] <= 0:
        raise ValueError("Invalid tone parameters")
    rgb = (
        np.stack(
            [
                (masters[c] - params["backgrounds_adu_per_sec"][c]) * g
                for c, g in zip("RGB", params["rgb_gains"], strict=True)
            ],
            axis=-1,
        )
        / params["white"]
    )
    # A single pointwise, monotonic curve on the largest channel, preserving
    # linear channel ratios. Rational shoulder approaches white without clipping.
    peak = np.maximum(rgb.max(-1), 1e-8)
    curve = np.arcsinh(strength * peak) / np.arcsinh(strength)
    linear_display = rgb * (curve / ((1 + curve) * peak))[..., None]
    encoded = encode_srgb(linear_display)
    return np.clip(pedestal + (1 - pedestal) * encoded, 0, 1).astype(np.float32)


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def comparison(images, labels, path, width=950):
    panels = []
    for image, label in zip(images, labels, strict=True):
        panel = Image.fromarray(np.uint8(np.rint(np.clip(image, 0, 1) * 255)))
        panel.thumbnail((width, width), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, panel.height + 32), "#141414")
        canvas.paste(panel, ((width - panel.width) // 2, 32))
        ImageDraw.Draw(canvas).text((12, 10), label, fill="white")
        panels.append(canvas)
    sheet = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width
    sheet.save(path, icc_profile=ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())


def core_audit(out, masters, params, crop, strength, pedestal, folder):
    """Map raw clipping flags into the actual registered core; never repair pixels."""
    summary = json.loads((out / "reports/validation_summary.json").read_text())
    cx, cy = summary["core_xy_in_cropped_frame"]
    h, w = masters["R"].shape
    left, right = max(0, cx - 256), min(w, cx + 256)
    top, bottom = max(0, cy - 256), min(h, cy + 256)
    patch = {c: a[top:bottom, left:right] for c, a in masters.items()}
    baseline = baseline_tile(patch, params, strength, pedestal)
    enhanced = render_tile(patch, params, "enhanced")
    yy, xx = np.mgrid[top:bottom, left:right]
    coords = np.stack([xx + crop[2], yy + crop[0], np.ones_like(xx)], axis=-1)
    records = json.loads((out / "reports/registration_report.json").read_text())
    transforms = {r["name"]: np.array(r["matrix"]) for r in records}
    rows = json.loads((out / "reports/metadata_summary.json").read_text())
    all_masks, audit = [], []
    for channel in "RGB":
        masks, exact = [], []
        for frame in (r for r in rows if r["filter"] == channel):
            original = coords @ np.linalg.inv(transforms[frame["name"]]).T
            x, y = original[..., 0], original[..., 1]
            x0, x1 = max(0, int(x.min()) - 2), min(frame["width"], int(x.max()) + 4)
            y0, y1 = max(0, int(y.min()) - 2), min(frame["height"], int(y.max()) + 4)
            with fits.open(frame["path"], memmap=False) as hdus:
                raw = hdus[0].section[y0:y1, x0:x1]
            near = map_coordinates(
                (raw >= 0.98 * frame["saturation"]).astype(np.uint8),
                [y - y0, x - x0],
                order=0,
                mode="constant",
                cval=0,
            ).astype(bool)
            clipped = map_coordinates(
                (raw >= frame["saturation"]).astype(np.uint8),
                [y - y0, x - x0],
                order=0,
                mode="constant",
                cval=0,
            ).astype(bool)
            masks.append(near)
            exact.append(clipped)
        near_any = np.any(masks, axis=0)
        all_masks.append(near_any)
        audit.append(
            {
                "channel": channel,
                "near_saturation_any_exposure_pixels": int(near_any.sum()),
                "near_saturation_every_exposure_pixels": int(np.all(masks, axis=0).sum()),
                "detector_ceiling_every_exposure_pixels": int(np.all(exact, axis=0).sum()),
            }
        )
    unreliable = np.any(all_masks, axis=0)
    marked = baseline.copy()
    marked[binary_dilation(unreliable, iterations=2)] = [1, 0.3, 0]
    comparison(
        [enhanced, baseline, marked],
        [
            "Previous enhanced LRGB / native crop",
            "New baseline RGB / native crop",
            "Orange = near saturation in >=1 RGB exposure",
        ],
        folder / "core_comparison.png",
        width=512,
    )
    return {
        "crop_xyxy": [left, top, right, bottom],
        "threshold_fraction": 0.98,
        "channels": audit,
        "note": "Nearest-neighbour raw flags; not a repair mask.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--factor", type=int, choices=[1, 4], default=1)
    parser.add_argument("--strength", type=float, default=15)
    parser.add_argument("--pedestal", type=float, default=0.035)
    args = parser.parse_args()
    out, work = Path(args.output), Path(args.work)
    folder = out / "reports/observation_baseline"
    folder.mkdir(parents=True, exist_ok=True)
    for directory in ["previews", "final", "scripts"]:
        (out / directory).mkdir(exist_ok=True)
    source = out / "reports/colour_calibration/render_parameters.json"
    params = json.loads(source.read_text())
    crop = json.loads((out / f"reports/crop_bin{args.factor}.json").read_text())
    y0, y1, x0, x1 = crop
    paths = {c: work / f"master_{c}_bin{args.factor}.npy" for c in "RGBL"}
    masters = {c: np.load(p, mmap_mode="r")[y0:y1, x0:x1] for c, p in paths.items()}
    h, w = masters["R"].shape
    result = np.lib.format.open_memmap(
        work / f"observation_baseline_bin{args.factor}.npy",
        mode="w+",
        dtype=np.float32,
        shape=(h, w, 3),
    )
    black, white = 0, 0
    for y in range(0, h, 256):
        tile = baseline_tile(
            {c: a[y : y + 256] for c, a in masters.items()}, params, args.strength, args.pedestal
        )
        result[y : y + 256] = tile
        quantized = np.rint(tile * 255)
        black += int(np.all(quantized == 0, axis=-1).sum())
        white += int(np.any(quantized == 255, axis=-1).sum())
    result.flush()
    factor = max(1, int(np.ceil(w / 2800)))
    preview = np.stack([bin_mean(result[..., i], factor) for i in range(3)], axis=-1)
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    stem = f"{args.target}_observation_baseline_RGB"
    Image.fromarray(np.uint8(np.rint(preview * 255))).save(
        out / f"previews/{stem}_preview.png",
        icc_profile=profile,
    )
    previous = (
        np.asarray(Image.open(out / f"previews/{args.target}_reference_enhanced_preview.png")) / 255
    )
    comparison(
        [previous, preview],
        [
            "Previous: reference-star calibrated / enhanced LRGB",
            "New: observation-only RGB baseline",
        ],
        folder / "baseline_comparison.png",
    )
    products = []
    if args.factor == 1:
        tif = out / f"final/{stem}_fullres_16bit.tif"
        tifffile.imwrite(
            tif,
            np.uint16(np.rint(result * 65535)),
            photometric="rgb",
            iccprofile=profile,
            metadata={"axes": "YXS"},
        )
        png = out / f"final/{stem}_fullres.png"
        Image.fromarray(np.uint8(np.rint(result * 255))).save(png, icc_profile=profile)
        with tifffile.TiffFile(tif) as tf:
            assert tf.series[0].shape == (h, w, 3) and tf.series[0].dtype == np.uint16
        products = [str(tif), str(png)]
    audit = (
        core_audit(out, masters, params, crop, args.strength, args.pedestal, folder)
        if args.factor == 1
        else None
    )
    rows = json.loads((out / "reports/metadata_summary.json").read_text())
    report = {
        "description": "Observed R/G/B only; L defines registration coordinates but adds no flux.",
        "external_image_pixels": False,
        "generated_detail": False,
        "spatial_enhancement": False,
        "luminance_injection": False,
        "rgb_gains": params["rgb_gains"],
        "colour_reference": params["colour_method"],
        "tone_curve": "Linked peak asinh then t/(1+t), sRGB encoding, neutral display pedestal",
        "srgb_specification": "https://www.w3.org/Graphics/Color/srgb",
        "strength": args.strength,
        "display_pedestal": args.pedestal,
        "white_scale": params["white"],
        "backgrounds": params["backgrounds_adu_per_sec"],
        "dimensions_wh": [w, h],
        "all_channels_zero_8bit_percent": 100 * black / (h * w),
        "any_channel_255_8bit_percent": 100 * white / (h * w),
        "rgb_exposure_seconds": sum(r["exposure"] for r in rows if r["filter"] in "RGB"),
        "captured_total_seconds": sum(r["exposure"] for r in rows),
        "rgb_input_files": [
            {k: r[k] for k in ["name", "filter", "exposure", "sha256"]}
            for r in rows
            if r["filter"] in "RGB"
        ],
        "master_sha256": {c: file_hash(paths[c]) for c in "RGB"},
        "render_script_sha256": file_hash(__file__),
        "calibration_parameters_sha256": file_hash(source),
        "core_saturation": audit,
        "products": products,
        "limitations": [
            "Raw inputs lack dark/flat calibration.",
            "Gaia calibration corrects broadband balance; it does not recover "
            "emission-line spectra or CIE tristimulus colours.",
            "sRGB encoding is a display convention; "
            "instrument RGB primaries are only approximations.",
            "Some core pixels are saturated; their true brightness and colours are unavailable.",
            "Noise and gradients are retained. This is a diagnostic baseline, "
            "not a production-finished image.",
        ],
    }
    save_json(folder / "baseline_report.json", report)
    shutil.copy2(__file__, out / "scripts/render_observation_baseline.py")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
