#!/usr/bin/env python3
"""Natural Enhanced LRGB, with per-filter rejection of clipped exposure samples.

Only this observation's frames supply image pixels. Catalogue colour gains stay
fixed. No missing channel is inferred from another channel or external image.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import tifffile
from astropy.io import fits
from PIL import Image, ImageCms
from process_nina_lrgb import bin_mean, save_json
from render_observation_baseline import comparison, encode_srgb, file_hash
from scipy.ndimage import gaussian_filter

from alcor_imaging import warp_affine

WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def classify_availability(counts, frame_counts):
    """0 clean; 1 partial rejection; 2 missing RGB channel(s); 3 all RGB missing."""
    values = np.stack([counts[c] for c in "RGB"])
    missing = values == 0
    affected = np.any(values < np.array([frame_counts[c] for c in "RGB"])[:, None, None], axis=0)
    return np.where(missing.all(0), 3, np.where(missing.any(0), 2, affected)).astype(np.uint8)


def reject_clipped(cube, flags, fallback):
    valid = ~flags & np.isfinite(cube)
    count = valid.sum(0).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmedian(np.where(valid, cube, np.nan), axis=0)
    # Preserve measured lower bounds for display, explicitly flagged count==0.
    # Scientific consumers must mask these; this is not a recovered flux estimate.
    return np.where(count > 0, result, fallback).astype(np.float32), count


def prepare(out, work, crop, folder):
    rows = json.loads((out / "reports/metadata_summary.json").read_text())
    records = json.loads((out / "reports/registration_report.json").read_text())
    transforms = {r["name"]: np.array(r["matrix"]) for r in records}
    y0, y1, x0, x1 = crop
    masters, counts, ceiling_counts, summaries = {}, {}, {}, {}
    for channel in "LRGB":
        selected = [r for r in rows if r["filter"] == channel]
        if len({r["exposure"] for r in selected}) != 1:
            raise ValueError(
                "This pass requires equal exposures within each filter; "
                "use the HDR integrator for mixed exposures."
            )
        flags, ceilings, aligned = [], [], []
        for r in selected:
            # Verify that the raw files still match the recorded observation.
            if file_hash(r["path"]) != r["sha256"]:
                raise ValueError("Raw input changed: " + r["name"])
            raw = fits.getdata(r["path"])
            for fraction, suffix in [(0.98, "near"), (1.0, "ceiling")]:
                flag = (raw >= fraction * r["saturation"]).astype(np.float32)
                projected = warp_affine(
                    flag, transforms[r["name"]], flag.shape, order=1, backend="cpu"
                )
                # Near: reject if ANY contributing detector sample is near ceiling.
                # Ceiling: definite full clipping only if ALL contributors reach ceiling.
                mask = projected > 1e-6 if suffix == "near" else projected >= 1 - 1e-6
                path = work / f"{Path(r['name']).stem}_{suffix}_support.npy"
                np.save(path, mask[y0:y1, x0:x1])
                (flags if suffix == "near" else ceilings).append(np.load(path, mmap_mode="r"))
                del flag, projected, mask
            del raw
            cache = work / f"{Path(r['name']).stem}_aligned_bin1.npy"
            aligned.append(np.load(cache, mmap_mode="r")[y0:y1, x0:x1])
            print("Mapped raw saturation:", r["name"], flush=True)
            gc.collect()
        original = np.load(work / f"master_{channel}_bin1.npy", mmap_mode="r")[y0:y1, x0:x1]
        masters[channel] = np.lib.format.open_memmap(
            work / f"safe_{channel}.npy", mode="w+", dtype="float32", shape=original.shape
        )
        counts[channel] = np.lib.format.open_memmap(
            work / f"valid_{channel}.npy", mode="w+", dtype="uint8", shape=original.shape
        )
        ceiling_counts[channel] = np.lib.format.open_memmap(
            work / f"ceiling_{channel}.npy", mode="w+", dtype="uint8", shape=original.shape
        )
        for y in range(0, original.shape[0], 128):
            cube = np.array(
                [a[y : y + 128] / selected[i]["exposure"] for i, a in enumerate(aligned)]
            )
            combined, number = reject_clipped(
                cube, np.array([a[y : y + 128] for a in flags]), original[y : y + 128]
            )
            # Exactly retain original integration wherever no sample was rejected.
            combined[number == len(selected)] = original[y : y + 128][number == len(selected)]
            masters[channel][y : y + 128] = combined
            counts[channel][y : y + 128] = number
            ceiling_counts[channel][y : y + 128] = np.sum(
                [a[y : y + 128] for a in ceilings], axis=0
            )
        for a in [masters[channel], counts[channel], ceiling_counts[channel]]:
            a.flush()
        n = counts[channel]
        summaries[channel] = {
            "exposures": len(selected),
            "all_samples_usable_pixels": int(np.sum(n == len(selected))),
            "some_rejected_with_usable_remaining_pixels": int(
                np.sum((n > 0) & (n < len(selected)))
            ),
            "no_fully_usable_exposure_pixels": int(np.sum(n == 0)),
            "entire_support_at_detector_ceiling_every_exposure_pixels": int(
                np.sum(ceiling_counts[channel] == len(selected))
            ),
        }
        print(channel, summaries[channel], flush=True)
        del aligned, flags, ceilings
        gc.collect()
    return masters, counts, ceiling_counts, summaries, rows


def natural_tile(masters, params, safe_weight, mix=0.7, pedestal=0.035, chroma=0.95):
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
    # Restrained display colour, with negative noise carried until final encoding.
    peak = np.maximum(rgb.max(-1), 1e-8)
    curve = np.arcsinh(15 * peak) / np.arcsinh(15)
    base = encode_srgb(rgb * (curve / ((1 + curve) * peak))[..., None])
    base_y = base @ WEIGHTS
    # Same strong linked curve and L weighting as the earlier enhanced image.
    positive = np.maximum(rgb, 0)
    y = positive @ WEIGHTS
    ll = (
        np.maximum(masters["L"] - params["backgrounds_adu_per_sec"]["L"], 0)
        * params["l_scale"]
        / params["white"]
    )
    ratio = np.clip(ll / np.maximum(y, 1e-5), 0.6, 1.6)
    positive *= (1 + 0.55 * safe_weight * (ratio - 1))[..., None]
    y = positive @ WEIGHTS
    target = (np.arcsinh(55 * y) / np.arcsinh(55)) ** 0.9
    bright = positive * (target / np.maximum(y, 1e-12))[..., None]
    peak = bright.max(-1)
    soft = np.where(
        peak <= 0.85, peak, 0.85 + 0.15 * (1 - np.exp(-np.maximum(peak - 0.85, 0) / 0.15))
    )
    bright *= (soft / np.maximum(peak, 1e-12))[..., None]
    # Do not strengthen highlights with missing channels or unreliable L.
    blend = mix * safe_weight
    desired = blend * (bright @ WEIGHTS) + (1 - blend) * base_y
    # Signed background noise can cancel to nearly zero luminance. A direct
    # desired/base_y ratio would turn those faint colour fluctuations into spikes.
    # Fade only the enhancement in this regime; retain the baseline samples.
    faint_weight = np.clip(base_y / 0.08, 0, 1)
    faint_weight = faint_weight**2 * (3 - 2 * faint_weight)
    scale = 1 + faint_weight * (desired - base_y) / np.maximum(base_y, 0.08)
    image = base * scale[..., None]
    gray = (image @ WEIGHTS)[..., None]
    image = gray + chroma * (image - gray)
    peak = image.max(-1)
    soft = np.where(
        peak <= 0.85, peak, 0.85 + 0.15 * (1 - np.exp(-np.maximum(peak - 0.85, 0) / 0.15))
    )
    image *= np.where(peak > 1e-8, soft / np.maximum(peak, 1e-8), 1)[..., None]
    return np.clip(pedestal + (1 - pedestal) * image, 0, 1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--stage", choices=["all", "render"], default="all")
    args = parser.parse_args()
    out, work = Path(args.output), Path(args.work)
    folder = out / "reports/natural_enhanced"
    folder.mkdir(parents=True, exist_ok=True)
    crop = json.loads((out / "reports/crop_bin1.json").read_text())
    params = json.loads((out / "reports/colour_calibration/render_parameters.json").read_text())
    if args.stage == "all":
        masters, counts, ceilings, summaries, rows = prepare(out, work, crop, folder)
        save_json(
            folder / "integration_audit.json",
            {"channels": summaries, "raw_files_sha256_verified": True},
        )
    else:
        masters = {c: np.load(work / f"safe_{c}.npy", mmap_mode="r") for c in "LRGB"}
        counts = {c: np.load(work / f"valid_{c}.npy", mmap_mode="r") for c in "LRGB"}
        ceilings = {c: np.load(work / f"ceiling_{c}.npy", mmap_mode="r") for c in "LRGB"}
        summaries = json.loads((folder / "integration_audit.json").read_text())["channels"]
        rows = json.loads((out / "reports/metadata_summary.json").read_text())
    frame_counts = {c: summaries[c]["exposures"] for c in "LRGB"}
    h, w = masters["R"].shape
    result = np.lib.format.open_memmap(
        work / "natural_enhanced.npy", mode="w+", dtype="float32", shape=(h, w, 3)
    )
    classification = np.lib.format.open_memmap(
        work / "saturation_class.npy", mode="w+", dtype="uint8", shape=(h, w)
    )
    class_counts = np.zeros(4, dtype=np.int64)
    definite_all = 0
    black = 0
    white = 0
    for y in range(0, h, 128):
        lo, hi = max(0, y - 20), min(h, y + 148)
        cls = classify_availability({c: a[lo:hi] for c, a in counts.items()}, frame_counts)
        unsafe = (cls >= 2) | (counts["L"][lo:hi] == 0)
        # Smooth only the luminance weight, never the captured image structure.
        risk = np.maximum(unsafe, gaussian_filter(unsafe.astype("float32"), 3, truncate=4))
        weight = (1 - risk)[y - lo : min(y + 128, h) - lo]
        tile = natural_tile({c: a[y : y + 128] for c, a in masters.items()}, params, weight)
        result[y : y + 128] = tile
        center = cls[y - lo : min(y + 128, h) - lo]
        classification[y : y + 128] = center
        class_counts += np.bincount(center.ravel(), minlength=4)
        definite_all += int(
            np.logical_and.reduce(
                [ceilings[c][y : y + 128] == frame_counts[c] for c in "RGB"]
            ).sum()
        )
        quantized = np.rint(tile * 255)
        black += int(np.all(quantized == 0, axis=-1).sum())
        white += int(np.any(quantized == 255, axis=-1).sum())
    result.flush()
    classification.flush()
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    stem = f"{args.target}_natural_enhanced_LRGB"
    factor = max(1, int(np.ceil(w / 2800)))
    preview = np.stack([bin_mean(result[..., i], factor) for i in range(3)], axis=-1)
    Image.fromarray(np.uint8(np.rint(preview * 255))).save(
        out / f"previews/{stem}_preview.png", icc_profile=profile
    )
    tif = out / f"final/{stem}_fullres_16bit.tif"
    png = out / f"final/{stem}_fullres.png"
    tifffile.imwrite(tif, np.uint16(np.rint(result * 65535)), photometric="rgb", iccprofile=profile)
    Image.fromarray(np.uint8(np.rint(result * 255))).save(png, icc_profile=profile)
    with tifffile.TiffFile(tif) as tf:
        assert tf.series[0].shape == (h, w, 3) and tf.series[0].dtype == np.uint16
    # Compact native masks retain the per-filter distinction for later work.
    hdus = [fits.PrimaryHDU()]
    for c in "LRGB":
        hdus.append(fits.CompImageHDU(counts[c], name=f"VALID_{c}", compression_type="RICE_1"))
        hdus[-1].header["NFRAMES"] = frame_counts[c]
        hdus[-1].header["COMMENT"] = (
            "Usable: finite and no interpolation contributor at >=98% ceiling."
        )
    fits.HDUList(hdus).writeto(folder / "valid_exposure_counts.fits", overwrite=True)
    cx, cy = json.loads((out / "reports/validation_summary.json").read_text())[
        "core_xy_in_cropped_frame"
    ]
    ys, xs = slice(max(0, cy - 256), min(h, cy + 256)), slice(max(0, cx - 256), min(w, cx + 256))
    core = np.asarray(result[ys, xs])
    classes = np.asarray(classification[ys, xs])
    marked = core.copy()
    for value, colour in [(1, [1, 0.75, 0]), (2, [1, 0.25, 0]), (3, [0.8, 0.1, 1])]:
        marked[classes == value] = colour
    baseline = np.load(work / "observation_baseline_bin1.npy", mmap_mode="r")
    comparison(
        [baseline[ys, xs], core, marked],
        [
            "RGB baseline",
            "Natural Enhanced LRGB",
            "Amber: partial; red: channel missing; purple: all RGB missing",
        ],
        folder / "core_comparison.png",
        width=512,
    )
    old = (
        np.asarray(Image.open(out / f"previews/{args.target}_reference_enhanced_preview.png")) / 255
    )
    base = (
        np.asarray(Image.open(out / f"previews/{args.target}_observation_baseline_RGB_preview.png"))
        / 255
    )
    comparison(
        [old, preview, base],
        ["Original enhanced LRGB", "Natural Enhanced LRGB", "Observation RGB baseline"],
        folder / "three_way_comparison.png",
        width=850,
    )
    report = {
        "dimensions_wh": [w, h],
        "rgb_gains": params["rgb_gains"],
        "luminance_weight": 0.55,
        "enhanced_brightness_mix": 0.7,
        "global_chroma_factor": 0.95,
        "faint_enhancement_transition_display_luminance": 0.08,
        "display_pedestal": 0.035,
        "backgrounds": params["backgrounds_adu_per_sec"],
        "white": params["white"],
        "saturation_rejection_threshold": 0.98,
        "classification_counts": dict(
            zip(
                [
                    "all_RGB_exposures_usable",
                    "some_rejected_but_every_RGB_channel_has_usable_data",
                    "some_RGB_channels_have_no_usable_exposure",
                    "all_RGB_channels_have_no_usable_exposure",
                ],
                map(int, class_counts),
                strict=True,
            )
        ),
        "all_RGB_channels_entire_support_at_detector_ceiling_every_exposure_pixels": definite_all,
        "core_classification_counts": np.bincount(classes.ravel(), minlength=4).tolist(),
        "all_channels_zero_8bit_percent": 100 * black / (h * w),
        "any_channel_255_8bit_percent": 100 * white / (h * w),
        "raw_input_sha256_verified": True,
        "input_inventory": [
            {k: r[k] for k in ["name", "filter", "exposure", "sha256"]} for r in rows
        ],
        "integration": summaries,
        "script_sha256": file_hash(__file__),
        "products": [str(tif), str(png)],
        "handling": (
            "Exclude near-clipped exposure samples per filter. Retain original measured "
            "values only where no usable sample exists; valid_count=0 marks unreliable flux. "
            "No cross-channel synthesis. L boost and enhanced brightness fade around these "
            "flagged pixels. Global gentle chroma scaling is a presentation choice, "
            "not a new colour calibration."
        ),
        "limitations": [
            "Flagged display values are retained measurements, not recovered flux or colour. "
            "Definite detector-clipped values are lower bounds; near-ceiling values may be intact.",
            "Partial-sample stacks may retain PSF differences between exposures.",
            "The 98% threshold is conservative; "
            "definite detector-ceiling loss is separately counted.",
            "Availability also excludes nonfinite registration-edge samples; "
            "partial rejection is not always detector clipping.",
            "No dark/flat correction, external image data, "
            "generated detail, or artificial core restoration.",
        ],
    }
    save_json(folder / "natural_enhanced_report.json", report)
    shutil.copy2(__file__, out / "scripts/render_natural_enhanced.py")
    print(
        json.dumps({k: v for k, v in report.items() if k != "input_inventory"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
