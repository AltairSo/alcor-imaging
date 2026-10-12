#!/usr/bin/env python3
"""Validate NINA workflow products and save real-data inspection crops/report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from astropy.io import fits
from PIL import Image, ImageDraw
from process_nina_lrgb import bin_mean, save_json
from scipy.ndimage import gaussian_filter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--target", required=True)
    args = p.parse_args()
    out, work = Path(args.output), Path(args.work)
    rows = json.loads((out / "reports/metadata_summary.json").read_text())
    registrations = json.loads((out / "reports/registration_report.json").read_text())
    params = json.loads((out / "reports/render_parameters_bin1.json").read_text())
    y0, y1, x0, x1 = json.loads((out / "reports/crop_bin1.json").read_text())
    shape = (y1 - y0, x1 - x0)
    products = []
    # Validate this target's original products; later calibrated/baseline exports
    # may coexist in these directories when rerunning the notebook.
    linear_paths = [out / f"linear/{args.target}_master_{c}_uncalibrated.fits" for c in "LRGB"]
    linear_paths.append(out / f"linear/{args.target}_linear_RGB.fits")
    for path in linear_paths:
        with fits.open(path, memmap=True) as hdul:
            a = hdul[0].data
            assert a.shape[-2:] == shape, path
            assert hdul[0].header["BITPIX"] == -32, path
            for row in a.reshape(-1, a.shape[-1]):
                if not np.isfinite(row).all():
                    raise ValueError(f"Nonfinite output: {path}")
            products.append({"file": path.name, "shape": a.shape, "dtype": str(a.dtype)})
    for profile in ["natural", "enhanced", "inspection"]:
        path = out / f"final/{args.target}_{profile}_fullres_16bit.tif"
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            assert page.dtype == np.dtype("uint16") and page.shape == (*shape, 3)
            products.append({"file": path.name, "shape": page.shape, "dtype": "uint16"})
    with Image.open(out / f"final/{args.target}_natural_fullres.png") as im:
        assert im.size == shape[::-1]
        products.append(
            {"file": f"{args.target}_natural_fullres.png", "shape": shape, "dtype": "uint8 RGB"}
        )
    linear = {
        c: np.load(work / f"master_{c}_bin1.npy", mmap_mode="r")[y0:y1, x0:x1] for c in "LRGB"
    }
    display = np.load(work / "natural_bin1.npy", mmap_mode="r")
    core_small = gaussian_filter(bin_mean(linear["L"], 16), 3)
    cy, cx = np.unravel_index(np.argmax(core_small), core_small.shape)
    cy, cx = int(cy * 16 + 8), int(cx * 16 + 8)
    quality = out / "reports/quality"
    Image.fromarray(
        np.uint8(255 * display[max(0, cy - 256) : cy + 256, max(0, cx - 256) : cx + 256])
    ).save(quality / "core_native_100pct.png")
    # Inspect four corners at native sampling and label their positions.
    sheet = Image.new("RGB", (640, 680), "#111111")
    for i, (yy, xx) in enumerate(
        [(0, 0), (0, shape[1] - 320), (shape[0] - 320, 0), (shape[0] - 320, shape[1] - 320)]
    ):
        patch = np.uint8(255 * display[yy : yy + 320, xx : xx + 320])
        sheet.paste(Image.fromarray(patch), ((i % 2) * 320, (i // 2) * 340 + 20))
        ImageDraw.Draw(sheet).text(
            ((i % 2) * 320 + 8, (i // 2) * 340 + 3), f"Native corner x={xx}, y={yy}", fill="white"
        )
    sheet.save(quality / "corners_native_100pct.png")
    # Choose an unsaturated off-nebula star from the reference catalogue.
    catalogs = json.loads((work / "catalogs.json").read_text())
    reference = registrations[0]["reference"]
    points = np.array(catalogs[reference]) - np.array([x0, y0])
    good = points[
        (points[:, 0] > 160)
        & (points[:, 0] < shape[1] - 160)
        & (points[:, 1] > 160)
        & (points[:, 1] < shape[0] - 160)
    ]
    distances = np.linalg.norm(good - np.array([cx, cy]), axis=1)
    sx, sy = good[np.argmin(abs(distances - 2200))].astype(int)
    Image.fromarray(np.uint8(255 * display[sy - 160 : sy + 160, sx - 160 : sx + 160])).save(
        quality / "star_native_100pct.png"
    )
    # A native core patch from every raw frame quantifies real sensor clipping.
    saturation = []
    for r in rows:
        a = fits.getdata(r["path"])
        patch = a[max(0, cy + y0 - 200) : cy + y0 + 200, max(0, cx + x0 - 200) : cx + x0 + 200]
        saturation.append(
            {
                "name": r["name"],
                "core_near_saturation_percent": float(
                    100 * np.mean(patch >= 0.98 * r["saturation"])
                ),
            }
        )
        del a
    comparison = []
    for c in "LRGB":
        selected = [r for r in rows if r["filter"] == c]
        patches = [
            np.load(work / (Path(r["name"]).stem + "_aligned_bin4.npy"), mmap_mode="r")[
                100:350, 100:350
            ]
            / r["exposure"]
            for r in selected
        ]
        cube = np.array(patches)

        def noise(a):
            d = np.diff(a, axis=1)
            return float(1.4826 * np.nanmedian(abs(d - np.nanmedian(d))) / np.sqrt(2))

        comparison.append(
            {
                "filter": c,
                "mean_noise_adu_per_second": noise(np.nanmean(cube, axis=0)),
                "median_noise_adu_per_second": noise(np.nanmedian(cube, axis=0)),
            }
        )
    sample = display[::8, ::8]
    summary = {
        "dimensions_wh": shape[::-1],
        "total_seconds": sum(r["exposure"] for r in rows),
        "frames": len(rows),
        "rejected": [],
        "crop_y0_y1_x0_x1": [y0, y1, x0, x1],
        "calibration": "No matching darks, bias or flats were supplied/applied.",
        "core_xy_in_cropped_frame": [cx, cy],
        "core_saturation": saturation,
        "stack_comparison": comparison,
        "verified_products": products,
        "display_all_channels_black_percent": float(100 * np.mean(sample.max(-1) == 0)),
        "display_any_channel_at_one_percent": float(100 * np.mean(sample.max(-1) >= 1)),
        "registration_median_residual_range_px": [
            min(
                r["median_residual_native_px"] for r in registrations if r["name"] != r["reference"]
            ),
            max(r["median_residual_native_px"] for r in registrations),
        ],
        "limitations": [
            f"Total integration is {sum(r['exposure'] for r in rows):g} seconds.",
            "Empirical colour balance, not photometric calibration.",
            "Uncalibrated fixed-pattern noise/hot pixels may remain.",
            "Pixels clipped in all contributing exposures cannot be reconstructed.",
        ],
    }
    save_json(out / "reports/validation_summary.json", summary)
    report = [
        f"{args.target} — Alcor Imaging processing report",
        f"{len(rows)} frames, {sum(r['exposure'] for r in rows):.0f} seconds total; all retained.",
        f"Native output: {shape[1]} x {shape[0]} pixels. Crop: {[y0, y1, x0, x1]}.",
        summary["calibration"],
        "Registration: stellar similarity transforms into the selected L reference; "
        "one resampling per frame.",
        "Stack: see processing_parameters.json; linear masters are normalized to ADU/s.",
        "Mean/median noise comparison is in validation_summary.json. "
        "Median favors outlier resistance.",
        "FWHM values are native-scaled SEP second-moment estimates "
        "from 2x binned detection, not PSF fits.",
        "Colour: R/G/B filters map to RGB. Ensemble-star empirical gains: "
        + str(params["rgb_gains"]),
        "Display: 55% L luminance contribution, scalar background offsets, "
        "linked asinh (natural 25; enhanced 55; inspection 15), gamma 0.9; "
        "highlight compression preserves ratios.",
        "No sharpening, gradient fitting, generated detail or deconvolution was applied.",
        "Linear RGB FITS has no L injection; L contributes only to the display images.",
        "All FITS masters opened with finite float32 data; all TIFFs verified as RGB uint16.",
        "",
        "Measured frame quality (filter, estimated FWHM px, ellipticity, near-saturated %):",
    ]
    report += [
        f"{r['filter']}: {r['fwhm_native_estimate']:.2f}, {r['ellipticity']:.3f}, "
        f"{r['saturated_percent']:.5f} — {r['name']}"
        for r in rows
    ]
    report += ["", "Limits:"] + summary["limitations"]
    (out / "reports/processing_report.txt").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
