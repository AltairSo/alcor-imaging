#!/usr/bin/env python3
"""Validate reference-colour exports and save auditable fit diagnostics."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from astropy.io import fits
from astropy.table import Table
from PIL import Image
from process_nina_lrgb import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="target")
    parser.add_argument("--equipment-source", default="Not supplied")
    args = parser.parse_args()
    out = Path(args.output)
    folder = out / "reports/colour_calibration"
    result = json.loads((folder / "calibration.json").read_text())
    measurements = Table.read(folder / "gaia_photometry.ecsv")
    used = np.isfinite(measurements["R_reference_mag"]) & np.isfinite(
        measurements["B_reference_mag"]
    )
    stars = measurements[used]
    xy = np.column_stack([stars["x_bin4"], stars["y_bin4"]]) * 4 + 1.5
    frames = json.loads((out / "reports/metadata_summary.json").read_text())
    transforms = {
        r["name"]: np.array(r["matrix"])
        for r in json.loads((out / "reports/registration_report.json").read_text())
    }
    saturation_checks = []
    for frame in frames:
        if frame["filter"] not in "RGB":
            continue
        rawxy = np.column_stack([xy, np.ones(len(xy))]) @ np.linalg.inv(transforms[frame["name"]]).T
        maximum = []
        with fits.open(frame["path"], memmap=True) as hdus:
            for x, y, _ in rawxy:
                x, y = round(x), round(y)
                patch = hdus[0].section[y - 12 : y + 13, x - 12 : x + 13]
                maximum.append(float(np.max(patch)))
        saturation_checks.append(
            {
                "frame": frame["name"],
                "largest_calibration_star_raw_peak": max(maximum),
                "saturation_adu": frame["saturation"],
                "stars_above_90_percent": int(
                    np.sum(np.array(maximum) > 0.9 * frame["saturation"])
                ),
            }
        )
    assert not any(r["stars_above_90_percent"] for r in saturation_checks), (
        "A calibration star may be saturated; refit without it."
    )
    verified = []
    y0, y1, x0, x1 = json.loads((out / "reports/crop_bin1.json").read_text())
    expected = (y1 - y0, x1 - x0, 3)
    for profile in ["natural", "enhanced", "inspection"]:
        file = out / f"final/{args.target}_reference_{profile}_fullres_16bit.tif"
        with tifffile.TiffFile(file) as t:
            assert t.series[0].shape == expected and t.series[0].dtype == np.uint16
        verified.append({"file": file.name, "shape": expected, "dtype": "uint16"})
    png = out / f"final/{args.target}_reference_natural_fullres.png"
    with Image.open(png) as image:
        assert image.size == expected[1::-1] and image.mode == "RGB"
    verified.append({"file": png.name, "shape": expected, "dtype": "uint8 RGB"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, c, colour in zip(axes, "RB", ["#c24a41", "#3b78b7"], strict=True):
        x = np.asarray(stars[c + "_reference_mag"])
        y = np.asarray(stars[c + "_measured_mag"])
        f = result["fits"][c]
        ax.scatter(x, y, color=colour, s=28)
        line = np.linspace(x.min(), x.max(), 100)
        ax.plot(line, f["intercept_mag"] + f["slope"] * line, color=colour)
        ax.axvline(0, color="#888888", linestyle=":", linewidth=1)
        ax.set(
            xlabel=f"Gaia spectral {c}/G colour relative to G2V (mag)",
            ylabel="Measured instrumental colour (mag)",
            title=f"{c}: gain {f['gain']:.3f}; {len(x)} stars; RMS {f['rms_mag']:.3f} mag",
        )
        ax.grid(alpha=0.15)
    fig.suptitle(f"{args.target} colour calibration — native apertures against Gaia DR3 spectra")
    fig.savefig(folder / "colour_fit.png", dpi=170)
    plt.close(fig)
    root = Path(__file__).resolve().parents[1]
    provenance = {
        "revision_at_validation": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "calibration_script_sha256": hashlib.sha256(
            (out / "scripts/calibrate_nina_colour.py").read_bytes()
        ).hexdigest(),
    }
    save_json(
        folder / "validation.json",
        {
            "products": verified,
            "raw_saturation_checks": saturation_checks,
            "provenance": provenance,
        },
    )
    gain = result["rgb_gains"]
    text = (
        f"{args.target} reference-star colour revision\n\n"
        f"Method: {result['method']}; {len(stars)} Gaia DR3 XP reference stars.\n"
        f"Aperture measurements: {result['photometry_resolution']} resolution.\n"
        f"RGB gains: {gain}. White reference: G2V.\n"
        "Equipment models: "
        + ", ".join(m["name"] for m in result["response_models"])
        + "\n"
        + f"Equipment source: {args.equipment_source}\n"
        "Response curves: https://gitlab.com/free-astro/siril-spcc-database\n"
        "Reference data: ESA/Gaia/DPAC DR3 XP spectra; APASS DR9 cross-check.\n"
        f"R fit RMS: {result['fits']['R']['rms_mag']:.4f} mag; "
        f"B fit RMS: {result['fits']['B']['rms_mag']:.4f} mag.\n"
        f"R gain bootstrap 95%: {result['fits']['R']['gain_95_percent_bootstrap']}.\n"
        f"B gain bootstrap 95%: {result['fits']['B']['gain_95_percent_bootstrap']}.\n"
        f"APASS cross-check gains R/B: {result['apass_crosscheck']['R']['gain']:.4f}, "
        f"{result['apass_crosscheck']['B']['gain']:.4f}.\n"
        "Calibration-star cores checked in every RGB exposure: "
        "none above 90% detector saturation.\n"
        f"Verified output: {expected[1]} x {expected[0]} native pixels; "
        "three RGB uint16 TIFFs and one RGB PNG.\n"
        "Original scientific masters and earlier display images remain unchanged.\n\nLimitations:\n"
        + "\n".join("- " + s for s in result["limitations"])
        + "\n"
    )
    (folder / "colour_report.txt").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
