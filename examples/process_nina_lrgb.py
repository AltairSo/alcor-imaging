#!/usr/bin/env python3
"""Memory-bounded, reproducible mono LRGB workflow for NINA FITS and Colab.

Run --stage inspect, --stage preview, then --stage full. The output directory
contains the inventory, transforms, linear masters, real-data previews and TIFFs.
No synthetic detail, automatic frame deletion, or source-file writes are used.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
from pathlib import Path

import astroalign
import numpy as np
import sep
import tifffile
from astropy.io import fits
from PIL import Image, ImageDraw
from skimage.transform import SimilarityTransform

from alcor_imaging import (
    StackConfig,
    hdr_combine_with_mask,
    luminance,
    stack,
    warp_affine,
)

FILTERS = {
    "L": "L",
    "LUM": "L",
    "LUMINANCE": "L",
    "R": "R",
    "RED": "R",
    "G": "G",
    "GREEN": "G",
    "B": "B",
    "BLUE": "B",
}


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str))


def group_filter(header, filename):
    raw = str(header.get("FILTER", "")).strip().upper()
    detected = FILTERS.get(raw)
    match = re.search(r"_(L|R|G|B)_", filename, re.I)
    fallback = match.group(1).upper() if match else None
    if raw and detected is None:
        raise ValueError(f"Unsupported FILTER={raw!r}: {filename}")
    if detected and fallback and detected != fallback:
        raise ValueError(f"Header/filename filter disagreement: {filename}")
    if not (detected or fallback):
        raise ValueError(f"Cannot determine filter: {filename}")
    return detected or fallback


def bin_mean(a, factor):
    if factor == 1:
        return np.asarray(a, dtype=np.float32)
    h, w = a.shape
    h, w = h // factor * factor, w // factor * factor
    return a[:h, :w].reshape(h // factor, factor, w // factor, factor).mean((1, 3))


def change_pixel_scale(matrix, factor):
    # Binned pixel centers: native = factor * binned + (factor-1)/2.
    s = np.array([[factor, 0, (factor - 1) / 2], [0, factor, (factor - 1) / 2], [0, 0, 1.0]])
    return np.linalg.inv(s) @ matrix @ s


def catalog(a, factor=2):
    small = np.ascontiguousarray(bin_mean(a, factor), dtype=np.float32)
    bg = sep.Background(small, bw=64, bh=64, fw=3, fh=3)
    objects = sep.extract(small - bg.back(), 6.0, err=bg.globalrms, minarea=6)
    good = objects[
        (objects["flag"] == 0)
        & (objects["a"] > 0.65)
        & (objects["b"] / np.maximum(objects["a"], 0.01) > 0.45)
        & (objects["npix"] < 350)
        & (objects["x"] > 12)
        & (objects["y"] > 12)
        & (objects["x"] < small.shape[1] - 12)
        & (objects["y"] < small.shape[0] - 12)
    ]
    good = good[np.argsort(good["flux"])[::-1]][:250]
    points = np.column_stack((good["x"], good["y"])) * factor + (factor - 1) / 2
    fwhm = 2.3548 * np.sqrt(good["a"] * good["b"]) * factor
    return points, {
        "background": float(bg.globalback),
        "noise_binned": float(bg.globalrms),
        "detection_bin": factor,
        "star_count": len(good),
        "fwhm_native_estimate": float(np.median(fwhm)) if len(good) else None,
        "ellipticity": float(np.median(1 - good["b"] / good["a"])) if len(good) else None,
    }


def mono_display(a):
    lo, hi = np.nanpercentile(a, [5, 99.95])
    return np.arcsinh(25 * np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)) / np.arcsinh(25)


def inspect(args):
    out, work = Path(args.output), Path(args.work)
    for d in ["reports/quality", "previews", "linear", "final", "scripts"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p
        for p in Path(args.input).rglob("*")
        if p.suffix.lower() in {".fits", ".fit", ".fts"}
        and out.resolve() not in p.resolve().parents
    )
    rows, points, thumbs = [], {}, []
    for p in files:
        h = fits.getheader(p)
        typ = str(h.get("IMAGETYP", h.get("FRAME", "LIGHT"))).upper()
        if "LIGHT" not in typ:
            print("Calibration/non-light file (not automatically applied):", p.name, flush=True)
            continue
        filt = group_filter(h, p.name)
        a = fits.getdata(p).astype(np.float32)
        if a.ndim != 2 or not np.isfinite(a).all() or np.ptp(a) == 0:
            raise ValueError(f"Invalid, nonfinite or constant mono image: {p}")
        exposure = float(h.get("EXPTIME", h.get("EXPOSURE", 0)))
        if exposure <= 0:
            raise ValueError(f"Missing/invalid exposure: {p}")
        sat = args.saturation or h.get("SATURATE")
        if sat is None and h.get("BITPIX") == 16 and h.get("BZERO") == 32768:
            sat = 65535.0
        if sat is None:
            raise ValueError("Set --saturation to the detector ceiling for non-uint16 input.")
        xy, quality = catalog(a)
        # Remove saturated stars from registration and empirical colour calibration.
        valid = [
            np.max(a[max(0, int(y) - 6) : int(y) + 7, max(0, int(x) - 6) : int(x) + 7]) < 0.9 * sat
            for x, y in xy
        ]
        xy = xy[valid]
        quality["registration_stars"] = len(xy)
        with p.open("rb") as f:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
            digest = hasher.hexdigest()
        row = dict(
            path=str(p),
            name=p.name,
            filter=filt,
            exposure=exposure,
            width=a.shape[1],
            height=a.shape[0],
            gain=h.get("GAIN"),
            offset=h.get("OFFSET"),
            temperature=h.get("CCD-TEMP"),
            xbin=h.get("XBINNING"),
            ybin=h.get("YBINNING"),
            camera=h.get("INSTRUME"),
            telescope=h.get("TELESCOP"),
            object=h.get("OBJECT"),
            date_obs=h.get("DATE-OBS"),
            minimum=float(a.min()),
            maximum=float(a.max()),
            saturation=float(sat),
            saturated_percent=float(100 * np.count_nonzero(a >= 0.98 * sat) / a.size),
            sha256=digest,
            status="retained",
            **quality,
        )
        points[p.name] = xy.tolist()
        preview = bin_mean(a, 4)
        np.save(work / (p.stem + "_bin4.npy"), preview)
        im = Image.fromarray(np.uint8(255 * mono_display(preview))).convert("RGB")
        im.thumbnail((570, 380))
        canvas = Image.new("RGB", (590, 425), "#111111")
        canvas.paste(im, ((590 - im.width) // 2, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f"{filt} | {p.stem}", fill="white")
        draw.text(
            (10, 403),
            f"FWHM~{quality['fwhm_native_estimate']:.1f}px  sat={row['saturated_percent']:.4f}%",
            fill="white",
        )
        thumbs.append(canvas)
        rows.append(row)
        print(
            f"{filt}: {p.name}; stars={len(xy)}, "
            f"FWHM~{row['fwhm_native_estimate']:.2f}px, "
            f"saturated={row['saturated_percent']:.5f}%",
            flush=True,
        )
        del a, preview
        gc.collect()
    if set(r["filter"] for r in rows) != set("LRGB"):
        raise ValueError("All four L/R/G/B filters are required.")
    for key in ["width", "height", "gain", "offset", "xbin", "ybin", "object"]:
        values = {r[key] for r in rows if r[key] is not None}
        if len(values) > 1:
            raise ValueError(f"Inconsistent {key}: {values}")
    sheet = Image.new("RGB", (590 * 3, 425 * ((len(thumbs) + 2) // 3)), "#111111")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % 3) * 590, (i // 3) * 425))
    sheet.save(out / "reports/quality/raw_contact_sheet.jpg", quality=92)
    with (out / "reports/input_inventory.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(out / "reports/metadata_summary.json", rows)
    save_json(work / "catalogs.json", points)
    summary = {
        c: {
            "frames": sum(r["filter"] == c for r in rows),
            "seconds": sum(r["exposure"] for r in rows if r["filter"] == c),
        }
        for c in "LRGB"
    }
    save_json(out / "reports/dataset_summary.json", summary)
    print(summary, flush=True)
    return rows, points


def register(args, rows, points):
    candidates = [r for r in rows if r["filter"] == "L"]
    ref = max(
        candidates, key=lambda r: r["registration_stars"] / max(r["fwhm_native_estimate"], 1) ** 2
    )
    target = np.array(points[ref["name"]])
    records = []
    for row in rows:
        xy = np.array(points[row["name"]])
        if row["name"] == ref["name"]:
            t = SimilarityTransform()
            residuals = np.zeros(len(xy))
            matches = len(xy)
        else:
            t, (src, dst) = astroalign.find_transform(
                xy[:150], target[:150], max_control_points=150
            )
            residuals = np.linalg.norm(t(src) - dst, axis=1)
            matches = len(src)
            if matches < 8 or np.median(residuals) > 2.0 or abs(t.scale - 1) > 0.01:
                raise RuntimeError(
                    f"Unreliable alignment for {row['name']}: inspect before continuing"
                )
        rec = {
            "name": row["name"],
            "filter": row["filter"],
            "reference": ref["name"],
            "matrix": t.params.tolist(),
            "matched_stars": matches,
            "median_residual_native_px": float(np.median(residuals)),
            "p95_residual_native_px": float(np.percentile(residuals, 95)),
            "translation_xy": t.translation.tolist(),
            "rotation_degrees": float(np.degrees(t.rotation)),
        }
        records.append(rec)
        print(
            f"Align {row['filter']} {row['name']}: {matches} matches, "
            f"residual={np.median(residuals):.3f}px",
            flush=True,
        )
    save_json(Path(args.output) / "reports/registration_report.json", records)
    return records


def integrate(args, rows, records, factor):
    """Resample each original once directly into the common L reference grid."""
    out, work = Path(args.output), Path(args.work)
    matrices = {r["name"]: np.array(r["matrix"]) for r in records}
    result = {}
    for channel in "LRGB":
        selected = [r for r in rows if r["filter"] == channel]
        arrays = []
        for row in selected:
            p = Path(row["path"])
            cache = work / f"{p.stem}_aligned_bin{factor}.npy"
            # Deliberately recompute: never reuse stale registration/calibration caches.
            if factor == 4:
                a = np.load(work / (p.stem + "_bin4.npy"))
            else:
                a = bin_mean(fits.getdata(p).astype(np.float32), factor)
            if args.calibration:
                raise ValueError(
                    "Use calibrated LIGHT inputs; automatic master matching is not supported."
                )
            aligned = warp_affine(
                a,
                change_pixel_scale(matrices[row["name"]], factor),
                a.shape,
                backend="cpu",
                order=1,
            )
            np.save(cache, aligned)
            del aligned, a
            arrays.append(np.load(cache, mmap_mode="r"))
            print(f"Resample bin{factor}: {p.name}", flush=True)
        times = [r["exposure"] for r in selected]
        mixed = len(set(times)) > 1
        destination = work / f"master_{channel}_bin{factor}.npy"
        master = np.lib.format.open_memmap(
            destination, mode="w+", dtype="float32", shape=arrays[0].shape
        )
        backgrounds = [float(np.nanpercentile(a[::8, ::8], 20)) for a in arrays]
        for y in range(0, master.shape[0], 256):
            tile = [a[y : y + 256] for a in arrays]
            if mixed:
                # Include a halo for saturation-mask dilation at tile boundaries.
                lo, hi = max(0, y - 2), min(master.shape[0], y + 258)
                combined, _lost = hdr_combine_with_mask(
                    [a[lo:hi] for a in arrays],
                    times,
                    saturation_levels=[r["saturation"] for r in selected],
                    background_offsets=backgrounds,
                )
                master[y : y + 256] = combined[y - lo : min(y + 256, master.shape[0]) - lo]
            else:
                master[y : y + 256] = (
                    stack(tile, StackConfig(method=args.stack, tile_size=512)) / times[0]
                )
        master.flush()
        result[channel] = np.load(destination, mmap_mode="r")
        print(
            f"Master {channel}: {len(selected)} frames; "
            f"{'exposure-aware HDR' if mixed else args.stack}",
            flush=True,
        )
        del arrays, master
        gc.collect()
    # A strict rectangular intersection avoids nonfinite corners without filling data.
    valid = np.logical_and.reduce([np.isfinite(a) for a in result.values()])
    yy = np.flatnonzero(valid.any(axis=1))
    xx = np.flatnonzero(valid.any(axis=0))
    y0, y1, x0, x1 = int(yy.min()), int(yy.max() + 1), int(xx.min()), int(xx.max() + 1)
    while not valid[y0:y1, x0:x1].all():
        bad = [
            np.count_nonzero(~valid[y0, x0:x1]),
            np.count_nonzero(~valid[y1 - 1, x0:x1]),
            np.count_nonzero(~valid[y0:y1, x0]),
            np.count_nonzero(~valid[y0:y1, x1 - 1]),
        ]
        edge = int(np.argmax(bad))
        if edge == 0:
            y0 += 1
        elif edge == 1:
            y1 -= 1
        elif edge == 2:
            x0 += 1
        else:
            x1 -= 1
        if y1 - y0 < 32 or x1 - x0 < 32:
            raise RuntimeError("No useful rectangular common overlap.")
    crop = [y0, y1, x0, x1]
    save_json(out / f"reports/crop_bin{factor}.json", crop)
    return {c: a[y0:y1, x0:x1] for c, a in result.items()}, crop


def colour_parameters(masters, gains=None):
    small = {
        c: np.asarray(a[:: max(1, a.shape[1] // 1800), :: max(1, a.shape[1] // 1800)])
        for c, a in masters.items()
    }
    backgrounds = {c: float(np.percentile(a, 10)) for c, a in small.items()}
    stars, _ = catalog(np.asarray(small["G"], dtype=np.float32), 1)
    fluxes = []
    for x, y in stars:
        x, y = int(x), int(y)
        if x < 10 or y < 10 or x + 11 >= small["G"].shape[1] or y + 11 >= small["G"].shape[0]:
            continue
        vals = []
        for c in "RGBL":
            p = small[c][y - 8 : y + 9, x - 8 : x + 9]
            sky = np.median(np.concatenate([p[0], p[-1], p[:, 0], p[:, -1]]))
            vals.append(float(np.sum(p[5:12, 5:12] - sky)))
        if min(vals) > 0:
            fluxes.append(vals)
    if len(fluxes) < 8 and gains is None:
        raise ValueError("Too few stars for empirical colour balance; supply --gains R G B.")
    f = np.array(fluxes)
    gain = (
        np.array(gains)
        if gains
        else np.array([np.median(f[:, 1] / f[:, 0]), 1, np.median(f[:, 1] / f[:, 2])])
    )
    if not np.isfinite(gain).all() or np.any(gain <= 0):
        raise ValueError("Invalid channel gains.")
    signal = {c: np.maximum(a - backgrounds[c], 0) for c, a in small.items()}
    y = sum(
        signal[c] * g * k for c, g, k in zip("RGB", gain, [0.2126, 0.7152, 0.0722], strict=True)
    )
    mask = (y > np.percentile(y, 85)) & (y < np.percentile(y, 99)) & (signal["L"] > 0)
    lscale = float(np.median(y[mask] / signal["L"][mask]))
    white = float(np.percentile(y, 99.98))
    return {
        "backgrounds_adu_per_sec": backgrounds,
        "rgb_gains": gain.tolist(),
        "colour_method": "empirical ensemble-star neutral balance; not photometric calibration",
        "colour_star_count": len(fluxes),
        "l_scale": lscale,
        "white": white,
    }


def render_tile(masters, params, profile="natural"):
    bg = params["backgrounds_adu_per_sec"]
    rgb = np.stack(
        [
            np.maximum(masters[c] - bg[c], 0) * g
            for c, g in zip("RGB", params["rgb_gains"], strict=True)
        ],
        axis=-1,
    )
    y = luminance(rgb)
    ll = np.maximum(masters["L"] - bg["L"], 0) * params["l_scale"]
    weight = 0 if profile == "inspection" else 0.55
    ratio = np.clip(ll / np.maximum(y, params["white"] * 1e-5), 0.6, 1.6)
    rgb *= ((1 - weight) + weight * ratio)[..., None]
    # Equal faint/highlight strength gives a monotonic linked asinh curve.
    # The white point measured from the preview is shared by every output tile.
    strength = {"natural": 25.0, "enhanced": 55.0, "inspection": 15.0}[profile]
    white = params["white"]
    light = luminance(rgb) / white
    target = (np.arcsinh(strength * light) / np.arcsinh(strength)) ** 0.9
    image = rgb / white * (target / np.maximum(light, 1e-12))[..., None]
    peak = image.max(-1)
    compressed = np.where(
        peak <= 0.85, peak, 0.85 + 0.15 * (1 - np.exp(-np.maximum(peak - 0.85, 0) / 0.15))
    )
    image *= (compressed / np.maximum(peak, 1e-12))[..., None]
    return np.clip(image, 0, 1).astype(np.float32)


def render(args, masters, factor, params=None):
    out = Path(args.output)
    params = params or colour_parameters(masters, args.gains)
    save_json(out / f"reports/render_parameters_bin{factor}.json", params)
    h, w = masters["L"].shape
    for profile in ["natural", "enhanced", "inspection"]:
        output = np.lib.format.open_memmap(
            Path(args.work) / f"{profile}_bin{factor}.npy",
            mode="w+",
            dtype="float32",
            shape=(h, w, 3),
        )
        for y in range(0, h, 256):
            output[y : y + 256] = render_tile(
                {c: a[y : y + 256] for c, a in masters.items()}, params, profile
            )
        output.flush()
        preview_bin = max(1, int(np.ceil(w / 2800)))
        sampled = np.stack([bin_mean(output[..., i], preview_bin) for i in range(3)], axis=-1)
        im = Image.fromarray(np.uint8(np.round(255 * sampled)))
        im.thumbnail((2800, 2800), Image.Resampling.LANCZOS)
        im.save(out / f"previews/{args.target}_{profile}_preview.png")
        if factor == 1:
            display = np.uint16(np.round(output * 65535))
            tifffile.imwrite(
                out / f"final/{args.target}_{profile}_fullres_16bit.tif",
                display,
                photometric="rgb",
                metadata={"axes": "YXS", "processing": params},
            )
            del display
            if profile == "natural":
                Image.fromarray(np.uint8(np.round(output * 255))).save(
                    out / f"final/{args.target}_natural_fullres.png"
                )
        print(f"Exported {profile} bin{factor}: {w} x {h}", flush=True)
        del output
        gc.collect()
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work", default="/tmp/alcor_lrgb_work")
    parser.add_argument("--target", default="target")
    parser.add_argument(
        "--stage", choices=["inspect", "preview", "full", "render"], default="preview"
    )
    parser.add_argument("--stack", choices=["median", "mean", "sigma_clip_mean"], default="median")
    parser.add_argument("--gains", nargs=3, type=float)
    parser.add_argument("--saturation", type=float)
    parser.add_argument("--calibration", help="Reserved: supply pre-calibrated LIGHT files instead")
    args = parser.parse_args()
    out, work = Path(args.output), Path(args.work)
    if out.resolve() == Path(args.input).resolve():
        parser.error("Output must be a separate directory.")
    if args.stage in {"inspect", "preview"}:
        rows, points = inspect(args)
        records = register(args, rows, points)
    else:
        rows = json.loads((out / "reports/metadata_summary.json").read_text())
        records = json.loads((out / "reports/registration_report.json").read_text())
        # Refuse accidentally processing a different folder from a previous run.
        if any(
            not Path(r["path"]).resolve().is_relative_to(Path(args.input).resolve()) for r in rows
        ):
            parser.error("Inventory belongs to a different input; rerun preview.")
    save_json(out / "reports/processing_parameters.json", vars(args))
    shutil.copy2(__file__, out / "scripts/process_nina_lrgb.py")
    versions = {
        p: importlib.metadata.version(p)
        for p in [
            "alcor-imaging",
            "numpy",
            "scipy",
            "astropy",
            "astroalign",
            "sep",
            "scikit-image",
            "tifffile",
            "Pillow",
        ]
    }
    save_json(
        out / "scripts/environment.json",
        {
            "python": sys.version,
            "packages": versions,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    if args.stage == "inspect":
        return
    factor = 1 if args.stage in {"full", "render"} else 4
    if args.stage == "render":
        y0, y1, x0, x1 = json.loads((out / "reports/crop_bin1.json").read_text())
        masters = {
            c: np.load(work / f"master_{c}_bin1.npy", mmap_mode="r")[y0:y1, x0:x1] for c in "LRGB"
        }
    else:
        masters, _crop = integrate(args, rows, records, factor)
    params = None
    if factor == 1:
        params = json.loads((out / "reports/render_parameters_bin4.json").read_text())
    if args.stage == "full":
        # Keep the preview's chosen colour/curve parameters for the final image.
        params = json.loads((out / "reports/render_parameters_bin4.json").read_text())
        for c, a in masters.items():
            h = fits.Header(
                {
                    "FILTER": c,
                    "BUNIT": "ADU/s",
                    "CALSTAT": "UNCALIB",
                    "EXPTIME": sum(r["exposure"] for r in rows if r["filter"] == c),
                    "NCOMBINE": sum(r["filter"] == c for r in rows),
                    "HISTORY": "Registered linear stack; no display stretch. See reports.",
                }
            )
            fits.writeto(
                out / f"linear/{args.target}_master_{c}_uncalibrated.fits",
                np.asarray(a),
                h,
                overwrite=True,
            )
        # Stored as R/G/B planes; these linear channels have no injected luminance.
        cube = np.lib.format.open_memmap(
            work / "linear_RGB.npy", mode="w+", dtype="float32", shape=(3, *masters["R"].shape)
        )
        for i, c in enumerate("RGB"):
            cube[i] = masters[c]
        cube.flush()
        fits.writeto(
            out / f"linear/{args.target}_linear_RGB.fits",
            cube,
            fits.Header({"BUNIT": "ADU/s", "CTYPE3": "RGB", "CALSTAT": "UNCALIB"}),
            overwrite=True,
        )
        del cube
    render(args, masters, factor, params)
    print("SUCCESS:", out, flush=True)


if __name__ == "__main__":
    main()
