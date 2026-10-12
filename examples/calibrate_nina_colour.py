#!/usr/bin/env python3
"""Reference-star colour calibration of existing NINA LRGB linear masters.

Uses APASS astrometry/photometry and, when response curves are supplied, Gaia DR3
XP spectra. The colour reference is a Sun-like G2V star, not the average field
star. No pixel data is uploaded: only the field centre and public source IDs
are used in catalogue requests. Requires the preview/full stages of
process_nina_lrgb.py and leaves their original display products intact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import astroalign
import numpy as np
import sep
from astropy.io import fits
from astropy.io.votable import parse
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from process_nina_lrgb import colour_parameters, render, save_json
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform


def fetch(path, base, parameters):
    """Cache public catalogue responses; errors remain visible to the caller."""
    path = Path(path)
    if not path.exists():
        url = base + "?" + urlencode(parameters)
        data = urlopen(url, timeout=60).read()
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(data)
        temporary.replace(path)
    return path


def read_apass(path):
    lines = [s for s in Path(path).read_text().splitlines() if s and not s.startswith("#")]
    t = Table.read("\n".join([lines[0], *lines[3:]]), format="ascii.tab")
    for c in t.colnames:
        t[c] = np.array(
            [float(v) if str(v).strip() and not np.ma.is_masked(v) else np.nan for v in t[c]], float
        )
    return t


def detect_stars(green):
    a = np.nan_to_num(green)
    sky = sep.Background(a)
    objects = sep.extract(a - sky.back(), 5, err=sky.globalrms)
    good = (
        (objects["b"] / objects["a"] > 0.65)
        & (objects["a"] < 5)
        & (objects["a"] > 0.6)
        & (objects["flag"] == 0)
    )
    objects = objects[good]
    return objects[np.argsort(objects["flux"])[::-1]]


def match_catalogue(table, objects, header, shape, cache):
    h, w = shape
    scale = 206.264806 * float(header["XPIXSZ"]) / float(header["FOCALLEN"]) * 4 / 3600
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [w / 2, h / 2]
    wcs.wcs.crval = [float(header["RA"]), float(header["DEC"])]
    wcs.wcs.cd = np.diag([-scale, scale])
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    cat_xy = wcs.all_world2pix(np.column_stack([table["RAJ2000"], table["DEJ2000"]]), 0)
    pts = np.column_stack([objects["x"], objects["y"]])
    tree = cKDTree(pts)
    solution = cache / "astrometric_solution.json"
    # A solution is only reused for precisely the same source catalogue/image.
    fingerprint = hashlib.sha256(cat_xy.tobytes() + pts.tobytes()).hexdigest()
    old = json.loads(solution.read_text()) if solution.exists() else {}
    if old.get("fingerprint") == fingerprint:
        parity = old["parity"]
        matrix = np.array(old["matrix"])
    else:
        inside = (
            (cat_xy[:, 0] > -50)
            & (cat_xy[:, 0] < w + 50)
            & (cat_xy[:, 1] > -50)
            & (cat_xy[:, 1] < h + 50)
            & np.isfinite(table["Vmag"])
        )
        indices = np.flatnonzero(inside)
        indices = indices[np.argsort(table["Vmag"][indices])][:100]
        for parity in [-1, 1]:
            try:
                transform, _ = astroalign.find_transform(
                    cat_xy[indices] * [parity, 1], pts[:100], max_control_points=100
                )
                predicted = transform(cat_xy * [parity, 1])
                distances, nearest = tree.query(predicted)
                ok = distances < 2
                if ok.sum() < 20:
                    continue
                refine, _ = ransac(
                    (predicted[ok], pts[nearest[ok]]),
                    AffineTransform,
                    min_samples=4,
                    residual_threshold=1.2,
                    rng=42,
                )
                matrix = refine.params @ transform.params
                break
            except astroalign.MaxIterError:
                continue
        else:
            raise RuntimeError("Could not match the field to APASS; check RA/DEC and image scale.")
    predicted = AffineTransform(matrix=matrix)(cat_xy * [parity, 1])
    distances, _ = tree.query(predicted)
    count = int(np.sum(distances < 1.5))
    if count < 20:
        raise RuntimeError("Too few catalogue matches for a defensible calibration.")
    save_json(
        solution,
        {
            "fingerprint": fingerprint,
            "parity": parity,
            "matrix": matrix.tolist(),
            "tan_wcs": wcs.to_header_string(),
            "matches": count,
            "median_residual_bin4_px": float(np.median(distances[distances < 1.5])),
        },
    )
    print(f"APASS astrometry: {count} matches.", flush=True)
    return wcs, matrix, parity


def aperture_flux(image, x, y, factor=1):
    """Local sky-plane aperture photometry, returning flux, growth and S/N.

    Coordinates/apertures originate in the bin4 frame. At native resolution
    pixel centres and radii are transformed exactly. Processing uses cutouts.
    """
    x, y = factor * np.array([x, y]) + (factor - 1) / 2
    radius, outer = 6 * factor, 15 * factor
    ix, iy = round(x), round(y)
    if (
        min(ix, iy) < outer + 1
        or ix + outer + 1 >= image.shape[1]
        or iy + outer + 1 >= image.shape[0]
    ):
        return np.nan, np.nan, 0, np.nan
    patch = np.asarray(image[iy - outer : iy + outer + 1, ix - outer : ix + outer + 1], dtype=float)
    yy, xx = np.indices(patch.shape, dtype=float)
    xx += ix - outer - x
    yy += iy - outer - y
    rr = xx**2 + yy**2
    ann = (rr >= (10 * factor) ** 2) & (rr <= outer**2) & np.isfinite(patch)
    design = np.stack([np.ones_like(xx), xx / outer, yy / outer], axis=-1)
    keep = ann.copy()
    for _ in range(3):
        coef = np.linalg.lstsq(design[keep], patch[keep], rcond=None)[0]
        residual = patch - design @ coef
        sigma = max(1.4826 * np.median(abs(residual[keep] - np.median(residual[keep]))), 1e-8)
        keep = ann & (abs(residual) < 3 * sigma)
    aperture = rr <= radius**2
    aperture2 = rr <= (8 * factor) ** 2
    flux = residual[aperture].sum()
    flux2 = residual[aperture2].sum()
    noise = sigma * np.sqrt(aperture.sum() * (1 + aperture.sum() / max(keep.sum(), 1)))
    return flux, flux2 / flux if flux > 0 else np.nan, flux / noise, float(coef[0])


def measure(table, coords, objects, masters, native=None):
    pts = np.column_stack([objects["x"], objects["y"]])
    tree = cKDTree(pts)
    distance, index = tree.query(coords)
    nn = tree.query(pts, k=2)[0][:, 1]
    table = table.copy()
    table["x_bin4"], table["y_bin4"] = pts[index].T
    table["match_distance_bin4"] = distance
    table["neighbour_distance_bin4"] = nn[index]
    table["green_peak_bin4"] = objects["peak"][index]
    good = (
        (distance < 1.5)
        & (nn[index] > 18)
        & (objects["peak"][index] < 600)
        & (objects["peak"][index] > 2)
    )
    for c in "RGB":
        image = native[c] if native is not None else masters[c]
        measured = np.array(
            [
                aperture_flux(image, x, y, 4 if native is not None else 1)
                if accept
                else [np.nan, np.nan, 0, np.nan]
                for (x, y), accept in zip(pts[index], good, strict=True)
            ]
        )
        table["flux_" + c], table["growth_" + c], table["snr_" + c], table["sky_" + c] = measured.T
        good &= (
            (measured[:, 0] > 0)
            & (measured[:, 1] > 0.93)
            & (measured[:, 1] < 1.10)
            & (measured[:, 2] > 40)
        )
    table["photometry_accepted"] = good
    return table


def fit_colour(x, y, seed=42):
    """Fit measured instrumental colour against reference colour centered at white."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 8 or np.ptp(x) < 0.25 or not x.min() < 0 < x.max():
        raise ValueError("Need at least eight stars spanning the white-reference colour.")

    def fit(a, b):
        initial = np.polynomial.polynomial.polyfit(a, b, 1)
        return least_squares(
            lambda v: v[0] + v[1] * a - b, initial, loss="soft_l1", f_scale=0.035
        ).x

    coef = fit(x, y)
    residual = y - coef[0] - coef[1] * x
    rng = np.random.default_rng(seed)
    intercepts = []
    for _ in range(300):
        idx = rng.integers(0, len(x), len(x))
        if np.ptp(x[idx]) > 0.15:
            intercepts.append(fit(x[idx], y[idx])[0])
    gains = 10 ** (np.array(intercepts) / 2.5)
    return {
        "gain": float(10 ** (coef[0] / 2.5)),
        "intercept_mag": float(coef[0]),
        "slope": float(coef[1]),
        "star_count": len(x),
        "rms_mag": float(np.sqrt(np.mean(residual**2))),
        "gain_95_percent_bootstrap": np.percentile(gains, [2.5, 97.5]).tolist(),
        "reference_colour_range_mag": [float(x.min()), float(x.max())],
    }


def response_values(record, wavelength):
    wave = np.asarray(record["wavelength"]["value"], float)
    units = record["wavelength"]["units"]
    if units.startswith("angstrom"):
        wave /= 10
    elif units != "nm":
        raise ValueError(f"Unsupported wavelength units: {units}")
    values = np.asarray(record["values"]["value"]) / record["values"]["range"]
    return np.interp(wavelength, wave, values, left=0, right=0)


def spectral_photons(wavelength, energy_flux, response):
    # Photon count is proportional to energy flux * lambda / (h*c).
    # The common hc constant cancels in colour ratios.
    return np.trapezoid(response * energy_flux * wavelength, wavelength, axis=-1)


def load_response(directory, file, name):
    records = json.loads((directory / file).read_text())
    return next(r for r in records if r["name"] == name)


def gaia_calibration(args, header, radius, objects, masters, native, wcs, matrix, parity, cache):
    ra, dec = float(header["RA"]), float(header["DEC"])
    query = (
        "SELECT source_id,ra,dec,pmra,pmdec,phot_g_mean_mag,bp_rp,ruwe,phot_variable_flag "
        "FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),"
        f"CIRCLE('ICRS',{ra},{dec},{radius})) AND has_xp_sampled='True' "
        "AND phot_g_mean_mag<16"
    )
    fetch(
        cache / "gaia_stars.csv",
        "https://gea.esac.esa.int/tap-server/tap/sync",
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
    )
    g = Table.read(cache / "gaia_stars.csv", format="ascii.csv")
    dt = Time(header["DATE-OBS"]).jyear - 2016.0
    decs = np.array(g["dec"]) + np.ma.filled(g["pmdec"], 0) * dt / 3.6e6
    ras = np.array(g["ra"]) + np.ma.filled(g["pmra"], 0) * dt / (
        3.6e6 * np.cos(np.deg2rad(g["dec"]))
    )
    xy = AffineTransform(matrix=matrix)(
        wcs.all_world2pix(np.column_stack([ras, decs]), 0) * [parity, 1]
    )
    g = measure(g, xy, objects, masters, native)
    good = g["photometry_accepted"] & (g["ruwe"] < 1.4) & (g["phot_variable_flag"] != "VARIABLE")
    g = g[good]
    print(f"Gaia: {len(g)} isolated, unsaturated, non-variable candidates.", flush=True)
    directory = Path(args.responses)
    names = [args.red_filter, args.green_filter, args.blue_filter]
    filters = [load_response(directory, "mono_filters/" + args.filter_file, name) for name in names]
    sensor = load_response(directory, "mono_sensors/" + args.sensor_file, args.sensor)
    white = load_response(directory, "wb_refs/" + args.white_file, args.white_reference)
    wavelength = np.arange(380.0, 781.0, 1.0)
    response = np.array(
        [response_values(f, wavelength) * response_values(sensor, wavelength) for f in filters]
    )
    white_flux = spectral_photons(wavelength, response_values(white, wavelength), response)
    spectra = cache / "gaia_spectra"
    spectra.mkdir(exist_ok=True)
    predicted, failures = [], []
    for row in g:
        source = str(row["source_id"])
        try:
            f = fetch(
                spectra / f"{source}.xml",
                "https://gea.esac.esa.int/data-server/data",
                {"RETRIEVAL_TYPE": "XP_SAMPLED", "ID": source, "RELEASE": "Gaia DR3"},
            )
            table = next(iter(parse(f).iter_tables())).to_table()
            flux = np.interp(wavelength, table["wavelength"], table["flux"])
            predicted.append(spectral_photons(wavelength, flux, response))
        except Exception as e:
            failures.append({"source_id": source, "error": str(e)})
            predicted.append([np.nan] * 3)
    predicted = np.array(predicted)
    result = {}
    for c, j in [("R", 0), ("B", 2)]:
        x = -2.5 * np.log10(predicted[:, j] / predicted[:, 1] / (white_flux[j] / white_flux[1]))
        y = -2.5 * np.log10(g["flux_" + c] / g["flux_G"])
        result[c] = fit_colour(x, y)
        g[c + "_reference_mag"] = x
        g[c + "_measured_mag"] = y
    g.write(cache / "gaia_photometry.ecsv", overwrite=True)
    save_json(cache / "gaia_download_failures.json", failures)
    save_json(cache / "spectral_response_models.json", [*filters, sensor, white])
    if (directory / "LICENSE.md").exists():
        shutil.copy2(directory / "LICENSE.md", cache / "response_database_LICENSE.txt")
    curve_info = []
    for record in [*filters, sensor, white]:
        curve_info.append({k: v for k, v in record.items() if k not in {"wavelength", "values"}})
    for c in "RB":
        if result[c]["rms_mag"] > 0.12 or not 0.3 < result[c]["slope"] < 1.7:
            raise RuntimeError(
                f"Unreliable Gaia {c}/G fit; inspect the photometry before rendering."
            )
    return result, curve_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Existing alcor_output directory")
    parser.add_argument("--work", required=True, help="Existing linear-master cache")
    parser.add_argument("--target", default="target")
    parser.add_argument("--method", choices=["apass", "gaia"], default="apass")
    parser.add_argument("--responses", help="Local checkout of siril-spcc-database (data only)")
    parser.add_argument("--filter-file", default="Astrodon_Gen2_RGB.json")
    parser.add_argument("--red-filter", default="Astrodon Red (E series)")
    parser.add_argument("--green-filter", default="Astrodon Green (E series)")
    parser.add_argument("--blue-filter", default="Astrodon Blue (E / I series)")
    parser.add_argument("--sensor-file", default="Sony_IMX.json")
    parser.add_argument("--sensor", default="Sony IMX411/455/461/533/571")
    parser.add_argument("--white-file", default="Star_g2v.json")
    parser.add_argument("--white-reference", default="Star, type G2(v)")
    parser.add_argument("--stage", choices=["calibrate", "render"], default="calibrate")
    args = parser.parse_args()
    args.report_prefix = "reference_"
    out, work = Path(args.output), Path(args.work)
    cache = out / "reports/colour_calibration"
    cache.mkdir(exist_ok=True)
    masters = {c: np.load(work / f"master_{c}_bin4.npy", mmap_mode="r") for c in "LRGB"}
    native = (
        {c: np.load(work / f"master_{c}_bin1.npy", mmap_mode="r") for c in "RGB"}
        if (work / "master_G_bin1.npy").exists()
        else None
    )
    if args.stage == "calibrate":
        rows = json.loads((out / "reports/metadata_summary.json").read_text())
        ref = json.loads((out / "reports/registration_report.json").read_text())[0]["reference"]
        header = fits.getheader(next(r["path"] for r in rows if r["name"] == ref))
        h, w = masters["G"].shape
        radius = (
            np.hypot(h, w)
            * 206.264806
            * float(header["XPIXSZ"])
            / float(header["FOCALLEN"])
            * 4
            / 3600
            * 0.55
        )
        fetch(
            cache / "apass.tsv",
            "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
            {
                "-source": "II/336/apass9",
                "-c": f"{header['RA']} {header['DEC']}",
                "-c.rd": radius,
                "-out.max": 20000,
                "-out.all": "",
            },
        )
        apass = read_apass(cache / "apass.tsv")
        objects = detect_stars(masters["G"])
        wcs, matrix, parity = match_catalogue(apass, objects, header, (h, w), cache)
        xy = AffineTransform(matrix=matrix)(
            wcs.all_world2pix(np.column_stack([apass["RAJ2000"], apass["DEJ2000"]]), 0)
            * [parity, 1]
        )
        apass = measure(apass, xy, objects, masters, native)
        good = (
            apass["photometry_accepted"]
            & (apass["e_B-V"] > 0)
            & (apass["e_B-V"] < 0.07)
            & (apass["B-V"] > 0.1)
            & (apass["B-V"] < 1.5)
        )
        apass["accepted"] = good
        apass.write(cache / "apass_photometry.ecsv", overwrite=True)
        check = {
            c: fit_colour(
                apass["B-V"][good] - 0.653,
                -2.5 * np.log10(apass["flux_" + c][good] / apass["flux_G"][good]),
            )
            for c in "RB"
        }
        selected, curves = check, []
        if args.method == "gaia":
            if not args.responses:
                parser.error(
                    "--responses is required for Gaia calibration; confirm actual equipment first."
                )
            selected, curves = gaia_calibration(
                args, header, radius, objects, masters, native, wcs, matrix, parity, cache
            )
        gain = [selected["R"]["gain"], 1.0, selected["B"]["gain"]]
        y0, y1, x0, x1 = json.loads((out / "reports/crop_bin4.json").read_text())
        cropped = {c: a[y0:y1, x0:x1] for c, a in masters.items()}
        params = colour_parameters(cropped, gain)
        params.update(
            colour_method=(
                "Gaia DR3 XP spectra, measured filter/sensor response model, G2V white"
                if args.method == "gaia"
                else "APASS B-V regression, solar B-V=0.653 white"
            ),
            colour_star_count=selected["R"]["star_count"],
        )
        report = {
            "method": args.method,
            "rgb_gains": gain,
            "fits": selected,
            "apass_crosscheck": check,
            "response_models": curves,
            "photometry_resolution": "native" if native is not None else "4x binned",
            "aperture_native_px": 24,
            "sky_annulus_native_px": [40, 60],
            "limitations": [
                "Raw data lack dark/flat calibration.",
                "Published model curves are not measurements of this individual filter/camera.",
                "Bootstrap ranges measure sampling uncertainty, not all systematic errors.",
                "Broadband colour balance is not a full CIE reconstruction of emission spectra.",
                "Display stretch is nonlinear; saturated measurements remain unrecoverable.",
            ],
        }
        save_json(cache / "calibration.json", report)
        save_json(cache / "render_parameters.json", params)
        args.target += "_reference"
        render(args, cropped, 4, params)
        print(json.dumps(report, indent=2), flush=True)
    else:
        params = json.loads((cache / "render_parameters.json").read_text())
        y0, y1, x0, x1 = json.loads((out / "reports/crop_bin1.json").read_text())
        full = {
            c: np.load(work / f"master_{c}_bin1.npy", mmap_mode="r")[y0:y1, x0:x1] for c in "LRGB"
        }
        args.target += "_reference"
        render(args, full, 1, params)
        del full
        gc.collect()
    shutil.copy2(__file__, out / "scripts/calibrate_nina_colour.py")
    print("Reference-colour products saved:", out, flush=True)


if __name__ == "__main__":
    main()
