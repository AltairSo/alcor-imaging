"""Dropbox shared-link intake; runs in Colab, never needs a Dropbox secret.

Keep source manifests private: shared URLs are access-bearing links.
The Alcor core remains independent of storage providers.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests


def safe_relative(value):
    p = PurePosixPath(value)
    if p.is_absolute() or not p.parts or any(x in {"..", "."} for x in p.parts) or "\\" in value:
        raise ValueError("Unsafe archive path")
    return Path(*p.parts)


def download_url(url):
    p = urlsplit(url)
    if p.scheme != "https" or p.hostname not in {"dropbox.com", "www.dropbox.com", "dl.dropboxusercontent.com"}:
        raise ValueError("Expected an HTTPS Dropbox shared link")
    query = dict(parse_qsl(p.query)); query.pop("raw", None); query["dl"] = "1"
    return urlunsplit((p.scheme, p.netloc, p.path, urlencode(query), ""))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str))
    os.replace(tmp, path)


def validate_fits(path):
    from astropy.io import fits
    with fits.open(path, memmap=False, lazy_load_hdus=False) as hdus:
        hdus.verify("exception")
        images = [h for h in hdus if int(h.header.get("NAXIS", 0)) >= 2]
        if not images:
            raise ValueError("FITS has no image HDU")
        for h in images:
            info = h.fileinfo()
            if info and info["datLoc"] + info["datSpan"] > Path(path).stat().st_size:
                raise ValueError("Truncated FITS payload")
            if "CHECKSUM" in h.header and h.verify_checksum() != 1:
                raise ValueError("FITS CHECKSUM mismatch")


def import_one(item, root, retries=4):
    root = Path(root)
    relative = safe_relative(item["path"])
    dest = root / "raw" / relative
    receipt = root / "receipts" / relative.with_name(relative.name + ".json")
    if dest.exists() and receipt.exists():
        old = json.loads(receipt.read_text())
        if old.get("bytes") == dest.stat().st_size and old.get("sha256") == digest(dest):
            return {**old, "status": "verified_existing"}
    # A runtime may stop between moving the raw file and writing its receipt.
    # Re-download to a temporary file, compare, and adopt only identical bytes.
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last = None
    for attempt in range(retries):
        try:
            # Restart this one incomplete file, while retaining all completed files.
            h = hashlib.sha256(); size = 0
            with requests.get(download_url(item["url"]), stream=True, timeout=(30, 180)) as r:
                r.raise_for_status()
                if "text/html" in r.headers.get("Content-Type", ""):
                    raise ValueError("Dropbox returned a web page rather than a file")
                expected = int(r.headers.get("Content-Length", 0))
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(4 * 1024**2):
                        if chunk:
                            f.write(chunk); h.update(chunk); size += len(chunk)
                if expected and size != expected:
                    raise ValueError("Incomplete HTTP payload")
            fits_error = None
            if relative.suffix.lower() in {".fits", ".fit", ".fts"}:
                try:
                    validate_fits(tmp)
                except Exception as e:
                    # Preserve a damaged source too; quality screening excludes it.
                    fits_error = type(e).__name__ + ": " + str(e)
            if dest.exists():
                if digest(dest) != h.hexdigest():
                    raise FileExistsError(f"Existing original differs: {relative}")
                tmp.unlink()
            else:
                os.replace(tmp, dest)
            result = {"path": str(relative), "bytes": size, "sha256": h.hexdigest(),
                      "night": item.get("night", ""), "status": "downloaded", "fits_error": fits_error,
                      "archived_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            atomic_json(receipt, result)
            return result
        except Exception as e:
            last = type(e).__name__ + ": " + str(e).split(" for url:")[0]
            if attempt + 1 < retries:
                time.sleep(min(30, 2 ** (attempt + 1)))
    # Do not log access-bearing URLs.
    return {"path": str(relative), "status": "failed", "error": last}


def import_manifest(manifest, root, workers=4):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    files = manifest["files"]
    paths = [str(safe_relative(x["path"])) for x in files]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate destination paths in manifest")
    atomic_json(root / "source_manifest.private.json", manifest)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(import_one, item, root): item for item in files}
        for job in as_completed(jobs):
            try:
                r = job.result()
            except Exception as e:
                r = {"path": jobs[job]["path"], "status": "failed", "error": str(e)}
            results.append(r)
            print(f"{len(results)}/{len(files)} {r['status']}: {r['path']}", flush=True)
            atomic_json(root / "transfer_report.json", results)
    return results


def import_folder_link(url, root, label, staging="/content/omi_staging", *, refresh=True, keep_staging=False):
    """Future runs: paste one observing-night folder link, no file list needed.

    Downloads a ZIP to Colab's temporary disk, safely extracts into Drive,
    checks FITS and hashes, and keeps completed originals on repeat runs.
    Large archives should be split into observing-night links to fit disk.
    """
    relative = safe_relative(label); root = Path(root)
    stage = Path(staging); stage.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    zip_path = stage / (key + ".zip")
    # A shared night folder may receive more exposures. Do not silently reuse
    # an older ZIP snapshot; reuse is only an explicit extraction-resume option.
    if refresh or not zip_path.exists() or not zipfile.is_zipfile(zip_path):
        tmp = zip_path.with_suffix(".part")
        with requests.get(download_url(url), stream=True, timeout=(30, 180)) as r:
            r.raise_for_status()
            if "text/html" in r.headers.get("Content-Type", ""):
                raise ValueError("Shared link did not return a downloadable folder")
            n = int(r.headers.get("Content-Length", 0))
            if n and n > shutil.disk_usage(stage).free - 1024**3:
                raise ValueError("Not enough Colab disk; use a smaller night folder")
            with tmp.open("wb") as out:
                for chunk in r.iter_content(8 * 1024**2):
                    if shutil.disk_usage(stage).free < 512 * 1024**2:
                        raise ValueError("Colab staging disk is full; use a smaller folder")
                    out.write(chunk)
        if not zipfile.is_zipfile(tmp):
            raise ValueError("Dropbox response is not a ZIP archive")
        os.replace(tmp, zip_path)
    results = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            member = safe_relative(info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("ZIP symlinks are not accepted")
            rel = relative / member
            dest = root / "raw" / rel
            receipt = root / "receipts" / rel.with_name(rel.name + ".json")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            h = hashlib.sha256()
            with archive.open(info) as src, tmp.open("wb") as out:
                for chunk in iter(lambda: src.read(8 * 1024**2), b""):
                    out.write(chunk); h.update(chunk)
            fits_error = None
            if dest.suffix.lower() in {".fit", ".fits", ".fts"}:
                try:
                    validate_fits(tmp)
                except Exception as e:
                    fits_error = type(e).__name__ + ": " + str(e)
            sha = h.hexdigest()
            if dest.exists():
                if digest(dest) != sha:
                    raise FileExistsError(f"Original differs; choose another label: {rel}")
                tmp.unlink()  # only the newly generated temporary copy
            else:
                os.replace(tmp, dest)
            r = {"path": str(rel), "sha256": sha, "bytes": info.file_size,
                 "night": label, "status": "archived", "fits_error": fits_error}
            atomic_json(receipt, r); results.append(r)
    atomic_json(root / "folder_imports" / (key + ".json"), results)
    if not keep_staging:
        zip_path.unlink()  # verified originals and receipts are already on Drive
    return results
