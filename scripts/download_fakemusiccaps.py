#!/usr/bin/env python3
"""Resume-download the official FakeMusicCaps v2 Zenodo ZIP and verify MD5."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

URL = "https://zenodo.org/records/15063698/files/FakeMusicCaps.zip?download=1"
EXPECTED_MD5 = "db418dc95ab7dc378a55f29d6021fd66"
APPROX_BYTES = 13_850_000_000  # Zenodo lists the file as 12.9 GB.


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, destination: Path, expected_md5: str, timeout: int = 60) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "music-ai-svd-course-project/0.1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    with requests.get(
        url, headers=headers, stream=True, timeout=timeout, allow_redirects=True
    ) as r:
        if existing and r.status_code == 200:
            # Server ignored Range: restart to avoid corrupt concatenation.
            existing, mode = 0, "wb"
        elif existing and r.status_code == 206:
            mode = "ab"
        else:
            r.raise_for_status()
            mode = "wb"
        r.raise_for_status()
        length = r.headers.get("Content-Length")
        total = int(length) + existing if length and length.isdigit() else None
        with part.open(mode) as f, tqdm(
            total=total,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc="FakeMusicCaps.zip",
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    got = md5_file(part)
    if got.lower() != expected_md5.lower():
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"MD5 mismatch for downloaded archive: expected {expected_md5}, got {got}. Removed the corrupt partial; retry to start again."
        )
    part.replace(destination)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("data/raw/fakemusiccaps"))
    p.add_argument("--url", default=URL)
    p.add_argument("--expected-md5", default=EXPECTED_MD5)
    p.add_argument(
        "--extract",
        action="store_true",
        help="Extract verified ZIP under output-dir/extracted",
    )
    p.add_argument("--force-download", action="store_true")
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "FakeMusicCaps.zip"
    if shutil.disk_usage(out).free < APPROX_BYTES and not archive.exists():
        print(
            f"Warning: only {shutil.disk_usage(out).free/1e9:.1f} GB free; archive is listed as 12.9 GB.",
            file=sys.stderr,
        )
    if args.force_download:
        archive.unlink(missing_ok=True)
        archive.with_suffix(archive.suffix + ".part").unlink(missing_ok=True)
    if archive.exists() and not args.force_download:
        got = md5_file(archive)
        if got.lower() != args.expected_md5.lower():
            print(
                f"Existing archive MD5 {got} does not match; removing it.",
                file=sys.stderr,
            )
            archive.unlink()
        else:
            print(f"Verified existing archive: {archive}")
    if not archive.exists():
        download(args.url, archive, args.expected_md5)
    if args.extract:
        target = out / "extracted"
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"ZIP integrity check failed at member {bad}")
            zf.extractall(target)
        print(f"Extracted FakeMusicCaps to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
