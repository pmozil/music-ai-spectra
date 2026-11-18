#!/usr/bin/env python3
"""Download/extract every audio file referenced by a MusicCaps/FakeMusicCaps manifest.

Human rows (label=0): download the [start_s, end_s] YouTube segment with yt-dlp.
AI rows (label=1): download and verify the official FakeMusicCaps ZIP, then extract
only the archive members matching each row's YouTube ID and generator.

The input manifest is never modified. A new manifest with local audio_path values,
per-row download_status/download_error fields, and a status CSV are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

FAKEMUSICCAPS_URL = "https://zenodo.org/records/15063698/files/FakeMusicCaps.zip?download=1"
FAKEMUSICCAPS_MD5 = "db418dc95ab7dc378a55f29d6021fd66"
FAKEMUSICCAPS_APPROX_BYTES = 13_850_000_000
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm", ".wma"}
YTID_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, archive: Path, expected_md5: str) -> None:
    """Resumable HTTP download, followed by the repository's expected MD5 check."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    part = archive.with_suffix(archive.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "music-ai-spectra-course-project/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    with requests.get(url, headers=headers, stream=True, timeout=(30, 180), allow_redirects=True) as response:
        if offset and response.status_code == 206:
            mode, initial = "ab", offset
        elif offset and response.status_code == 200:
            # Server ignored Range; restart rather than corrupting the ZIP.
            mode, initial = "wb", 0
        else:
            response.raise_for_status()
            mode, initial = "wb", 0
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        total = initial + int(content_length) if content_length and content_length.isdigit() else None
        with part.open(mode) as out, tqdm(total=total, initial=initial, unit="B", unit_scale=True,
                                          desc="FakeMusicCaps ZIP") as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)
                    bar.update(len(chunk))

    actual = md5_file(part)
    if actual.lower() != expected_md5.lower():
        raise RuntimeError(
            f"FakeMusicCaps ZIP MD5 mismatch: expected {expected_md5}, got {actual}. "
            f"The partial archive is kept at {part}; rerun to resume."
        )
    part.replace(archive)
    print(f"Verified archive: {archive}")


def ensure_archive(args: argparse.Namespace) -> Path:
    archive = args.fake_dir / "FakeMusicCaps.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        free = shutil.disk_usage(archive.parent).free
        if free < FAKEMUSICCAPS_APPROX_BYTES:
            print(f"Warning: less than ~13.9 GB free for the archive ({free / 1e9:.1f} GB available).", file=sys.stderr)
        print(f"Downloading official FakeMusicCaps archive from {FAKEMUSICCAPS_URL}")
        download_archive(args.fake_url, archive, args.fake_md5)
    else:
        actual = md5_file(archive)
        if actual.lower() != args.fake_md5.lower():
            raise RuntimeError(f"Existing archive checksum mismatch ({actual}); remove {archive} and rerun.")
        print(f"Using verified existing archive: {archive}")
    return archive


def get_archive_candidates(zf: zipfile.ZipFile, ytids: set[str]) -> dict[str, list[str]]:
    """Index archive audio files by an exact 11-character YouTube ID in file stem."""
    found: dict[str, list[str]] = {ytid: [] for ytid in ytids}
    for info in zf.infolist():
        if info.is_dir():
            continue
        member = info.filename
        path = PurePosixPath(member)
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        for match in YTID_RE.finditer(path.stem):
            ytid = match.group(0)
            if ytid in found:
                found[ytid].append(member)
    return found


def select_fake_member(members: list[str], generator: str) -> str | None:
    if not members:
        return None
    gen = str(generator).strip().lower()
    # build_manifest.py defines generator as the immediate parent directory.
    exact_parent = [m for m in members if PurePosixPath(m).parent.name.lower() == gen]
    if len(exact_parent) == 1:
        return exact_parent[0]
    if len(exact_parent) > 1:
        return sorted(exact_parent)[0]
    # Be tolerant of a wrapper directory or minor layout differences.
    path_match = [m for m in members if any(gen == part.lower() for part in PurePosixPath(m).parts[:-1])]
    if len(path_match) == 1:
        return path_match[0]
    if len(path_match) > 1:
        return sorted(path_match)[0]
    # Only choose a fallback if the ID has exactly one synthetic file in the ZIP.
    return members[0] if len(members) == 1 else None


def extract_fake_rows(df: pd.DataFrame, args: argparse.Namespace, archive: Path) -> dict[str, dict[str, Any]]:
    fake_rows = df[df["label"].astype(int).eq(1)]
    needed_ids = set(fake_rows["ytid"].astype(str))
    results: dict[str, dict[str, Any]] = {}
    (args.fake_dir / "extracted").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        bad_member = zf.testzip()
        if bad_member:
            raise RuntimeError(f"ZIP integrity test failed at {bad_member}")
        indexed = get_archive_candidates(zf, needed_ids)
        for row in tqdm(fake_rows.itertuples(index=False), total=len(fake_rows), desc="Extract FakeMusicCaps"):
            sample_id, ytid, generator = str(row.sample_id), str(row.ytid), str(row.generator)
            try:
                member = select_fake_member(indexed.get(ytid, []), generator)
                if member is None:
                    raise FileNotFoundError(
                        f"No unambiguous FakeMusicCaps archive member for ytid={ytid}, generator={generator}; "
                        f"candidates={indexed.get(ytid, [])[:5]}"
                    )
                basename = PurePosixPath(member).name
                out_dir = args.fake_dir / "extracted" / generator
                out_dir.mkdir(parents=True, exist_ok=True)
                destination = out_dir / basename
                if args.force or not destination.exists() or destination.stat().st_size != zf.getinfo(member).file_size:
                    temp = destination.with_suffix(destination.suffix + ".part")
                    with zf.open(member) as source, temp.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    temp.replace(destination)
                results[sample_id] = {"audio_path": str(destination.resolve()), "download_status": "extracted", "download_error": ""}
            except Exception as exc:
                results[sample_id] = {"audio_path": "", "download_status": "failed", "download_error": f"{type(exc).__name__}: {exc}"}
    return results


def probe_duration(path: Path) -> float:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is required; install FFmpeg and ensure ffprobe is on PATH")
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(json.loads(completed.stdout).get("format", {}).get("duration", 0.0))


def download_human(row: dict[str, Any], audio_dir: Path, retries: int, force: bool) -> dict[str, Any]:
    sample_id, ytid = str(row["sample_id"]), str(row["ytid"])
    start, end = float(row["start_s"]), float(row["end_s"])
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", ytid):
        return {"sample_id": sample_id, "audio_path": "", "download_status": "failed", "download_error": "Invalid YouTube ID"}
    if end <= start:
        return {"sample_id": sample_id, "audio_path": "", "download_status": "failed", "download_error": f"Invalid segment: start={start}, end={end}"}
    audio_dir.mkdir(parents=True, exist_ok=True)
    existing = next((p for p in audio_dir.glob(f"{ytid}.*") if p.is_file() and not p.name.endswith((".part", ".ytdl"))), None)
    if existing and not force:
        try:
            duration = probe_duration(existing)
            if max(0.0, end - start - 1.0) <= duration <= end - start + 1.0:
                return {"sample_id": sample_id, "audio_path": str(existing.resolve()), "download_status": "already_present", "download_error": ""}
        except Exception:
            pass

    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import download_range_func
    except ImportError:
        return {"sample_id": sample_id, "audio_path": "", "download_status": "failed", "download_error": "yt-dlp missing; install it with python -m pip install yt-dlp"}

    temp_dir = audio_dir / ".download_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(temp_dir / f"{ytid}.%(ext)s")
    options = {
        "format": "bestaudio/best", "outtmpl": output_template, "noplaylist": True,
        "quiet": True, "no_warnings": True, "ignoreerrors": False,
        "retries": 3, "fragment_retries": 3, "extractor_retries": 3,
        "socket_timeout": 30, "overwrites": True,
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
    }
    last_error = "unknown error"
    for attempt in range(1, max(1, retries) + 1):
        try:
            with YoutubeDL(options) as ydl:
                result = ydl.download([f"https://www.youtube.com/watch?v={ytid}"])
            candidates = [p for p in temp_dir.glob(f"{ytid}.*") if p.is_file() and not p.name.endswith((".part", ".ytdl"))]
            if not candidates:
                raise RuntimeError(f"yt-dlp produced no file (return code {result}); the video may be unavailable")
            source = max(candidates, key=lambda p: p.stat().st_size)
            duration = probe_duration(source)
            expected = end - start
            if not max(0.0, expected - 1.0) <= duration <= expected + 1.0:
                raise RuntimeError(f"Downloaded duration {duration:.2f}s is not near expected {expected:.2f}s")
            destination = audio_dir / f"{ytid}{source.suffix}"
            if destination.exists():
                destination.unlink()
            source.replace(destination)
            for stale in temp_dir.glob(f"{ytid}.*"):
                stale.unlink(missing_ok=True)
            return {"sample_id": sample_id, "audio_path": str(destination.resolve()), "download_status": "downloaded", "download_error": ""}
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:1200]
            time.sleep(min(2 * attempt, 8) + random.random())
    return {"sample_id": sample_id, "audio_path": "", "download_status": "failed", "download_error": last_error}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    ap.add_argument("--output-manifest", type=Path, default=Path("data/processed/manifest_downloaded.csv"))
    ap.add_argument("--status-csv", type=Path, default=Path("data/raw/manifest_download_status.csv"))
    ap.add_argument("--human-dir", type=Path, default=Path("data/raw/musiccaps_audio"))
    ap.add_argument("--fake-dir", type=Path, default=Path("data/raw/fakemusiccaps"))
    ap.add_argument("--fake-url", default=FAKEMUSICCAPS_URL)
    ap.add_argument("--fake-md5", default=FAKEMUSICCAPS_MD5)
    ap.add_argument("--only", choices=("both", "human", "ai"), default="both")
    ap.add_argument("--limit-pairs", type=int, default=0, help="Process only the first N distinct ytid pairs")
    ap.add_argument("--workers", type=int, default=2, help="Concurrent human YouTube downloads; keep low")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="Redownload/overwrite files even if already present")
    args = ap.parse_args()

    if not args.manifest.is_file():
        ap.error(f"Manifest not found: {args.manifest}")
    df = pd.read_csv(args.manifest, dtype={"ytid": str})
    required = {"sample_id", "ytid", "audio_path", "label", "generator", "start_s", "end_s"}
    missing = required - set(df.columns)
    if missing:
        ap.error(f"Manifest missing required columns: {sorted(missing)}")
    if not df["label"].isin([0, 1]).all():
        ap.error("Manifest label column must contain 0 for human and 1 for AI-generated rows")
    if args.limit_pairs > 0:
        ids = df["ytid"].drop_duplicates().head(args.limit_pairs)
        df = df[df["ytid"].isin(ids)].copy()

    results: dict[str, dict[str, Any]] = {}
    if args.only in ("both", "human"):
        human = df[df.label.astype(int).eq(0)]
        rows = [r._asdict() for r in human.itertuples(index=False)]
        print(f"Human clips to process: {len(rows)}")
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(download_human, row, args.human_dir, args.retries, args.force) for row in rows]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Download MusicCaps human clips"):
                rec = future.result()
                results[rec["sample_id"]] = rec
                args.status_csv.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(results.values()).to_csv(args.status_csv, index=False)

    if args.only in ("both", "ai"):
        fake = df[df.label.astype(int).eq(1)]
        print(f"AI-generated clips to extract: {len(fake)}")
        archive = ensure_archive(args)
        results.update(extract_fake_rows(df, args, archive))

    out = df.copy()
    out["audio_path_original"] = out["audio_path"]
    out["download_status"] = out["sample_id"].map(lambda sid: results.get(str(sid), {}).get("download_status", "not_requested"))
    out["download_error"] = out["sample_id"].map(lambda sid: results.get(str(sid), {}).get("download_error", ""))
    successful_path = out["sample_id"].map(lambda sid: results.get(str(sid), {}).get("audio_path", ""))
    out["audio_path"] = successful_path.where(successful_path.ne(""), "")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_manifest, index=False)
    out[["sample_id", "ytid", "label", "generator", "audio_path", "download_status", "download_error"]].to_csv(args.status_csv, index=False)
    status = out["download_status"].value_counts(dropna=False).to_dict()
    print("\nDownload summary:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print(f"\nManifest with local paths: {args.output_manifest}")
    print(f"Status log: {args.status_csv}")
    failed = out[out.download_status.eq("failed")]
    if not failed.empty:
        print(f"\n{len(failed)} rows failed. Inspect download_error in {args.output_manifest} and rerun.", file=sys.stderr)
    return 0 if failed.empty else 2

if __name__ == "__main__":
    raise SystemExit(main())
