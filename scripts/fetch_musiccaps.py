#!/usr/bin/env python3
"""Fetch MusicCaps metadata and resumably download available 10-second clips.

Uses the public Hugging Face MusicCaps metadata and yt-dlp for the matching
YouTube segment. It does not bypass access controls; deleted/private/unavailable
clips are logged so the project can report the surviving subset transparently.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def get_metadata(output_csv: Path, refresh: bool = False) -> pd.DataFrame:
    if output_csv.exists() and not refresh:
        return pd.read_csv(output_csv, dtype={"ytid": str})
    # Download the public CSV directly instead of relying on the heavier
    # Hugging Face `datasets` loader. This file contains metadata, not audio.
    url = "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv?download=true"
    try:
        response = requests.get(
            url, timeout=60, headers={"User-Agent": "music-ai-svd-course-project/0.1"}
        )
        response.raise_for_status()
        df = pd.read_csv(io.BytesIO(response.content))
    except Exception as exc:
        raise RuntimeError(
            "Could not download MusicCaps metadata. Check network access, or place the official "
            f"musiccaps-public.csv at {output_csv}. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    required = {"ytid", "start_s", "end_s", "caption"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"MusicCaps metadata is missing expected columns: {sorted(missing)}"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def _download_one(
    row: dict[str, object], out_dir: Path, retries: int
) -> dict[str, object]:
    ytid = str(row["ytid"])
    start, end = float(row["start_s"]), float(row["end_s"])
    if not ytid or len(ytid) != 11:
        return {
            "ytid": ytid,
            "status": "failed",
            "error": "invalid YouTube ID",
            "path": "",
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = next(
        (
            p
            for p in out_dir.iterdir()
            if p.is_file()
            and p.stem == ytid
            and not p.name.endswith((".part", ".ytdl"))
        ),
        None,
    )
    if existing:
        return {
            "ytid": ytid,
            "status": "downloaded_existing",
            "error": "",
            "path": str(existing),
        }
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import download_range_func
    except ImportError:
        return {
            "ytid": ytid,
            "status": "failed",
            "error": "yt-dlp not installed; run pip install -e .",
            "path": "",
        }
    tmp = out_dir / ".download_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(tmp / f"{ytid}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": 20,
        "overwrites": False,
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "extractor_retries": 2,
    }
    last_err = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            with YoutubeDL(options) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={ytid}"])
            candidates = [
                p
                for p in tmp.glob(f"{ytid}.*")
                if p.is_file() and not p.name.endswith((".part", ".ytdl"))
            ]
            if not candidates:
                raise RuntimeError(
                    "yt-dlp completed without an output file; clip may be unavailable"
                )
            source = max(candidates, key=lambda p: p.stat().st_size)
            # Verify the sectioning API returned the captioned 10-second slice,
            # not an accidentally downloaded full-length source track.
            if shutil.which("ffprobe") is None:
                raise RuntimeError(
                    "ffprobe is required to validate clip duration; install FFmpeg"
                )
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "json",
                    str(source),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            duration = float(
                json.loads(probe.stdout).get("format", {}).get("duration", 0.0)
            )
            if not (max(0.0, end - start - 1.0) <= duration <= (end - start + 1.0)):
                source.unlink(missing_ok=True)
                raise RuntimeError(
                    f"downloaded section duration {duration:.2f}s is not near {end-start:.2f}s; refusing to save full/wrong segment"
                )
            destination = out_dir / f"{ytid}{source.suffix}"
            if source.resolve() != destination.resolve():
                if destination.exists():
                    source.unlink(missing_ok=True)
                else:
                    source.replace(destination)
            for p in tmp.glob(f"{ytid}.*"):
                p.unlink(missing_ok=True)
            return {
                "ytid": ytid,
                "status": "downloaded",
                "error": "",
                "path": str(destination),
            }
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"[:1000]
            time.sleep(min(3 * attempt, 10) + random.random())
    # Leave partial downloads in .download_tmp so yt-dlp may resume them on a later run.
    return {"ytid": ytid, "status": "failed", "error": last_err, "path": ""}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metadata-csv", type=Path, default=Path("data/raw/musiccaps_metadata.csv")
    )
    ap.add_argument("--audio-dir", type=Path, default=Path("data/raw/musiccaps_audio"))
    ap.add_argument(
        "--status-csv",
        type=Path,
        default=Path("data/raw/musiccaps_download_status.csv"),
    )
    ap.add_argument("--limit", type=int, default=0, help="First N rows only (0 = all)")
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Keep concurrency low to avoid aggressive requests",
    )
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--refresh-metadata", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    df = get_metadata(args.metadata_csv, refresh=args.refresh_metadata)
    if args.limit > 0:
        df = df.head(args.limit)
    successes: dict[str, dict[str, object]] = {}
    if args.status_csv.exists() and not args.retry_failed:
        old = pd.read_csv(args.status_csv, dtype={"ytid": str}).fillna("")
        successes = {
            str(r["ytid"]): r.to_dict()
            for _, r in old.iterrows()
            if r.get("status") in {"downloaded", "downloaded_existing"}
        }
    rows = [r.to_dict() for _, r in df.iterrows() if str(r["ytid"]) not in successes]
    results = list(successes.values())
    print(
        f"Metadata rows: {len(df)}; to attempt this run: {len(rows)}; audio directory: {args.audio_dir.resolve()}"
    )
    args.status_csv.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(_download_one, row, args.audio_dir, max(1, args.retries))
            for row in rows
        ]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="MusicCaps clips"
        ):
            results.append(fut.result())
            pd.DataFrame(results).to_csv(args.status_csv, index=False)
    status = pd.DataFrame(results)
    status.to_csv(args.status_csv, index=False)
    counts = status.get("status", pd.Series(dtype=str)).value_counts().to_dict()
    print(f"Download status: {counts}\nStatus log: {args.status_csv}")
    if counts.get("failed", 0):
        print(
            "Some clips are unavailable or failed; inspect the error column and rerun with --retry-failed.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
