#!/usr/bin/env python3
"""Record raw audio provenance, estimate bandwidth baseline, and harmonize clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from music_svd_detection.audio import canonicalize_audio, decode_float_mono
from music_svd_detection.parallel import limit_worker_threads
from music_svd_detection.spectral import raw_bandwidth_cutoff_hz


def _config_fingerprint(cfg: dict) -> str:
    """Invalidate cached WAVs whenever a relevant preprocessing setting changes."""
    keys = (
        "sample_rate", "duration_seconds", "lowpass_hz", "mp3_bitrate",
        "loudness_i_lufs", "loudness_tp_db", "loudness_lra",
    )
    payload = {key: cfg.get(key) for key in keys}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _process_one(task: tuple[dict, str, dict | None, dict, bool]) -> tuple[dict, dict | None]:
    record, dst_text, previous, cfg, force = task
    sid = str(record["sample_id"])
    dst = Path(dst_text)
    fingerprint = _config_fingerprint(cfg)
    if previous and not force and previous.get("preprocess_config_hash") == fingerprint:
        cached_text = str(previous.get("processed_path", ""))
        cached = Path(cached_text)
        if cached.is_file():
            try:
                info = sf.info(str(cached))
                duration = float(info.frames) / float(info.samplerate)
                if (int(info.samplerate) == int(cfg["sample_rate"])
                        and abs(duration - float(cfg["duration_seconds"])) <= 0.05):
                    return previous, None
            except Exception:
                pass  # stale/corrupt cache: regenerate below

    src_text = str(record.get("audio_path", ""))
    try:
        src = Path(src_text)
        if not src.is_file():
            raise FileNotFoundError(f"Source audio not found: {src}")
        raw_audio, raw_sr = decode_float_mono(src)
        bandwidth = raw_bandwidth_cutoff_hz(raw_audio, raw_sr)
        raw_duration = float(len(raw_audio)) / float(raw_sr)
        expected_duration = float(cfg["duration_seconds"])
        if abs(raw_duration - expected_duration) > 1.0:
            raise ValueError(
                f"expected about {expected_duration:.1f}s raw segment, found {raw_duration:.2f}s; verify section extraction"
            )
        source_info = canonicalize_audio(
            src,
            dst,
            sample_rate=int(cfg["sample_rate"]),
            duration_seconds=float(cfg["duration_seconds"]),
            lowpass_hz=int(cfg["lowpass_hz"]),
            mp3_bitrate=str(cfg["mp3_bitrate"]),
            loudness_i_lufs=float(cfg["loudness_i_lufs"]),
            loudness_tp_db=float(cfg["loudness_tp_db"]),
            loudness_lra=float(cfg["loudness_lra"]),
        )
        record.update(source_info)
        record["raw_bandwidth_cutoff_hz"] = bandwidth
        record["processed_path"] = str(dst.resolve())
        record["preprocess_config_hash"] = fingerprint
        record["processing_status"] = "ok"
        record.pop("processing_error", None)
        return record, None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:1000]
        record["processed_path"] = ""
        record["processing_status"] = "failed"
        record["processing_error"] = error
        return record, {"sample_id": sid, "audio_path": src_text, "error": error}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    ap.add_argument("--output-manifest", type=Path, default=Path("data/processed/manifest_harmonized.csv"))
    ap.add_argument("--audio-out-dir", type=Path, default=Path("data/processed/audio"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--force", action="store_true", help="Regenerate even if cached processed WAVs exist")
    ap.add_argument(
        "--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)),
        help="Parallel worker processes (each worker may start an FFmpeg process)",
    )
    ap.add_argument("--chunksize", type=int, default=2)
    args = ap.parse_args()
    if args.workers < 1 or args.chunksize < 1:
        ap.error("--workers and --chunksize must be positive")
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)

    cfg = json.loads(args.config.read_text())
    df = pd.read_csv(args.manifest)
    if "sample_id" not in df or "audio_path" not in df:
        raise ValueError("Manifest must contain sample_id and audio_path columns")
    args.audio_out_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    old: dict[str, dict] = {}
    if args.output_manifest.exists() and not args.force:
        prior = pd.read_csv(args.output_manifest).fillna("")
        old = {str(r["sample_id"]): r.to_dict() for _, r in prior.iterrows()}
    tasks = []
    for record in df.to_dict("records"):
        sid = str(record["sample_id"])
        dst = args.audio_out_dir / f"{sid}.wav"
        tasks.append((record, str(dst), old.get(sid), cfg, args.force))

    errors: list[dict] = []
    results: list[dict] = []
    with tqdm(total=len(tasks), desc=f"Harmonizing audio ({args.workers} workers)") as bar:
        if args.workers == 1:
            iterator = map(_process_one, tasks)
            for record, error in iterator:
                results.append(record)
                if error:
                    errors.append(error)
                bar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=limit_worker_threads) as pool:
                for record, error in pool.map(_process_one, tasks, chunksize=args.chunksize):
                    results.append(record)
                    if error:
                        errors.append(error)
                    bar.update(1)

    pd.DataFrame(results).to_csv(args.output_manifest, index=False)
    error_path = args.output_manifest.with_name("preprocess_errors.csv")
    if errors:
        pd.DataFrame(errors).to_csv(error_path, index=False)
    else:
        error_path.unlink(missing_ok=True)
    status = pd.DataFrame(results)["processing_status"].value_counts().to_dict()
    print(f"Processing status: {status}\nProcessed manifest: {args.output_manifest}")
    if errors:
        print(f"Error details: {error_path}")
    return 0 if status.get("ok", 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
