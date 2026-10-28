#!/usr/bin/env python3
"""Record raw audio provenance, estimate bandwidth baseline, and harmonize clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from music_svd_detection.audio import canonicalize_audio, decode_float_mono
from music_svd_detection.spectral import raw_bandwidth_cutoff_hz


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifest.csv")
    )
    ap.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/processed/manifest_harmonized.csv"),
    )
    ap.add_argument("--audio-out-dir", type=Path, default=Path("data/processed/audio"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    cfg = json.loads(args.config.read_text())
    df = pd.read_csv(args.manifest)
    args.audio_out_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    old = {}
    if args.output_manifest.exists() and not args.force:
        prior = pd.read_csv(args.output_manifest).fillna("")
        old = {str(r["sample_id"]): r.to_dict() for _, r in prior.iterrows()}
    results = []
    errors = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Harmonizing audio"):
        record = row.to_dict()
        sid = str(record["sample_id"])
        dst = args.audio_out_dir / f"{sid}.wav"
        previous = old.get(sid)
        if (
            previous
            and Path(str(previous.get("processed_path", ""))).is_file()
            and not args.force
        ):
            results.append(previous)
            continue
        src = Path(str(record["audio_path"]))
        try:
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
            record["processing_status"] = "ok"
            results.append(record)
        except Exception as exc:
            record["processed_path"] = ""
            record["processing_status"] = "failed"
            record["processing_error"] = f"{type(exc).__name__}: {exc}"[:1000]
            errors.append(
                {
                    "sample_id": sid,
                    "audio_path": str(src),
                    "error": record["processing_error"],
                }
            )
            results.append(record)
        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(args.output_manifest, index=False)
    pd.DataFrame(results).to_csv(args.output_manifest, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(
            args.output_manifest.with_name("preprocess_errors.csv"), index=False
        )
    status = pd.DataFrame(results)["processing_status"].value_counts().to_dict()
    print(f"Processing status: {status}\nProcessed manifest: {args.output_manifest}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
