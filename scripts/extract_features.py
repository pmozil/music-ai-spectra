#!/usr/bin/env python3
"""Extract per-clip SVD value, singular-vector and window-angle features."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from music_svd_detection.parallel import limit_worker_threads
from music_svd_detection.spectral import (
    SpectralConfig,
    log_magnitude_spectrogram,
    principal_angle_features,
    svd_clip_features,
)


def _extract_one(record: dict, spec_cfg: SpectralConfig, vector_modes: int, window_rank: int) -> dict:
    """Worker: extract all feature families for one audio clip."""
    row = dict(record)
    try:
        audio, sr = sf.read(str(row["processed_path"]), dtype="float32", always_2d=False)
        if np.asarray(audio).ndim == 2:
            audio = np.asarray(audio).mean(axis=1)
        if int(sr) != spec_cfg.sample_rate:
            raise ValueError(f"expected {spec_cfg.sample_rate} Hz, got {sr}")
        logmag = log_magnitude_spectrogram(audio, int(sr), spec_cfg)
        row.update(
            svd_clip_features(
                logmag,
                top_k=spec_cfg.local_top_k,
                vector_modes=vector_modes,
            )
        )
        row.update(
            principal_angle_features(
                logmag,
                top_k=window_rank,
                window_frames=spec_cfg.subspace_window_frames,
                step_frames=spec_cfg.subspace_step_frames,
            )
        )
        row["spectrogram_frequency_bins"] = float(logmag.shape[0])
        row["spectrogram_frames"] = float(logmag.shape[1])
        row["window_rank_used"] = int(window_rank)
        row["feature_status"] = "ok"
        row.pop("feature_error", None)
    except Exception as exc:  # keep a per-clip diagnostic rather than aborting the whole corpus
        row["feature_status"] = "failed"
        row["feature_error"] = f"{type(exc).__name__}: {exc}"[:1000]
    return row


def _config_from_json(cfg: dict) -> SpectralConfig:
    return SpectralConfig(
        sample_rate=int(cfg["sample_rate"]),
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
        log_epsilon=float(cfg["log_epsilon"]),
        local_top_k=int(cfg["local_top_k"]),
        subspace_window_frames=int(cfg["subspace_window_frames"]),
        subspace_step_frames=int(cfg["subspace_step_frames"]),
        max_freq_hz=float(cfg.get("max_freq_hz", 0)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_harmonized.csv"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/features.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--limit", type=int, default=0, help="First N complete human/synthetic pairs (0 = all)")
    ap.add_argument(
        "--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)),
        help="Parallel worker processes (1 disables multiprocessing; default up to 4)",
    )
    ap.add_argument("--chunksize", type=int, default=8, help="Tasks per process-pool dispatch chunk")
    args = ap.parse_args()
    if args.workers < 1 or args.chunksize < 1:
        ap.error("--workers and --chunksize must be positive")

    cfg = json.loads(args.config.read_text())
    spec_cfg = _config_from_json(cfg)
    vector_modes = int(cfg.get("vector_modes", 4))
    # Window-subspace rank is deliberately independent of the global clip-SVD rank.
    window_rank = int(cfg.get("window_rank", spec_cfg.local_top_k))

    manifest = pd.read_csv(args.manifest)
    required = {"processing_status", "processed_path", "ytid", "label", "sample_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    manifest = manifest[
        manifest["processing_status"].eq("ok") & manifest["processed_path"].notna()
    ].copy()
    # Preserve balance: do not retain an orphan if preprocessing lost its pair mate.
    counts = manifest.groupby("ytid")["label"].agg(["count", "sum"])
    good_ids = counts.index[(counts["count"] == 2) & (counts["sum"] == 1)]
    manifest = manifest[manifest["ytid"].isin(good_ids)]
    if args.limit > 0:
        chosen_ids = manifest["ytid"].drop_duplicates().head(args.limit)
        manifest = manifest[manifest["ytid"].isin(chosen_ids)]
    if manifest.empty:
        raise ValueError("No complete, successfully preprocessed human/AI pairs to extract")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest sample_id values must be unique")

    records = manifest.to_dict("records")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    successful = failed = completed = 0
    batch: list[dict] = []
    extract = partial(
        _extract_one, spec_cfg=spec_cfg, vector_modes=vector_modes, window_rank=window_rank
    )

    def save_batch(rows: list[dict]) -> None:
        nonlocal successful, failed, completed
        if not rows:
            return
        result = pd.DataFrame(rows)
        result.to_csv(args.output, mode="a", header=not args.output.exists(), index=False)
        statuses = result["feature_status"].value_counts().to_dict()
        successful += int(statuses.get("ok", 0))
        failed += int(statuses.get("failed", 0))
        completed += len(rows)

    with tqdm(total=len(records), desc=f"Extracting features ({args.workers} workers)") as bar:
        if args.workers == 1:
            iterator = map(extract, records)
            for row in iterator:
                batch.append(row)
                bar.update(1)
                if len(batch) >= 100:
                    save_batch(batch)
                    batch = []
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers, initializer=limit_worker_threads
            ) as pool:
                for row in pool.map(extract, records, chunksize=args.chunksize):
                    batch.append(row)
                    bar.update(1)
                    if len(batch) >= 100:
                        save_batch(batch)
                        batch = []
    save_batch(batch)

    print(f"Feature rows: {completed}; successful: {successful}; failed: {failed}")
    print(f"SVD rank: {spec_cfg.local_top_k}; window rank: {window_rank}; vector modes: {vector_modes}")
    print(f"Features: {args.output}")
    if successful == 0:
        print("ERROR: no clips were successfully processed; refusing a successful exit")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
