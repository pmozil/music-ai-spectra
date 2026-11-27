#!/usr/bin/env python3
"""Compute one joint SVD per caption-matched human/AI pair."""

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
    joint_pair_svd_features,
    log_magnitude_spectrogram,
)


def read_logmag(path: str, cfg: SpectralConfig) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if np.asarray(audio).ndim == 2:
        audio = np.asarray(audio).mean(axis=1)
    return log_magnitude_spectrogram(audio, int(sr), cfg)


def _process_pair(task: tuple[object, dict, dict, str, SpectralConfig, int]) -> list[dict]:
    pair_id, human, ai, pair_id_col, cfg, rank = task
    rows = []
    try:
        lh = read_logmag(str(human["processed_path"]), cfg)
        la = read_logmag(str(ai["processed_path"]), cfg)
        hf, af = joint_pair_svd_features(lh, la, top_k=rank)
        for item, features, side in ((human, hf, "human"), (ai, af, "ai")):
            rows.append({
                "sample_id": item["sample_id"], pair_id_col: pair_id,
                "split": item.get("split", ""), "pair_side": side,
                "pair_svd_status": "ok", **features,
            })
    except Exception as exc:
        for item, side in ((human, "human"), (ai, "ai")):
            rows.append({
                "sample_id": item["sample_id"], pair_id_col: pair_id,
                "split": item.get("split", ""), "pair_side": side,
                "pair_svd_status": "failed",
                "pair_svd_error": f"{type(exc).__name__}: {exc}"[:1000],
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_harmonized.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/pairwise_svd_features.csv"))
    ap.add_argument("--pair-id", default="ytid", help="Column identifying matched caption pairs")
    ap.add_argument("--rank", type=int, default=8, help="Number of leading joint modes to describe")
    ap.add_argument("--limit", type=int, default=0, help="Maximum number of pairs (0 = all)")
    ap.add_argument(
        "--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)),
        help="Parallel worker processes (1 disables multiprocessing)",
    )
    ap.add_argument("--chunksize", type=int, default=2)
    args = ap.parse_args()
    if args.workers < 1 or args.chunksize < 1 or args.rank < 1:
        ap.error("--workers, --chunksize and --rank must be positive")

    cfgd = json.loads(args.config.read_text())
    cfg = SpectralConfig(
        sample_rate=int(cfgd["sample_rate"]), n_fft=int(cfgd["n_fft"]),
        hop_length=int(cfgd["hop_length"]), log_epsilon=float(cfgd["log_epsilon"]),
        local_top_k=int(cfgd["local_top_k"]),
        subspace_window_frames=int(cfgd["subspace_window_frames"]),
        subspace_step_frames=int(cfgd["subspace_step_frames"]),
        max_freq_hz=float(cfgd.get("max_freq_hz", 0)),
    )
    df = pd.read_csv(args.manifest)
    df = df[df["processing_status"].eq("ok") & df["processed_path"].notna()].copy()
    if args.pair_id not in df or "label" not in df:
        raise ValueError(f"Manifest requires {args.pair_id!r} and 'label' columns")
    if "sample_id" not in df:
        raise ValueError("Manifest requires a unique sample_id column")
    if df["sample_id"].duplicated().any():
        raise ValueError("Manifest sample_id values must be unique")

    tasks = []
    for pair_id, group in df.groupby(args.pair_id, sort=False):
        labels = group["label"].astype(int).tolist()
        if len(group) == 2 and sorted(labels) == [0, 1]:
            ordered = group.sort_values("label")
            tasks.append((pair_id, ordered.iloc[0].to_dict(), ordered.iloc[1].to_dict(),
                          args.pair_id, cfg, args.rank))
    if args.limit > 0:
        tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("No complete pairs (exactly one label 0 and one label 1) found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    successful_pairs = failed_pairs = 0
    batch: list[dict] = []

    def save_batch(rows: list[dict]) -> None:
        nonlocal successful_pairs, failed_pairs
        if not rows:
            return
        result = pd.DataFrame(rows)
        result.to_csv(args.output, mode="a", header=not args.output.exists(), index=False)
        good = int(result["pair_svd_status"].eq("ok").sum() // 2)
        successful_pairs += good
        failed_pairs += len(result) // 2 - good

    with tqdm(total=len(tasks), desc=f"Pairwise joint SVD ({args.workers} workers)") as bar:
        if args.workers == 1:
            iterator = map(_process_pair, tasks)
            for pair_rows in iterator:
                batch.extend(pair_rows)
                bar.update(1)
                if len(batch) >= 100:
                    save_batch(batch)
                    batch = []
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=limit_worker_threads) as pool:
                for pair_rows in pool.map(_process_pair, tasks, chunksize=args.chunksize):
                    batch.extend(pair_rows)
                    bar.update(1)
                    if len(batch) >= 100:
                        save_batch(batch)
                        batch = []
    save_batch(batch)

    print(f"Pair features: {2 * (successful_pairs + failed_pairs)} rows from {len(tasks)} pairs")
    print(f"Successful pairs: {successful_pairs}; failed pairs: {failed_pairs}")
    print(f"Saved: {args.output}")
    return 0 if successful_pairs else 2


if __name__ == "__main__":
    raise SystemExit(main())
