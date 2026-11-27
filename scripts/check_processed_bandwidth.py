#!/usr/bin/env python3
"""Bandwidth-only baseline measured on the processed audio in the analysed band."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from music_svd_detection.parallel import limit_worker_threads
from music_svd_detection.spectral import raw_bandwidth_cutoff_hz


def _measure_one(task: tuple[str, float]) -> float:
    path, cap = task
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if np.asarray(audio).ndim == 2:
        audio = np.asarray(audio).mean(axis=1)
    return float(min(raw_bandwidth_cutoff_hz(audio, int(sr)), cap))


def best_threshold(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Threshold and direction maximising balanced accuracy on training data."""
    best = (-np.inf, 0.0, 1)
    for threshold in np.unique(x):
        for sign in (1, -1):
            pred = (sign * x) >= (sign * threshold)
            score = balanced_accuracy_score(y, pred)
            if score > best[0]:
                best = (float(score), float(threshold), sign)
    return best[1], best[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_harmonized.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/processed_bandwidth.csv"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--chunksize", type=int, default=8)
    args = ap.parse_args()
    if args.workers < 1 or args.chunksize < 1:
        ap.error("--workers and --chunksize must be positive")

    cfg = json.loads(args.config.read_text())
    cap = float(cfg.get("max_freq_hz", 0)) or float(cfg["sample_rate"]) / 2.0
    df = pd.read_csv(args.manifest)
    if "processing_status" in df:
        df = df[df["processing_status"].eq("ok")].copy()
    if args.limit:
        df = df[df["ytid"].isin(df["ytid"].drop_duplicates().head(args.limit))]
    tasks = list(zip(df["processed_path"].astype(str), [cap] * len(df)))

    with tqdm(total=len(tasks), desc=f"Processed bandwidth ({args.workers} workers)") as bar:
        if args.workers == 1:
            values = []
            for task in tasks:
                values.append(_measure_one(task))
                bar.update(1)
        else:
            values = []
            with ProcessPoolExecutor(max_workers=args.workers, initializer=limit_worker_threads) as pool:
                for value in pool.map(_measure_one, tasks, chunksize=args.chunksize):
                    values.append(value)
                    bar.update(1)

    df["processed_bandwidth_hz"] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df[["sample_id", "ytid", "caption_group", "label", "generator", "split", "processed_bandwidth_hz"]].to_csv(args.output, index=False)

    print(f"Analysed band: 0 to {cap:.0f} Hz")
    print(df.groupby("generator")["processed_bandwidth_hz"].describe(percentiles=[0.05, 0.5]).round(0).to_string())
    train, test = df[df["split"].eq("train")], df[df["split"].eq("test")]
    threshold, sign = best_threshold(train["processed_bandwidth_hz"].to_numpy(), train["label"].to_numpy())
    pred = (sign * test["processed_bandwidth_hz"].to_numpy()) >= (sign * threshold)
    print(f"\nBandwidth-only on processed audio, test accuracy: {accuracy_score(test['label'], pred):.3f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(test['label'], pred):.3f}")
    if test["label"].nunique() == 2 and test["processed_bandwidth_hz"].nunique() > 1:
        auc = roc_auc_score(test["label"], test["processed_bandwidth_hz"])
        print(f"ROC-AUC (direction-free): {max(auc, 1 - auc):.3f}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
