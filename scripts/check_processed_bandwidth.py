#!/usr/bin/env python3
"""Bandwidth-only baseline measured on the PROCESSED audio, inside the analysed band.

train_evaluate.py's bandwidth baseline uses raw_bandwidth_cutoff_hz from the raw
downloads, so it stays high no matter what preprocessing does. This script asks the
question that matters for the SVD features: after preprocessing and the max_freq_hz
crop, can bandwidth alone still tell human from AI? A result near 50% means the
confound is removed from what the SVD sees.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from music_svd_detection.spectral import raw_bandwidth_cutoff_hz


def best_threshold(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Threshold and direction maximising accuracy on (x, y)."""
    best = (0.0, 0.0, 1)
    for t in np.unique(x):
        for sign in (1, -1):
            acc = np.mean(((sign * x) >= (sign * t)) == y)
            if acc > best[0]:
                best = (acc, t, sign)
    return best[1], best[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_harmonized.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--output", type=Path, default=Path("artifacts/processed_bandwidth.csv"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    cap = float(cfg.get("max_freq_hz", 0)) or float(cfg["sample_rate"]) / 2.0
    df = pd.read_csv(args.manifest)
    if "processing_status" in df:
        df = df[df["processing_status"].eq("ok")].copy()
    if args.limit:
        df = df[df["ytid"].isin(df["ytid"].drop_duplicates().head(args.limit))]

    bw = []
    for path in tqdm(df["processed_path"].astype(str), desc="Processed bandwidth"):
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if np.asarray(audio).ndim == 2:
            audio = np.asarray(audio).mean(axis=1)
        bw.append(min(raw_bandwidth_cutoff_hz(audio, int(sr)), cap))
    df["processed_bandwidth_hz"] = bw
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df[["sample_id", "label", "generator", "split", "processed_bandwidth_hz"]].to_csv(args.output, index=False)

    print(f"Analysed band: 0 to {cap:.0f} Hz")
    print(df.groupby("generator")["processed_bandwidth_hz"].describe(percentiles=[0.05, 0.5]).round(0).to_string())
    train, test = df[df["split"].eq("train")], df[df["split"].eq("test")]
    t, sign = best_threshold(train["processed_bandwidth_hz"].to_numpy(), train["label"].to_numpy())
    pred = (sign * test["processed_bandwidth_hz"].to_numpy()) >= (sign * t)
    print(f"\nBandwidth-only on processed audio, test accuracy: {accuracy_score(test['label'], pred):.3f}")
    if test["label"].nunique() == 2 and test["processed_bandwidth_hz"].nunique() > 1:
        auc = roc_auc_score(test["label"], test["processed_bandwidth_hz"])
        print(f"ROC-AUC (direction-free): {max(auc, 1 - auc):.3f}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
