#!/usr/bin/env python3
"""Compute one joint SVD per caption-matched human/AI pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from music_svd_detection.spectral import (
    SpectralConfig,
    joint_pair_svd_features,
    log_magnitude_spectrogram,
)


def read_logmag(path: str, cfg: SpectralConfig):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)
    return log_magnitude_spectrogram(audio, int(sr), cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_harmonized.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--output", type=Path, default=Path("data/processed/pairwise_svd_features.csv"))
    ap.add_argument("--pair-id", default="ytid", help="Column identifying matched caption pairs")
    ap.add_argument("--rank", type=int, default=8, help="Number of leading joint modes to describe")
    ap.add_argument("--limit", type=int, default=0, help="Maximum number of pairs (0 = all)")
    args = ap.parse_args()

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

    pairs = []
    for pair_id, group in df.groupby(args.pair_id, sort=False):
        labels = group["label"].astype(int).tolist()
        if len(group) == 2 and sorted(labels) == [0, 1]:
            pairs.append((pair_id, group.sort_values("label")))
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise ValueError("No complete pairs (exactly one label 0 and one label 1) found")

    rows = []
    for pair_id, pair in tqdm(pairs, desc="Pairwise joint SVD"):
        h = pair[pair["label"].astype(int).eq(0)].iloc[0]
        a = pair[pair["label"].astype(int).eq(1)].iloc[0]
        try:
            lh = read_logmag(str(h["processed_path"]), cfg)
            la = read_logmag(str(a["processed_path"]), cfg)
            hf, af = joint_pair_svd_features(lh, la, top_k=args.rank)
            for item, features, side in ((h, hf, "human"), (a, af, "ai")):
                row = {
                    "sample_id": item["sample_id"], args.pair_id: pair_id,
                    "split": item.get("split", ""), "pair_side": side,
                    "pair_svd_status": "ok", **features,
                }
                rows.append(row)
        except Exception as exc:
            for item, side in ((h, "human"), (a, "ai")):
                rows.append({
                    "sample_id": item["sample_id"], args.pair_id: pair_id,
                    "split": item.get("split", ""), "pair_side": side,
                    "pair_svd_status": "failed",
                    "pair_svd_error": f"{type(exc).__name__}: {exc}"[:1000],
                })
        if len(rows) % 50 == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.output, index=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    print(f"Pair features: {len(result)} rows from {len(pairs)} pairs")
    print(f"Status: {result['pair_svd_status'].value_counts().to_dict()}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
