#!/usr/bin/env python3
"""Extract exact per-clip SVD and sliding-window principal-angle features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from music_svd_detection.spectral import (
    SpectralConfig,
    log_magnitude_spectrogram,
    principal_angle_features,
    singular_value_features,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifest_harmonized.csv")
    )
    ap.add_argument("--output", type=Path, default=Path("data/processed/features.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="First N complete human/synthetic pairs (0 = all)",
    )
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    spec_cfg = SpectralConfig(
        sample_rate=int(cfg["sample_rate"]),
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
        log_epsilon=float(cfg["log_epsilon"]),
        local_top_k=int(cfg["local_top_k"]),
        subspace_window_frames=int(cfg["subspace_window_frames"]),
        subspace_step_frames=int(cfg["subspace_step_frames"]),
    )
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[
        manifest["processing_status"].eq("ok") & manifest["processed_path"].notna()
    ].copy()
    # Preserve 1:1 class balance after preprocessing failures by dropping any
    # incomplete human/generated pair, rather than keeping only its surviving side.
    if not manifest.empty and {"ytid", "label"}.issubset(manifest.columns):
        counts = manifest.groupby("ytid")["label"].agg(["count", "sum"])
        good_ids = counts.index[(counts["count"] == 2) & (counts["sum"] == 1)]
        manifest = manifest[manifest["ytid"].isin(good_ids)]
    if args.limit > 0:
        # Limit by source ID (a human+synthetic pair), not by CSV row.
        chosen_ids = manifest["ytid"].drop_duplicates().head(args.limit)
        manifest = manifest[manifest["ytid"].isin(chosen_ids)]
    rows = []
    for _, item in tqdm(
        manifest.iterrows(), total=len(manifest), desc="Extracting features"
    ):
        row = item.to_dict()
        try:
            audio, sr = sf.read(
                str(row["processed_path"]), dtype="float32", always_2d=False
            )
            if np.asarray(audio).ndim == 2:
                audio = np.asarray(audio).mean(axis=1)
            if int(sr) != spec_cfg.sample_rate:
                raise ValueError(f"expected {spec_cfg.sample_rate} Hz, got {sr}")
            logmag = log_magnitude_spectrogram(audio, int(sr), spec_cfg)
            row.update(singular_value_features(logmag, top_k=spec_cfg.local_top_k))
            row.update(
                principal_angle_features(
                    logmag,
                    top_k=spec_cfg.local_top_k,
                    window_frames=spec_cfg.subspace_window_frames,
                    step_frames=spec_cfg.subspace_step_frames,
                )
            )
            row["spectrogram_frequency_bins"] = float(logmag.shape[0])
            row["spectrogram_frames"] = float(logmag.shape[1])
            row["feature_status"] = "ok"
        except Exception as exc:
            row["feature_status"] = "failed"
            row["feature_error"] = f"{type(exc).__name__}: {exc}"[:1000]
        rows.append(row)
        if len(rows) % 25 == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(args.output, index=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(
        f"Feature rows: {len(rows)}; status: {pd.DataFrame(rows).get('feature_status', pd.Series(dtype=str)).value_counts().to_dict()}"
    )
    print(f"Features: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
