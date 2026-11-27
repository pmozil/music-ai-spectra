#!/usr/bin/env python3
"""Compare within-class and between-class stacked-SVD subspace distances.

Uses training clips only. Each repetition splits caption groups into two disjoint
halves, builds four explicit stacked matrices (human A/B, AI A/B), and compares
their randomized-SVD left subspaces using principal angles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import train_evaluate as evaluation
from music_svd_detection.spectral import SpectralConfig


def _summary_row(repeat: int, comparison: str, b0: np.ndarray, b1: np.ndarray) -> tuple[dict, list[dict]]:
    summary, _, angles = evaluation.compare_class_bases(b0, b1)
    row = {
        "repeat": repeat,
        "comparison": comparison,
        "mean_principal_angle_deg": summary["mean_principal_angle_deg"],
        "median_principal_angle_deg": summary["median_principal_angle_deg"],
        "mean_squared_principal_cosine": summary["mean_squared_principal_cosine"],
        "mean_matched_abs_cosine": summary["mean_matched_abs_cosine"],
    }
    return row, [{"repeat": repeat, "comparison": comparison, **angle} for angle in angles]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_harmonized.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/subspace_stability"))
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--oversample", type=int, default=None)
    ap.add_argument("--power-iters", type=int, default=None)
    ap.add_argument("--stack-block-columns", type=int, default=65536)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunksize", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--keep-stacked-matrices", action="store_true")
    args = ap.parse_args()
    if min(args.repeats, args.workers, args.chunksize) < 1:
        ap.error("--repeats, --workers and --chunksize must be positive")

    cfgd = json.loads(args.config.read_text())
    cfg = evaluation._config_from_dict(cfgd)
    rank = int(args.rank or cfgd.get("class_space_rank", 16))
    oversample = int(args.oversample if args.oversample is not None else cfgd.get("class_space_oversample", 8))
    power_iters = int(args.power_iters if args.power_iters is not None else cfgd.get("class_space_power_iterations", 2))
    df = pd.read_csv(args.manifest)
    if "processing_status" in df:
        df = df[df["processing_status"].eq("ok")].copy()
    train = df[df["split"].eq("train")].copy()
    group_col = "caption_group" if "caption_group" in train else "ytid"
    train[group_col] = train[group_col].fillna(train["ytid"]).astype(str)
    # Retain only complete matched pairs and whole caption groups.
    pair_counts = train.groupby("ytid")["label"].agg(["count", "sum"])
    good_ids = pair_counts.index[(pair_counts["count"] == 2) & (pair_counts["sum"] == 1)]
    train = train[train["ytid"].isin(good_ids)].copy()
    groups = np.asarray(sorted(train[group_col].unique()))
    if len(groups) < 4:
        raise ValueError("Need at least four caption groups in training data")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict] = []
    angle_rows: list[dict] = []
    for repeat in range(1, args.repeats + 1):
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        cut = len(shuffled) // 2
        group_a, group_b = set(shuffled[:cut]), set(shuffled[cut:])
        parts = {
            "a": train[train[group_col].isin(group_a)],
            "b": train[train[group_col].isin(group_b)],
        }
        bases: dict[tuple[int, str], np.ndarray] = {}
        for part_name, subset in parts.items():
            for label, class_name in ((0, "human"), (1, "ai")):
                paths = subset.loc[subset["label"].astype(int).eq(label), "processed_path"].astype(str).tolist()
                if len(paths) <= rank:
                    raise ValueError(f"Repeat {repeat} {class_name}/{part_name} has only {len(paths)} clips")
                matrix_path = args.output_dir / f".stability_r{repeat}_{class_name}_{part_name}.f32"
                basis, _ = evaluation._fit_stacked_basis(
                    paths, cfg, matrix_path, rank, oversample, power_iters,
                    args.stack_block_columns, args.seed + 10000 * repeat + label * 10 + (part_name == "b"),
                    args.workers, args.chunksize, args.keep_stacked_matrices,
                )
                bases[(label, part_name)] = basis
                print(f"repeat {repeat}: {class_name} {part_name}: {len(paths)} clips, basis {basis.shape}")

        comparisons = [
            ("within_human", bases[(0, "a")], bases[(0, "b")]),
            ("within_ai", bases[(1, "a")], bases[(1, "b")]),
            ("between_human_ai", bases[(0, "a")], bases[(1, "a")]),
            ("between_human_ai_cross_split", bases[(0, "b")], bases[(1, "b")]),
        ]
        for name, b0, b1 in comparisons:
            row, angle_data = _summary_row(repeat, name, b0, b1)
            summary_rows.append(row)
            angle_rows.extend(angle_data)

    details = pd.DataFrame(summary_rows)
    details.to_csv(args.output_dir / "subspace_stability_repeats.csv", index=False)
    pd.DataFrame(angle_rows).to_csv(args.output_dir / "subspace_stability_angles.csv", index=False)
    aggregate = details.groupby("comparison").agg(
        repeats=("repeat", "nunique"),
        mean_angle_deg=("mean_principal_angle_deg", "mean"),
        std_angle_deg=("mean_principal_angle_deg", "std"),
        median_angle_deg=("median_principal_angle_deg", "median"),
        mean_squared_principal_cosine=("mean_squared_principal_cosine", "mean"),
        mean_matched_abs_cosine=("mean_matched_abs_cosine", "mean"),
    ).reset_index()
    aggregate.to_csv(args.output_dir / "subspace_stability_summary.csv", index=False)
    print(aggregate.to_string(index=False))
    print(f"Saved outputs to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
