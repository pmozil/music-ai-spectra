#!/usr/bin/env python3
"""Strict leave-one-AI-generator-out evaluation using clip-level SVD features.

For each held-out generator, its AI clips and matched human counterparts form
the test set. All clips sharing any held-out caption group are removed from
training/validation, preventing caption-level leakage.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

import train_evaluate as evaluation
from music_svd_detection.parallel import limit_worker_threads

_WORKER_CONTEXT: tuple[pd.DataFrame, list[str], int, int] | None = None


def _evaluate_generator(
    generator: str, df: pd.DataFrame, features: list[str], seed: int, bootstrap_reps: int,
) -> tuple[dict, pd.DataFrame]:
    ai_rows = df[df["label"].astype(int).eq(1) & df["generator"].astype(str).eq(generator)]
    pair_ids = ai_rows["ytid"].astype(str).unique()
    if not len(pair_ids):
        raise ValueError(f"No AI clips for generator {generator}")
    test = df[df["ytid"].astype(str).isin(pair_ids)].copy()
    # Require complete human/AI pairs for the held-out generator.
    pair_count = test.groupby("ytid")["label"].agg(["count", "sum"])
    valid_ids = pair_count.index[(pair_count["count"] == 2) & (pair_count["sum"] == 1)]
    test = test[test["ytid"].isin(valid_ids)].copy()
    if test.empty:
        raise ValueError(f"No complete held-out pairs for {generator}")

    group_col = "caption_group" if "caption_group" in df else "ytid"
    held_groups = set(test[group_col].astype(str))
    remainder = df[~df[group_col].astype(str).isin(held_groups)].copy()
    trainval = remainder[remainder["split"].isin(["train", "validation"])].copy()
    trainval = trainval[trainval["label"].isin([0, 1])]
    if trainval.empty or trainval["label"].nunique() != 2:
        raise ValueError(f"Insufficient non-held-out training data for {generator}")
    # Preserve the existing train/validation partition. The test set is formed
    # solely from complete matched pairs of the unseen generator.
    test["split"] = "test"
    dataset = pd.concat([trainval, test], ignore_index=True)
    te, prob, pred, threshold = evaluation.fit_predict_logistic(dataset, features, seed)
    y = te["label"].astype(int).to_numpy()
    groups = te[group_col].astype(str).to_numpy()
    ci_lo, ci_hi = evaluation.paired_bootstrap_balanced_accuracy(
        y, pred, groups, n_bootstrap=bootstrap_reps, seed=seed,
    )
    pair_acc, n_pairs = evaluation._paired_ranking_accuracy(te, prob)
    row = {
        "held_out_generator": generator,
        "n_train": int((dataset["split"] == "train").sum()),
        "n_validation": int((dataset["split"] == "validation").sum()),
        "n_test_clips": int(len(te)), "n_test_pairs": n_pairs,
        "accuracy": float(np.mean(y == pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "balanced_accuracy_ci95_low": ci_lo,
        "balanced_accuracy_ci95_high": ci_hi,
        "roc_auc": float(roc_auc_score(y, prob)) if np.unique(y).size == 2 else float("nan"),
        "paired_ranking_accuracy": pair_acc,
        "validation_threshold": threshold,
        "excluded_caption_groups": len(held_groups),
    }
    cols = [c for c in ("sample_id", "ytid", group_col, "generator", "label") if c in te]
    predictions = te[cols].copy()
    predictions["held_out_generator"] = generator
    predictions["probability_ai"] = prob
    predictions["prediction"] = pred
    return row, predictions


def _init_worker(df: pd.DataFrame, features: list[str], seed: int, reps: int) -> None:
    global _WORKER_CONTEXT
    limit_worker_threads()
    _WORKER_CONTEXT = (df, features, seed, reps)


def _worker(generator: str) -> tuple[dict, pd.DataFrame]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker not initialized")
    df, features, seed, reps = _WORKER_CONTEXT
    return _evaluate_generator(generator, df, features, seed, reps)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/leave_one_generator_out"))
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261127)
    args = ap.parse_args()
    if args.workers < 1 or args.bootstrap_reps < 1:
        ap.error("--workers and --bootstrap-reps must be positive")

    df = pd.read_csv(args.features)
    if "feature_status" in df:
        df = df[df["feature_status"].eq("ok")].copy()
    if "ytid" not in df or "generator" not in df or "split" not in df:
        raise ValueError("Feature CSV must contain ytid, generator, and split columns")
    pair_counts = df.groupby("ytid")["label"].agg(["count", "sum"])
    valid = pair_counts.index[(pair_counts["count"] == 2) & (pair_counts["sum"] == 1)]
    df = df[df["ytid"].isin(valid)].copy()
    vector_features = sorted(
        c for c in df.columns
        if c.startswith(("u_mode", "v_mode", "u_subspace", "v_subspace"))
        and pd.api.types.is_numeric_dtype(df[c])
    )
    features = evaluation.VALUE_FEATURES + vector_features + evaluation.ANGLE_FEATURES
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing SVD feature columns: {sorted(missing)}")
    generators = sorted(df.loc[df["label"].astype(int).eq(1), "generator"].dropna().astype(str).unique())
    if not generators:
        raise ValueError("No AI generator labels found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], []
    with tqdm(total=len(generators), desc=f"Leave-one-generator-out ({args.workers} workers)") as bar:
        if args.workers == 1:
            for generator in generators:
                row, pred = _evaluate_generator(generator, df, features, args.seed, args.bootstrap_reps)
                rows.append(row); predictions.append(pred); bar.update(1)
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init_worker,
                initargs=(df, features, args.seed, args.bootstrap_reps),
            ) as pool:
                for row, pred in pool.map(_worker, generators, chunksize=1):
                    rows.append(row); predictions.append(pred); bar.update(1)
    result = pd.DataFrame(rows).sort_values("held_out_generator")
    result.to_csv(args.output_dir / "leave_one_generator_out.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        args.output_dir / "leave_one_generator_out_predictions.csv", index=False,
    )
    print(result.to_string(index=False))
    print(f"Saved outputs to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
