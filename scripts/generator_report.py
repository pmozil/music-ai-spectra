#!/usr/bin/env python3
"""Per-generator accuracy/AUC and the low-bandwidth robustness check, from saved features.

No audio is read and no SVD is recomputed: the script refits the same logistic
regressions as compare_exact_randomized.py and train_evaluate.py from the feature
CSVs those scripts already wrote, so it takes seconds.

Per-generator: the models are trained on all generators as usual. Each test pair is
(one human clip, one AI clip from generator g) with the same caption, so the
"generator g" subset is the AI clips from g plus their paired human clips.

Robustness (--min-raw-bandwidth-hz 6000): clips whose RAW bandwidth is below the
threshold are removed (both sides of their pair), and every model is retrained
and retested on what is left.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_exact_randomized as compare  # noqa: E402
import train_evaluate as te  # noqa: E402
from check_processed_bandwidth import best_threshold  # noqa: E402

HUMAN = "human_source"


def complete_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only ytids that still have exactly one human and one AI clip."""
    c = df.groupby("ytid")["label"].agg(["count", "sum"])
    return df[df["ytid"].isin(c.index[(c["count"] == 2) & (c["sum"] == 1)])].copy()


def apply_bandwidth_filter(df: pd.DataFrame, bw: pd.Series, min_hz: float) -> pd.DataFrame:
    if not min_hz:
        return complete_pairs(df)
    keep = df["sample_id"].map(bw)
    missing = int(keep.isna().sum())
    if missing:
        raise ValueError(f"{missing} clips have no raw_bandwidth_cutoff_hz; pass --manifest with that column")
    return complete_pairs(df[keep >= min_hz])


def scores(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    two = np.unique(y).size == 2
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if two else float("nan"),
        "roc_auc": float(roc_auc_score(y, prob)) if two else float("nan"),
        "ai_detected": float(np.mean(pred[y == 1])) if (y == 1).any() else float("nan"),
        "human_correct": float(np.mean(pred[y == 0] == 0)) if (y == 0).any() else float("nan"),
    }


def breakdown(model: str, test: pd.DataFrame, prob: np.ndarray, pred: np.ndarray):
    """Overall row + one row per AI generator (its AI clips + their paired human clips)."""
    test = test.assign(_prob=np.asarray(prob, float), _pred=np.asarray(pred, int))
    y = test["label"].astype(int).to_numpy()
    overall = {"model": model, "n_test": len(test), **scores(y, test["_prob"].to_numpy(), test["_pred"].to_numpy())}
    rows = []
    ai = test[test["label"].eq(1)]
    for gen in sorted(ai["generator"].unique()):
        ids = ai.loc[ai["generator"].eq(gen), "ytid"]
        sub = test[test["ytid"].isin(ids)]
        ys = sub["label"].astype(int).to_numpy()
        sc = scores(ys, sub["_prob"].to_numpy(), sub["_pred"].to_numpy())
        # normal-approximation 95% half-width for accuracy (about 0.04 at 100 pairs)
        sc["accuracy_ci95_halfwidth"] = float(1.96 * np.sqrt(sc["accuracy"] * (1 - sc["accuracy"]) / len(sub)))
        rows.append({"model": model, "generator": gen, "n_test_pairs": int(len(ids)), **sc})
    return overall, rows


def threshold_model(df: pd.DataFrame, col: str):
    """1-feature threshold classifier fitted on train (like check_processed_bandwidth.py)."""
    tr, tst = df[df["split"].eq("train")], df[df["split"].eq("test")]
    t, sign = best_threshold(tr[col].to_numpy(), tr["label"].to_numpy())
    pred = ((sign * tst[col].to_numpy()) >= (sign * t)).astype(int)
    return tst, sign * tst[col].to_numpy(), pred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare-dir", type=Path, help="output dir of compare_exact_randomized.py or one rank_K dir of rank_sweep.py")
    ap.add_argument("--train-eval-dir", type=Path, help="output dir of train_evaluate.py (needs features_with_shared_basis.csv)")
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.csv"),
                    help="used only if --train-eval-dir has no features_with_shared_basis.csv (e.g. --no-shared-basis)")
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_harmonized.csv"),
                    help="source of raw_bandwidth_cutoff_hz for the --min-raw-bandwidth-hz filter")
    ap.add_argument("--processed-bandwidth", type=Path, default=Path("artifacts/processed_bandwidth.csv"),
                    help="output of check_processed_bandwidth.py; adds that baseline if the file exists")
    ap.add_argument("--min-raw-bandwidth-hz", type=float, default=0.0)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--compare-seed", type=int, default=20261112)
    ap.add_argument("--train-eval-seed", type=int, default=20261026)
    args = ap.parse_args()
    if not args.compare_dir and not args.train_eval_dir:
        ap.error("give --compare-dir and/or --train-eval-dir")

    bw = pd.Series(dtype=float)
    if args.min_raw_bandwidth_hz:
        man = pd.read_csv(args.manifest, usecols=["sample_id", "raw_bandwidth_cutoff_hz"])
        bw = man.set_index("sample_id")["raw_bandwidth_cutoff_hz"]

    overall, per_gen, pairs_info = [], [], {}

    if args.compare_dir:
        wide = pd.read_csv(args.compare_dir / "features_exact_vs_randomized.csv")
        wide = apply_bandwidth_filter(wide, bw, args.min_raw_bandwidth_hz)
        pairs_info["compare"] = wide["ytid"].nunique()
        for method in ("exact", "randomized"):
            test, prob = compare.fit_predict(wide, method, args.compare_seed)
            o, r = breakdown(f"compare_{method}_6feat", test, prob, (prob >= 0.5).astype(int))
            overall.append(o)
            per_gen += r

    if args.train_eval_dir:
        path = args.train_eval_dir / "features_with_shared_basis.csv"
        df = pd.read_csv(path if path.exists() else args.features)
        if "feature_status" in df:
            df = df[df["feature_status"].eq("ok")]
        if args.min_raw_bandwidth_hz:
            bw = bw.combine_first(df.set_index("sample_id")["raw_bandwidth_cutoff_hz"])
        df = apply_bandwidth_filter(df, bw, args.min_raw_bandwidth_hz)
        pairs_info["train_evaluate"] = df["ytid"].nunique()
        local = [c for c in te.LOCAL_FEATURES if c in df.columns]
        models = [("raw_bandwidth_only (raw, ignore)", te.BASELINE_FEATURES),
                  ("local_svd_15feat", local)]
        if "shared_basis_energy_fraction" in df.columns:
            models.append(("local_plus_shared_basis_17feat",
                           local + ["shared_basis_energy_fraction", "shared_basis_relative_residual"]))
        for name, cols in models:
            test, prob, pred, _ = te.fit_predict(df, cols, args.train_eval_seed)
            o, r = breakdown(name, test, prob, pred)
            overall.append(o)
            per_gen += r

        if args.processed_bandwidth.exists():
            pb = pd.read_csv(args.processed_bandwidth)[["sample_id", "processed_bandwidth_hz"]]
            sub = df.merge(pb, on="sample_id", how="inner")
            sub = complete_pairs(sub)
            test, score, pred = threshold_model(sub, "processed_bandwidth_hz")
            o, r = breakdown("processed_bandwidth_only (want ~0.5)", test, score, pred)
            overall.append(o)
            per_gen += r

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ov, pg = pd.DataFrame(overall), pd.DataFrame(per_gen)
    ov.insert(1, "min_raw_bandwidth_hz", args.min_raw_bandwidth_hz)
    pg.insert(1, "min_raw_bandwidth_hz", args.min_raw_bandwidth_hz)
    ov.to_csv(args.output_dir / "overall_metrics.csv", index=False)
    pg.to_csv(args.output_dir / "per_generator_metrics.csv", index=False)

    pd.set_option("display.width", 200)
    filt = f"clips with raw bandwidth >= {args.min_raw_bandwidth_hz:.0f} Hz" if args.min_raw_bandwidth_hz else "all clips"
    print(f"Subset: {filt}; pairs kept: " + ", ".join(f"{k} {v}" for k, v in pairs_info.items()))
    print("\nOverall test metrics")
    print(ov.drop(columns="min_raw_bandwidth_hz").round(3).to_string(index=False))
    for metric in ("accuracy", "roc_auc"):
        print(f"\nPer-generator {metric} (AI clips of that generator + their paired human clips)")
        tab = pg.pivot(index="model", columns="generator", values=metric).reindex(ov["model"].unique())
        tab["all"] = ov.set_index("model")[metric]
        print(tab.round(3).to_string())
    print("\nTest pairs per generator:")
    print(pg.drop_duplicates("generator").set_index("generator")["n_test_pairs"].to_string())
    print(f"\nSaved: {args.output_dir}/overall_metrics.csv, per_generator_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
