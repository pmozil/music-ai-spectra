#!/usr/bin/env python3
"""Evaluate local SVD features and KNN distances to human/AI stacked spaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from music_svd_detection.shared_basis import (
    randomized_svd_matrix,
    shared_basis_clip_features,
    write_stacked_matrix,
)
from music_svd_detection.spectral import SpectralConfig, log_magnitude_spectrogram

LOCAL_FEATURES = [
    "effective_rank", "sigma1_over_fro", "topk_energy_fraction", "spectral_entropy_bits",
    "principal_angle_mean_deg", "principal_angle_std_deg", "subspace_stability",
] + [f"sigma_ratio_{i}" for i in range(1, 9)]
BASELINE_FEATURES = ["raw_bandwidth_cutoff_hz"]
SPACE_FEATURES = [
    "human_space_energy_fraction", "human_space_relative_residual",
    "ai_space_energy_fraction", "ai_space_relative_residual",
]


def read_logmag(path: str, cfg: SpectralConfig) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if np.asarray(audio).ndim == 2:
        audio = np.asarray(audio).mean(axis=1)
    return log_magnitude_spectrogram(audio, int(sr), cfg)


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scores = [
        (balanced_accuracy_score(y_true, probabilities >= t), abs(float(t) - 0.5), float(t))
        for t in candidates
    ]
    return max(scores, key=lambda x: (x[0], -x[1]))[2]


def paired_bootstrap_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray | None,
    n_bootstrap: int = 2000,
    seed: int = 20261127,
) -> tuple[float, float]:
    """Percentile 95% CI, resampling matched caption pairs when possible."""
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if y.shape != pred.shape or y.size == 0:
        raise ValueError("y_true and y_pred must be non-empty vectors of equal length")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("balanced accuracy bootstrap requires both binary classes")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    indices_by_group = None
    if groups is not None and len(groups) == len(y):
        g = np.asarray(groups).astype(str)
        unique = np.unique(g)
        candidates = [np.flatnonzero(g == key) for key in unique]
        if all(np.unique(y[idx]).size == 2 for idx in candidates):
            indices_by_group = candidates
    scores = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        if indices_by_group is not None:
            selected = rng.integers(0, len(indices_by_group), size=len(indices_by_group))
            idx = np.concatenate([indices_by_group[i] for i in selected])
        else:
            # Stratified fallback preserves both classes if pair identifiers are unavailable.
            pieces = [rng.choice(np.flatnonzero(y == cls), size=np.sum(y == cls), replace=True)
                      for cls in (0, 1)]
            idx = np.concatenate(pieces)
        scores[b] = balanced_accuracy_score(y[idx], pred[idx])
    lo, hi = np.quantile(scores, [0.025, 0.975])
    return float(lo), float(hi)


def fit_predict_logistic(
    df: pd.DataFrame, features: list[str], seed: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    tr, va, te = (df[df["split"].eq(s)].copy() for s in ("train", "validation", "test"))
    if tr.empty or te.empty:
        raise ValueError("Need non-empty train and test partitions")
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
    )
    model.fit(tr[features].replace([np.inf, -np.inf], np.nan), tr["label"].astype(int))
    threshold = 0.5
    if not va.empty and va["label"].nunique() == 2:
        pval = model.predict_proba(va[features].replace([np.inf, -np.inf], np.nan))[:, 1]
        threshold = choose_threshold(va["label"].astype(int).to_numpy(), pval)
    prob = model.predict_proba(te[features].replace([np.inf, -np.inf], np.nan))[:, 1]
    return te, prob, (prob >= threshold).astype(int), float(threshold)


def fit_predict_knn(
    df: pd.DataFrame, features: list[str], seed: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int]:
    """Tune KNN neighbourhood size on validation; refit only on the training split."""
    del seed  # KNN is deterministic once the split and data order are fixed.
    tr, va, te = (df[df["split"].eq(s)].copy() for s in ("train", "validation", "test"))
    if tr.empty or te.empty:
        raise ValueError("Need non-empty train and test partitions")
    xtr = tr[features].replace([np.inf, -np.inf], np.nan)
    ytr = tr["label"].astype(int).to_numpy()
    candidates = [k for k in (1, 3, 5, 7, 11, 15) if k <= len(tr)]
    best_k, best_score = min(candidates), -np.inf
    if not va.empty and va["label"].nunique() == 2:
        xva = va[features].replace([np.inf, -np.inf], np.nan)
        yva = va["label"].astype(int).to_numpy()
        for k in candidates:
            candidate = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean"),
            )
            candidate.fit(xtr, ytr)
            pva = candidate.predict_proba(xva)[:, list(candidate[-1].classes_).index(1)]
            score = balanced_accuracy_score(yva, pva >= 0.5)
            if score > best_score or (score == best_score and k < best_k):
                best_k, best_score = k, float(score)
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        KNeighborsClassifier(n_neighbors=best_k, weights="distance", metric="euclidean"),
    )
    model.fit(xtr, ytr)
    xte = te[features].replace([np.inf, -np.inf], np.nan)
    classes = list(model[-1].classes_)
    prob = model.predict_proba(xte)[:, classes.index(1)]
    return te, prob, (prob >= 0.5).astype(int), int(best_k)


def metrics_row(
    df: pd.DataFrame,
    features: list[str],
    name: str,
    seed: int,
    estimator: str = "logistic",
) -> dict[str, object]:
    if estimator == "knn":
        te, prob, pred, setting = fit_predict_knn(df, features, seed)
        setting_name = "selected_k"
        threshold = np.nan
    else:
        te, prob, pred, setting = fit_predict_logistic(df, features, seed)
        setting_name = "threshold_from_validation"
        threshold = setting
    y = te["label"].astype(int).to_numpy()
    bootstrap_groups = (
        te["caption_group"].to_numpy()
        if "caption_group" in te
        else (te["ytid"].to_numpy() if "ytid" in te else None)
    )
    ci_lo, ci_hi = paired_bootstrap_balanced_accuracy(
        y, pred, bootstrap_groups, seed=seed
    )
    cm = confusion_matrix(y, pred, labels=[0, 1])
    row = {
        "model": name, "estimator": estimator, "n_features": len(features),
        "features": ",".join(features),
        "n_train": int((df["split"] == "train").sum()),
        "n_validation": int((df["split"] == "validation").sum()),
        "n_test": int(len(te)), setting_name: setting,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "balanced_accuracy_ci95_low": ci_lo, "balanced_accuracy_ci95_high": ci_hi,
        "roc_auc": float(roc_auc_score(y, prob)) if np.unique(y).size == 2 else float("nan"),
        "precision_ai": float(precision_score(y, pred, zero_division=0)),
        "recall_ai": float(recall_score(y, pred, zero_division=0)),
        "f1_ai": float(f1_score(y, pred, zero_division=0)),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    if estimator == "logistic":
        row["threshold_from_validation"] = threshold
    return row


def fit_class_space(
    df: pd.DataFrame,
    label: int,
    cfg: SpectralConfig,
    output_dir: Path,
    rank: int,
    oversample: int,
    power_iterations: int,
    block_columns: int,
    seed: int,
    keep_matrix: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build A_class=[L_1 ... L_n] on train clips and run randomized SVD on A_class."""
    paths = df.loc[df["split"].eq("train") & df["label"].astype(int).eq(label), "processed_path"].astype(str).tolist()
    if not paths:
        raise ValueError(f"No training clips found for class label {label}")

    def block_factory():
        for path in paths:
            yield read_logmag(path, cfg)

    class_name = "human" if label == 0 else "ai"
    matrix_path = output_dir / f".stacked_{class_name}.f32"
    matrix = write_stacked_matrix(block_factory, matrix_path)
    try:
        basis, singular_values = randomized_svd_matrix(
            matrix, rank=rank, oversample=oversample,
            n_power_iterations=power_iterations, seed=seed + label,
            block_columns=block_columns,
        )
    finally:
        del matrix
        if not keep_matrix:
            matrix_path.unlink(missing_ok=True)
    np.save(output_dir / f"{class_name}_space_basis.npy", basis)
    pd.DataFrame({"singular_value_approx": singular_values}).to_csv(
        output_dir / f"{class_name}_space_singular_values.csv", index=False
    )
    print(f"{class_name} stacked matrix: {len(paths)} clips; basis {basis.shape}")
    return basis, singular_values


def add_class_space_features(
    df: pd.DataFrame, cfg: SpectralConfig, human_basis: np.ndarray, ai_basis: np.ndarray
) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Projecting to class spaces"):
        lm = read_logmag(str(row["processed_path"]), cfg)
        h = shared_basis_clip_features(lm, human_basis)
        a = shared_basis_clip_features(lm, ai_basis)
        rows.append({
            "human_space_energy_fraction": h["shared_basis_energy_fraction"],
            "human_space_relative_residual": h["shared_basis_relative_residual"],
            "ai_space_energy_fraction": a["shared_basis_energy_fraction"],
            "ai_space_relative_residual": a["shared_basis_relative_residual"],
        })
    extra = pd.DataFrame(rows, index=df.index)
    return pd.concat([df.copy(), extra], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--pairwise-features", type=Path, default=None,
                    help="Optional output of scripts/pairwise_svd.py")
    ap.add_argument("--class-space-rank", type=int, default=None)
    ap.add_argument("--oversample", type=int, default=None)
    ap.add_argument("--power-iters", type=int, default=None)
    ap.add_argument("--stack-block-columns", type=int, default=65536)
    ap.add_argument("--keep-stacked-matrices", action="store_true")
    ap.add_argument("--seed", type=int, default=20261127)
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
    df = pd.read_csv(args.features)
    if "feature_status" in df:
        df = df[df["feature_status"].eq("ok")].copy()
    if "ytid" not in df:
        raise ValueError("features.csv must contain the caption-pair key 'ytid'")
    # Do not allow an orphaned half-pair into training/evaluation.
    pair_counts = df.groupby("ytid")["label"].agg(["count", "sum"])
    good_ids = pair_counts.index[(pair_counts["count"] == 2) & (pair_counts["sum"] == 1)]
    df = df[df["ytid"].isin(good_ids)].copy()
    required = {"split", "label", "processed_path", "ytid", "sample_id", *BASELINE_FEATURES, *LOCAL_FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in feature CSV: {sorted(missing)}")
    balance = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    if 0 not in balance.columns or 1 not in balance.columns or not (balance[0] == balance[1]).all():
        raise ValueError(f"Expected balanced classes in every split; got:\n{balance}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_space_rank = int(args.class_space_rank or cfgd.get("class_space_rank", 16))
    oversample = int(args.oversample if args.oversample is not None else cfgd.get("class_space_oversample", 8))
    power_iters = int(args.power_iters if args.power_iters is not None else cfgd.get("class_space_power_iterations", 2))
    if args.pairwise_features is not None:
        pair_df = pd.read_csv(args.pairwise_features)
        pair_df = pair_df[pair_df["pair_svd_status"].eq("ok")].copy()
        if pair_df["sample_id"].duplicated().any():
            raise ValueError("Pairwise feature file has duplicate sample_id values")
        pair_cols = [c for c in pair_df.columns if c.startswith("pair_") and c not in {"pair_svd_status", "pair_side", "pair_svd_error"}]
        df = df.merge(pair_df[["sample_id", *pair_cols]], on="sample_id", how="left", validate="one_to_one")
        matched = int(df[pair_cols].notna().any(axis=1).sum()) if pair_cols else 0
        print(f"Pairwise SVD features merged for {matched}/{len(df)} clips")

    # Existing baseline and local-feature results are retained, now with pair bootstrap CIs.
    result_rows = [
        metrics_row(df, BASELINE_FEATURES, "bandwidth_only_baseline", args.seed, "logistic"),
        metrics_row(df, LOCAL_FEATURES, "local_svd_vectors_and_angles", args.seed, "logistic"),
    ]

    human_basis, _ = fit_class_space(
        df, 0, cfg, args.output_dir, class_space_rank, oversample,
        power_iters, args.stack_block_columns, args.seed, args.keep_stacked_matrices,
    )
    ai_basis, _ = fit_class_space(
        df, 1, cfg, args.output_dir, class_space_rank, oversample,
        power_iters, args.stack_block_columns, args.seed, args.keep_stacked_matrices,
    )
    space_df = add_class_space_features(df, cfg, human_basis, ai_basis)
    space_df.to_csv(args.output_dir / "features_with_class_spaces.csv", index=False)
    result_rows.append(metrics_row(space_df, SPACE_FEATURES, "knn_human_vs_ai_class_spaces", args.seed, "knn"))

    vector_cols = [c for c in df.columns if c.startswith(("u_", "v_"))]
    pair_cols = [c for c in df.columns if c.startswith("pair_") and pd.api.types.is_numeric_dtype(df[c])]
    rich_features = SPACE_FEATURES + vector_cols + pair_cols
    if vector_cols or pair_cols:
        result_rows.append(metrics_row(
            space_df, rich_features, "knn_class_spaces_plus_singular_vectors_and_pairs", args.seed, "knn"
        ))

    result = pd.DataFrame(result_rows)
    result.to_csv(args.output_dir / "metrics.csv", index=False)
    experiment = {
        "seed": args.seed,
        "feature_csv": str(args.features),
        "pairwise_feature_csv": str(args.pairwise_features) if args.pairwise_features else None,
        "class_counts_by_split": balance.to_dict(),
        "class_space_rank": class_space_rank,
        "oversample": oversample,
        "power_iterations": power_iters,
        "class_spaces": "separate randomized SVDs of explicit training-only matrices A_human=[L_i] and A_ai=[L_i], materialized as temporary disk-backed matrices",
        "knn": "distance-weighted Euclidean KNN; k selected by validation balanced accuracy",
        "bootstrap": "percentile 95% CI of balanced accuracy; resample caption groups (all matched pairs sharing a caption together), 2000 replicates",
        "pairwise_svd": "one exact SVD of [L_human L_AI] per matched pair; side-specific right-vector block shares and joint-subspace projection residuals",
        "models": result_rows,
    }
    (args.output_dir / "experiment.json").write_text(json.dumps(experiment, indent=2, default=int))
    print(result[["model", "estimator", "n_test", "balanced_accuracy", "balanced_accuracy_ci95_low", "balanced_accuracy_ci95_high", "roc_auc"]].to_string(index=False))
    print(f"Results written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
