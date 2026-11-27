#!/usr/bin/env python3
"""Evaluate local SVD descriptors and cross-fitted human/AI stacked subspaces.

The class-space classifier uses out-of-fold projections for its training rows:
no clip contributes to the basis onto which that same clip is projected. The
validation and test rows are projected onto bases fitted on the complete training
partition only.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from music_svd_detection.parallel import limit_worker_threads
from music_svd_detection.shared_basis import (
    compare_class_bases, randomized_svd_matrix, shared_basis_clip_features,
    write_stacked_matrix,
)
from music_svd_detection.spectral import SpectralConfig, log_magnitude_spectrogram

VALUE_FEATURES = [
    "effective_rank", "sigma1_over_fro", "topk_energy_fraction", "spectral_entropy_bits",
] + [f"sigma_ratio_{i}" for i in range(1, 9)]
ANGLE_FEATURES = [
    "principal_angle_mean_deg", "principal_angle_std_deg", "subspace_stability",
]
BASELINE_FEATURES = ["raw_bandwidth_cutoff_hz"]
SPACE_FEATURES = [
    "human_space_energy_fraction", "human_space_relative_residual",
    "ai_space_energy_fraction", "ai_space_relative_residual",
]


def _config_from_dict(cfg: dict) -> SpectralConfig:
    return SpectralConfig(
        sample_rate=int(cfg["sample_rate"]), n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]), log_epsilon=float(cfg["log_epsilon"]),
        local_top_k=int(cfg["local_top_k"]),
        subspace_window_frames=int(cfg["subspace_window_frames"]),
        subspace_step_frames=int(cfg["subspace_step_frames"]),
        max_freq_hz=float(cfg.get("max_freq_hz", 0)),
    )


def read_logmag(path: str, cfg: SpectralConfig) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if np.asarray(audio).ndim == 2:
        audio = np.asarray(audio).mean(axis=1)
    return log_magnitude_spectrogram(audio, int(sr), cfg)


def _read_logmag_task(task: tuple[str, SpectralConfig]) -> np.ndarray:
    return read_logmag(task[0], task[1])


def _iter_logmags(paths: list[str], cfg: SpectralConfig, workers: int, chunksize: int):
    """Restartable iterator: each invocation streams the requested clip matrices."""
    tasks = ((path, cfg) for path in paths)
    if workers == 1:
        for task in tasks:
            yield _read_logmag_task(task)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=limit_worker_threads,
        ) as pool:
            yield from pool.map(_read_logmag_task, tasks, chunksize=chunksize)


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scores = [
        (balanced_accuracy_score(y_true, probabilities >= t), abs(float(t) - 0.5), float(t))
        for t in candidates
    ]
    return max(scores, key=lambda x: (x[0], -x[1]))[2]


def _caption_groups(groups: np.ndarray | None, n: int) -> list[np.ndarray] | None:
    if groups is None or len(groups) != n:
        return None
    g = np.asarray(groups).astype(str)
    unique = np.unique(g)
    members = [np.flatnonzero(g == key) for key in unique]
    return members if all(len(idx) for idx in members) else None


def paired_bootstrap_balanced_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray | None,
    n_bootstrap: int = 2000, seed: int = 20261127,
) -> tuple[float, float]:
    """Percentile 95% CI, resampling whole caption groups (and matched pairs)."""
    y, pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if y.shape != pred.shape or y.size == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("balanced accuracy bootstrap requires equal non-empty binary vectors")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    members = _caption_groups(groups, len(y))
    if members is not None and not all(np.unique(y[idx]).size == 2 for idx in members):
        members = None
    scores = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        if members is not None:
            selected = rng.integers(0, len(members), size=len(members))
            idx = np.concatenate([members[i] for i in selected])
        else:
            idx = np.concatenate([
                rng.choice(np.flatnonzero(y == cls), size=np.sum(y == cls), replace=True)
                for cls in (0, 1)
            ])
        scores[b] = balanced_accuracy_score(y[idx], pred[idx])
    lo, hi = np.quantile(scores, [0.025, 0.975])
    return float(lo), float(hi)


def paired_bootstrap_delta(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    groups: np.ndarray | None, n_bootstrap: int = 2000, seed: int = 20261127,
) -> tuple[float, float, float]:
    """CI for BA(model B) - BA(model A), using identical group resamples."""
    y = np.asarray(y_true, dtype=int)
    a, b = np.asarray(pred_a, dtype=int), np.asarray(pred_b, dtype=int)
    if y.shape != a.shape or y.shape != b.shape or y.size == 0:
        raise ValueError("delta bootstrap vectors must have equal non-zero length")
    point = float(balanced_accuracy_score(y, b) - balanced_accuracy_score(y, a))
    rng = np.random.default_rng(seed)
    members = _caption_groups(groups, len(y))
    if members is not None and not all(np.unique(y[idx]).size == 2 for idx in members):
        members = None
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        if members is not None:
            selected = rng.integers(0, len(members), size=len(members))
            idx = np.concatenate([members[j] for j in selected])
        else:
            idx = np.concatenate([
                rng.choice(np.flatnonzero(y == cls), size=np.sum(y == cls), replace=True)
                for cls in (0, 1)
            ])
        deltas[i] = balanced_accuracy_score(y[idx], b[idx]) - balanced_accuracy_score(y[idx], a[idx])
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return point, float(lo), float(hi)


def _split_frames(df: pd.DataFrame):
    tr, va, te = (df[df["split"].eq(s)].copy() for s in ("train", "validation", "test"))
    if tr.empty or te.empty or tr["label"].nunique() != 2:
        raise ValueError("Need non-empty train/test partitions and both classes in train")
    return tr, va, te


def fit_predict_logistic(
    df: pd.DataFrame, features: list[str], seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, float]:
    tr, va, te = _split_frames(df)
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )
    model.fit(tr[features].replace([np.inf, -np.inf], np.nan), tr["label"].astype(int))
    threshold = 0.5
    if not va.empty and va["label"].nunique() == 2:
        pval = model.predict_proba(va[features].replace([np.inf, -np.inf], np.nan))[:, 1]
        threshold = choose_threshold(va["label"].astype(int).to_numpy(), pval)
    prob = model.predict_proba(te[features].replace([np.inf, -np.inf], np.nan))[:, 1]
    return te, prob, (prob >= threshold).astype(int), float(threshold)


def fit_predict_knn(
    df: pd.DataFrame, features: list[str], seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int]:
    del seed
    tr, va, te = _split_frames(df)
    xtr, ytr = tr[features].replace([np.inf, -np.inf], np.nan), tr["label"].astype(int).to_numpy()
    candidates = [k for k in (1, 3, 5, 7, 11, 15) if k <= len(tr)]
    best_k, best_score = min(candidates), -np.inf
    if not va.empty and va["label"].nunique() == 2:
        xva, yva = va[features].replace([np.inf, -np.inf], np.nan), va["label"].astype(int).to_numpy()
        for k in candidates:
            candidate = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean"),
            )
            candidate.fit(xtr, ytr)
            classes = list(candidate[-1].classes_)
            pva = candidate.predict_proba(xva)[:, classes.index(1)]
            score = balanced_accuracy_score(yva, pva >= 0.5)
            if score > best_score or (score == best_score and k < best_k):
                best_k, best_score = k, float(score)
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        KNeighborsClassifier(n_neighbors=best_k, weights="distance", metric="euclidean"),
    )
    model.fit(xtr, ytr)
    classes = list(model[-1].classes_)
    prob = model.predict_proba(te[features].replace([np.inf, -np.inf], np.nan))[:, classes.index(1)]
    return te, prob, (prob >= 0.5).astype(int), int(best_k)


def fit_predict_gmm(
    df: pd.DataFrame, features: list[str], seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int]:
    """Class-conditional diagonal GMM; choose component count on validation only."""
    tr, va, te = _split_frames(df)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    xtr = scaler.fit_transform(imputer.fit_transform(tr[features].replace([np.inf, -np.inf], np.nan)))
    xva = scaler.transform(imputer.transform(va[features].replace([np.inf, -np.inf], np.nan))) if not va.empty else np.empty((0, xtr.shape[1]))
    xte = scaler.transform(imputer.transform(te[features].replace([np.inf, -np.inf], np.nan)))
    ytr = tr["label"].astype(int).to_numpy()
    candidates = [k for k in (1, 2, 4, 8) if k <= int(np.min(np.bincount(ytr)))]
    best_k, best_score, best_threshold = candidates[0], -np.inf, 0.5
    for k in candidates:
        models = {}
        for cls in (0, 1):
            gm = GaussianMixture(
                n_components=k, covariance_type="diag", reg_covar=1e-4,
                max_iter=300, n_init=2, random_state=seed + cls,
            )
            gm.fit(xtr[ytr == cls])
            models[cls] = gm
        if xva.shape[0] and va["label"].nunique() == 2:
            llr = models[1].score_samples(xva) - models[0].score_samples(xva)
            pva = expit(np.clip(llr, -40, 40))
            threshold = choose_threshold(va["label"].astype(int).to_numpy(), pva)
            score = balanced_accuracy_score(va["label"].astype(int), pva >= threshold)
        else:
            threshold, score = 0.5, 0.0
        if score > best_score:
            best_k, best_score, best_threshold = k, float(score), float(threshold)
    final_models = {}
    for cls in (0, 1):
        gm = GaussianMixture(
            n_components=best_k, covariance_type="diag", reg_covar=1e-4,
            max_iter=300, n_init=2, random_state=seed + cls,
        )
        gm.fit(xtr[ytr == cls])
        final_models[cls] = gm
    llr_test = final_models[1].score_samples(xte) - final_models[0].score_samples(xte)
    prob = expit(np.clip(llr_test, -40, 40))
    return te, prob, (prob >= best_threshold).astype(int), int(best_k)


def _paired_ranking_accuracy(te: pd.DataFrame, probabilities: np.ndarray) -> tuple[float, int]:
    if "ytid" not in te:
        return float("nan"), 0
    temp = pd.DataFrame({"ytid": te["ytid"].astype(str).to_numpy(),
                         "label": te["label"].astype(int).to_numpy(),
                         "prob": probabilities})
    pairs = temp.pivot_table(index="ytid", columns="label", values="prob", aggfunc="first")
    if 0 not in pairs or 1 not in pairs:
        return float("nan"), 0
    pairs = pairs.dropna(subset=[0, 1])
    if pairs.empty:
        return float("nan"), 0
    # Count ties as one half; both items in each caption-matched pair are compared.
    wins = (pairs[1] > pairs[0]).astype(float) + 0.5 * pairs[1].eq(pairs[0]).astype(float)
    return float(wins.mean()), int(len(pairs))


def metrics_row(
    df: pd.DataFrame, features: list[str], name: str, seed: int,
    estimator: str = "logistic", prediction_store: list[pd.DataFrame] | None = None,
    bootstrap_reps: int = 2000,
) -> dict[str, object]:
    if estimator == "knn":
        te, prob, pred, setting = fit_predict_knn(df, features, seed)
        setting_name, threshold = "selected_k", np.nan
    elif estimator == "gmm":
        te, prob, pred, setting = fit_predict_gmm(df, features, seed)
        setting_name, threshold = "selected_components", np.nan
    else:
        te, prob, pred, setting = fit_predict_logistic(df, features, seed)
        setting_name, threshold = "threshold_from_validation", setting
    y = te["label"].astype(int).to_numpy()
    groups = te["caption_group"].to_numpy() if "caption_group" in te else (te["ytid"].to_numpy() if "ytid" in te else None)
    ci_lo, ci_hi = paired_bootstrap_balanced_accuracy(y, pred, groups, n_bootstrap=bootstrap_reps, seed=seed)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    pair_acc, pair_count = _paired_ranking_accuracy(te, prob)
    row: dict[str, object] = {
        "model": name, "estimator": estimator, "n_features": len(features),
        "features": ",".join(features),
        "n_train": int((df["split"] == "train").sum()),
        "n_validation": int((df["split"] == "validation").sum()),
        "n_test": int(len(te)), setting_name: setting,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "balanced_accuracy_ci95_low": ci_lo, "balanced_accuracy_ci95_high": ci_hi,
        "roc_auc": float(roc_auc_score(y, prob)) if np.unique(y).size == 2 else float("nan"),
        "paired_ranking_accuracy": pair_acc, "n_test_pairs": pair_count,
        "precision_ai": float(precision_score(y, pred, zero_division=0)),
        "recall_ai": float(recall_score(y, pred, zero_division=0)),
        "f1_ai": float(f1_score(y, pred, zero_division=0)),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    if estimator == "logistic":
        row["threshold_from_validation"] = threshold
    if prediction_store is not None:
        cols = [c for c in ("sample_id", "ytid", "caption_group", "generator", "label", "split") if c in te]
        pred_df = te[cols].copy()
        pred_df["model"] = name
        pred_df["probability_ai"] = prob
        pred_df["prediction"] = pred
        prediction_store.append(pred_df)
    return row


def _fit_stacked_basis(
    paths: list[str], cfg: SpectralConfig, matrix_path: Path, rank: int,
    oversample: int, power_iterations: int, block_columns: int, seed: int,
    workers: int, chunksize: int, keep_matrix: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("Cannot build a stacked basis from zero clips")
    def block_factory():
        yield from _iter_logmags(paths, cfg, workers, chunksize)
    matrix = write_stacked_matrix(block_factory, matrix_path)
    try:
        basis, singular_values = randomized_svd_matrix(
            matrix, rank=rank, oversample=oversample,
            n_power_iterations=power_iterations, seed=seed,
            block_columns=block_columns,
        )
    finally:
        del matrix
        if not keep_matrix:
            matrix_path.unlink(missing_ok=True)
    return basis, singular_values


def fit_class_space(
    df: pd.DataFrame, label: int, cfg: SpectralConfig, output_dir: Path,
    rank: int, oversample: int, power_iterations: int, block_columns: int,
    seed: int, keep_matrix: bool, workers: int = 1, chunksize: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Build explicit A_class=[L_1 ... L_n] from the training clips."""
    paths = df.loc[df["split"].eq("train") & df["label"].astype(int).eq(label), "processed_path"].astype(str).tolist()
    if not paths:
        raise ValueError(f"No training clips found for class label {label}")
    class_name = "human" if label == 0 else "ai"
    matrix_path = output_dir / f".stacked_{class_name}.f32"
    basis, singular_values = _fit_stacked_basis(
        paths, cfg, matrix_path, rank, oversample, power_iterations,
        block_columns, seed + label, workers, chunksize, keep_matrix,
    )
    np.save(output_dir / f"{class_name}_space_basis.npy", basis)
    pd.DataFrame({"singular_value_approx": singular_values}).to_csv(
        output_dir / f"{class_name}_space_singular_values.csv", index=False
    )
    print(f"{class_name} stacked matrix: {len(paths)} clips; basis {basis.shape}")
    return basis, singular_values


_PROJECT_CONTEXT: tuple[SpectralConfig, np.ndarray, np.ndarray] | None = None


def _init_projection_worker(cfg: SpectralConfig, human_basis: np.ndarray, ai_basis: np.ndarray) -> None:
    global _PROJECT_CONTEXT
    limit_worker_threads()
    _PROJECT_CONTEXT = (cfg, human_basis, ai_basis)


def _project_one(record: dict, cfg: SpectralConfig, human_basis: np.ndarray, ai_basis: np.ndarray) -> dict:
    lm = read_logmag(str(record["processed_path"]), cfg)
    h = shared_basis_clip_features(lm, human_basis)
    a = shared_basis_clip_features(lm, ai_basis)
    return {
        "index": int(record["_row_index"]),
        "human_space_energy_fraction": h["shared_basis_energy_fraction"],
        "human_space_relative_residual": h["shared_basis_relative_residual"],
        "ai_space_energy_fraction": a["shared_basis_energy_fraction"],
        "ai_space_relative_residual": a["shared_basis_relative_residual"],
    }


def _project_worker(record: dict) -> dict:
    if _PROJECT_CONTEXT is None:
        raise RuntimeError("Projection worker was not initialized")
    cfg, human_basis, ai_basis = _PROJECT_CONTEXT
    return _project_one(record, cfg, human_basis, ai_basis)


def add_class_space_features(
    df: pd.DataFrame, cfg: SpectralConfig, human_basis: np.ndarray, ai_basis: np.ndarray,
    workers: int = 1, chunksize: int = 8, desc: str = "Projecting to class spaces",
) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        rec = row.to_dict()
        rec["_row_index"] = int(idx)
        records.append(rec)
    rows: list[dict] = []
    with tqdm(total=len(records), desc=f"{desc} ({workers} workers)") as bar:
        if workers == 1:
            for rec in records:
                rows.append(_project_one(rec, cfg, human_basis, ai_basis))
                bar.update(1)
        else:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_init_projection_worker,
                initargs=(cfg, human_basis, ai_basis),
            ) as pool:
                for item in pool.map(_project_worker, records, chunksize=chunksize):
                    rows.append(item)
                    bar.update(1)
    extra = pd.DataFrame(rows).set_index("index") if rows else pd.DataFrame(index=df.index)
    extra.index.name = None
    return pd.concat([df.copy(), extra.reindex(df.index)], axis=1)


def crossfit_class_space_features(
    df: pd.DataFrame, cfg: SpectralConfig, human_basis: np.ndarray, ai_basis: np.ndarray,
    output_dir: Path, rank: int, oversample: int, power_iterations: int,
    block_columns: int, seed: int, workers: int, chunksize: int,
    n_folds: int, keep_matrices: bool,
) -> pd.DataFrame:
    """OOF train projections + validation/test projections from full-train bases."""
    train_df = df[df["split"].eq("train")].copy()
    groups = train_df.get("caption_group", train_df["ytid"]).fillna(train_df["ytid"]).astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    if n_folds < 2 or n_folds > n_groups:
        raise ValueError(f"--crossfit-folds must be in [2, {n_groups}], got {n_folds}")
    splitter = GroupKFold(n_splits=n_folds)
    fold_features = []
    for fold, (fit_idx, held_idx) in enumerate(splitter.split(train_df, train_df["label"], groups), start=1):
        fold_fit = train_df.iloc[fit_idx].copy()
        fold_hold = train_df.iloc[held_idx].copy()
        if fold_fit["label"].nunique() != 2:
            raise ValueError(f"Cross-fit fold {fold} training portion does not contain both classes")
        print(f"Cross-fitting class spaces: fold {fold}/{n_folds}; fit={len(fold_fit)}, held-out={len(fold_hold)}")
        fold_bases = {}
        for label, class_name in ((0, "human"), (1, "ai")):
            paths = fold_fit.loc[fold_fit["label"].astype(int).eq(label), "processed_path"].astype(str).tolist()
            matrix_path = output_dir / f".crossfit_fold_{fold}_{class_name}.f32"
            basis, _ = _fit_stacked_basis(
                paths, cfg, matrix_path, rank, oversample, power_iterations,
                block_columns, seed + 1000 * fold + label, workers, chunksize, False,
            )
            fold_bases[label] = basis
        feat = add_class_space_features(
            fold_hold, cfg, fold_bases[0], fold_bases[1], workers, chunksize,
            desc=f"OOF projection fold {fold}",
        )
        feat["class_space_fold"] = fold
        fold_features.append(feat)

    oof = pd.concat(fold_features).sort_index()
    oof.to_csv(output_dir / "class_space_features_oof_train.csv", index=False)
    nontrain = df[~df["split"].eq("train")].copy()
    projected_nontrain = add_class_space_features(
        nontrain, cfg, human_basis, ai_basis, workers, chunksize,
        desc="Validation/test projection using full training bases",
    )
    projected_nontrain["class_space_fold"] = 0
    result = pd.concat([oof, projected_nontrain]).sort_index()
    return result


def _write_paired_model_differences(predictions: pd.DataFrame, out_path: Path, reps: int, seed: int) -> None:
    rows = []
    pairs = [
        ("local_svd_values_only", "local_svd_all_features"),
        ("knn_human_vs_ai_class_spaces_crossfit", "knn_class_spaces_plus_singular_vectors_and_pairs"),
    ]
    for model_a, model_b in pairs:
        a = predictions[predictions["model"].eq(model_a)].copy()
        b = predictions[predictions["model"].eq(model_b)].copy()
        if a.empty or b.empty:
            continue
        keys = ["sample_id"]
        merged = a[keys + ["label", "prediction", "caption_group"]].merge(
            b[keys + ["prediction"]], on="sample_id", suffixes=("_a", "_b"), validate="one_to_one"
        )
        if merged.empty:
            continue
        groups = merged["caption_group"].to_numpy() if "caption_group" in merged else None
        delta, lo, hi = paired_bootstrap_delta(
            merged["label"].to_numpy(), merged["prediction_a"].to_numpy(),
            merged["prediction_b"].to_numpy(), groups, reps, seed,
        )
        rows.append({"model_a": model_a, "model_b": model_b,
                     "delta_balanced_accuracy_b_minus_a": delta,
                     "ci95_low": lo, "ci95_high": hi,
                     "n_test": len(merged)})
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--pairwise-features", type=Path, default=None)
    ap.add_argument("--class-space-rank", type=int, default=None)
    ap.add_argument("--oversample", type=int, default=None)
    ap.add_argument("--power-iters", type=int, default=None)
    ap.add_argument("--stack-block-columns", type=int, default=65536)
    ap.add_argument("--crossfit-folds", type=int, default=3,
                    help="caption-group folds for out-of-fold training projections (0 disables cross-fitting)")
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--chunksize", type=int, default=4)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--keep-stacked-matrices", action="store_true")
    ap.add_argument("--seed", type=int, default=20261127)
    args = ap.parse_args()
    if args.workers < 1 or args.chunksize < 1 or args.bootstrap_reps < 1:
        ap.error("--workers, --chunksize and --bootstrap-reps must be positive")
    if args.crossfit_folds not in (0,) and args.crossfit_folds < 2:
        ap.error("--crossfit-folds must be 0 or at least 2")

    cfgd = json.loads(args.config.read_text())
    cfg = _config_from_dict(cfgd)
    df = pd.read_csv(args.features)
    if "feature_status" in df:
        df = df[df["feature_status"].eq("ok")].copy()
    if "ytid" not in df:
        raise ValueError("features.csv must contain the caption-pair key 'ytid'")
    pair_counts = df.groupby("ytid")["label"].agg(["count", "sum"])
    good_ids = pair_counts.index[(pair_counts["count"] == 2) & (pair_counts["sum"] == 1)]
    df = df[df["ytid"].isin(good_ids)].copy()
    if df["sample_id"].duplicated().any():
        raise ValueError("Feature CSV contains duplicate sample_id values")

    vector_features = sorted(
        c for c in df.columns
        if c.startswith(("u_mode", "v_mode", "u_subspace", "v_subspace"))
        and pd.api.types.is_numeric_dtype(df[c])
    )
    required = {"split", "label", "processed_path", "ytid", "sample_id", "caption_group",
                *BASELINE_FEATURES, *VALUE_FEATURES, *ANGLE_FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in feature CSV: {sorted(missing)}")
    if not vector_features:
        raise ValueError("No numeric singular-vector descriptors (u_*/v_*) found; rerun extract_features.py")
    balance = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    if 0 not in balance.columns or 1 not in balance.columns or not (balance[0] == balance[1]).all():
        raise ValueError(f"Expected balanced classes in every split; got:\n{balance}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_space_rank = int(args.class_space_rank or cfgd.get("class_space_rank", 16))
    oversample = int(args.oversample if args.oversample is not None else cfgd.get("class_space_oversample", 8))
    power_iters = int(args.power_iters if args.power_iters is not None else cfgd.get("class_space_power_iterations", 2))

    pair_features: list[str] = []
    if args.pairwise_features is not None:
        pair_df = pd.read_csv(args.pairwise_features)
        if "pair_svd_status" not in pair_df or "sample_id" not in pair_df:
            raise ValueError("Pairwise feature CSV needs sample_id and pair_svd_status columns")
        pair_df = pair_df[pair_df["pair_svd_status"].eq("ok")].copy()
        if pair_df["sample_id"].duplicated().any():
            raise ValueError("Pairwise feature file has duplicate sample_id values")
        pair_features = [c for c in pair_df.columns if c.startswith("pair_") and pd.api.types.is_numeric_dtype(pair_df[c])]
        df = df.merge(pair_df[["sample_id", *pair_features]], on="sample_id", how="left", validate="one_to_one")
        matched = int(df[pair_features].notna().any(axis=1).sum()) if pair_features else 0
        print(f"Pairwise SVD features merged for {matched}/{len(df)} clips")

    value_features = VALUE_FEATURES
    angle_features = ANGLE_FEATURES
    all_local_features = value_features + vector_features + angle_features
    prediction_frames: list[pd.DataFrame] = []
    result_rows = []
    model_specs = [
        ("local_svd_values_only", value_features, "logistic"),
        ("local_svd_vectors_only", vector_features, "logistic"),
        ("local_svd_window_angles_only", angle_features, "logistic"),
        ("local_svd_all_features", all_local_features, "logistic"),
        ("local_svd_all_features_gmm", all_local_features, "gmm"),
        ("bandwidth_raw_diagnostic", BASELINE_FEATURES, "logistic"),
    ]
    for name, features, estimator in model_specs:
        result_rows.append(metrics_row(
            df, features, name, args.seed, estimator, prediction_frames, args.bootstrap_reps,
        ))

    human_basis, human_s = fit_class_space(
        df, 0, cfg, args.output_dir, class_space_rank, oversample,
        power_iters, args.stack_block_columns, args.seed, args.keep_stacked_matrices,
        args.workers, args.chunksize,
    )
    ai_basis, ai_s = fit_class_space(
        df, 1, cfg, args.output_dir, class_space_rank, oversample,
        power_iters, args.stack_block_columns, args.seed, args.keep_stacked_matrices,
        args.workers, args.chunksize,
    )
    alignment_summary, matched_vectors, principal_angles = compare_class_bases(
        human_basis, ai_basis, human_s, ai_s,
    )
    (args.output_dir / "class_space_comparison.json").write_text(json.dumps(alignment_summary, indent=2))
    pd.DataFrame(matched_vectors).to_csv(args.output_dir / "class_space_vector_alignment.csv", index=False)
    pd.DataFrame(principal_angles).to_csv(args.output_dir / "class_space_principal_angles.csv", index=False)
    print("Stacked-space alignment: "
          f"mean principal angle={alignment_summary['mean_principal_angle_deg']:.2f} deg; "
          f"mean matched |cos|={alignment_summary['mean_matched_abs_cosine']:.3f}")

    if args.crossfit_folds:
        space_df = crossfit_class_space_features(
            df, cfg, human_basis, ai_basis, args.output_dir,
            class_space_rank, oversample, power_iters, args.stack_block_columns,
            args.seed, args.workers, args.chunksize, args.crossfit_folds,
            args.keep_stacked_matrices,
        )
        space_mode = f"{args.crossfit_folds}-fold caption-group out-of-fold projections for training rows"
    else:
        space_df = add_class_space_features(df, cfg, human_basis, ai_basis, args.workers, args.chunksize)
        space_mode = "in-sample training projections (cross-fitting disabled)"
    space_df.to_csv(args.output_dir / "features_with_class_spaces.csv", index=False)
    result_rows.append(metrics_row(
        space_df, SPACE_FEATURES, "knn_human_vs_ai_class_spaces_crossfit", args.seed,
        "knn", prediction_frames, args.bootstrap_reps,
    ))

    combined_features = SPACE_FEATURES + vector_features + pair_features
    if pair_features or vector_features:
        result_rows.append(metrics_row(
            space_df, combined_features, "knn_class_spaces_plus_singular_vectors_and_pairs",
            args.seed, "knn", prediction_frames, args.bootstrap_reps,
        ))

    result = pd.DataFrame(result_rows)
    result.to_csv(args.output_dir / "metrics.csv", index=False)
    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
        _write_paired_model_differences(
            predictions, args.output_dir / "paired_model_differences.csv", args.bootstrap_reps, args.seed,
        )

    experiment = {
        "seed": args.seed, "feature_csv": str(args.features),
        "pairwise_feature_csv": str(args.pairwise_features) if args.pairwise_features else None,
        "class_counts_by_split": balance.to_dict(), "class_space_rank": class_space_rank,
        "oversample": oversample, "power_iterations": power_iters,
        "class_spaces": "explicit disk-backed class-specific training-only stacks; randomized truncated SVD",
        "stacked_vector_comparison": "left singular-vector bases compared with principal angles and Hungarian-matched absolute cosines",
        "class_space_training_projection": space_mode,
        "crossfit_folds": args.crossfit_folds,
        "knn": "distance-weighted Euclidean KNN; k selected by validation balanced accuracy",
        "gmm": "class-conditional diagonal Gaussian mixtures; component count selected on validation",
        "bootstrap": f"caption-group bootstrap, {args.bootstrap_reps} replicates; paired model deltas use identical group resamples",
        "paired_ranking": "within each held-out ytid pair, compare p(AI) for AI clip against p(AI) for human counterpart; ties count 0.5",
        "pairwise_svd": "one exact SVD of [L_human L_AI] per matched pair; paired features require the counterpart at inference",
        "models": result_rows,
    }
    (args.output_dir / "experiment.json").write_text(json.dumps(experiment, indent=2, default=int))
    cols = ["model", "estimator", "n_test", "balanced_accuracy", "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high", "roc_auc", "paired_ranking_accuracy"]
    print(result[cols].to_string(index=False))
    print(f"Results written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
