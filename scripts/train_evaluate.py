#!/usr/bin/env python3
"""Train/evaluate bandwidth-only, local-SVD, and shared-basis feature models."""

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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from music_svd_detection.shared_basis import (
    randomized_shared_svd_basis,
    shared_basis_clip_features,
)
from music_svd_detection.spectral import SpectralConfig, log_magnitude_spectrogram

LOCAL_FEATURES = [
    "effective_rank",
    "sigma1_over_fro",
    "topk_energy_fraction",
    "spectral_entropy_bits",
    "principal_angle_mean_deg",
    "principal_angle_std_deg",
    "subspace_stability",
] + [f"sigma_ratio_{i}" for i in range(1, 9)]
BASELINE_FEATURES = ["raw_bandwidth_cutoff_hz"]


def read_logmag(path: str, cfg: SpectralConfig) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if np.asarray(audio).ndim == 2:
        audio = np.asarray(audio).mean(axis=1)
    return log_magnitude_spectrogram(audio, int(sr), cfg)


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scores = [
        (
            balanced_accuracy_score(y_true, probabilities >= t),
            abs(float(t) - 0.5),
            float(t),
        )
        for t in candidates
    ]
    # Maximize validation balanced accuracy; ties prefer a threshold nearest 0.5.
    return max(scores, key=lambda x: (x[0], -x[1]))[2]


def evaluate_model(
    df: pd.DataFrame, features: list[str], name: str, seed: int
) -> dict[str, object]:
    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "validation"].copy()
    te = df[df["split"] == "test"].copy()
    if tr.empty or te.empty:
        raise ValueError("Manifest needs non-empty train and test partitions")
    xtr = tr[features].replace([np.inf, -np.inf], np.nan)
    ytr = tr["label"].astype(int).to_numpy()
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
    )
    model.fit(xtr, ytr)
    threshold = 0.5
    if not va.empty and va["label"].nunique() == 2:
        pval = model.predict_proba(va[features].replace([np.inf, -np.inf], np.nan))[
            :, 1
        ]
        threshold = choose_threshold(va["label"].astype(int).to_numpy(), pval)
    xte = te[features].replace([np.inf, -np.inf], np.nan)
    yte = te["label"].astype(int).to_numpy()
    prob = model.predict_proba(xte)[:, 1]
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    return {
        "model": name,
        "n_features": len(features),
        "features": ",".join(features),
        "n_train": int(len(tr)),
        "n_validation": int(len(va)),
        "n_test": int(len(te)),
        "threshold_from_validation": float(threshold),
        "accuracy": float(accuracy_score(yte, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "roc_auc": (
            float(roc_auc_score(yte, prob))
            if np.unique(yte).size == 2
            else float("nan")
        ),
        "precision_ai": float(precision_score(yte, pred, zero_division=0)),
        "recall_ai": float(recall_score(yte, pred, zero_division=0)),
        "f1_ai": float(f1_score(yte, pred, zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--features", type=Path, default=Path("data/processed/features.csv")
    )
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--no-shared-basis", action="store_true")
    ap.add_argument("--seed", type=int, default=20261026)
    args = ap.parse_args()
    cfgd = json.loads(args.config.read_text())
    cfg = SpectralConfig(
        sample_rate=int(cfgd["sample_rate"]),
        n_fft=int(cfgd["n_fft"]),
        hop_length=int(cfgd["hop_length"]),
        log_epsilon=float(cfgd["log_epsilon"]),
        local_top_k=int(cfgd["local_top_k"]),
        subspace_window_frames=int(cfgd["subspace_window_frames"]),
        subspace_step_frames=int(cfgd["subspace_step_frames"]),
    )
    df = pd.read_csv(args.features)
    if "feature_status" in df:
        df = df[df["feature_status"].eq("ok")].copy()
    # If feature extraction failed for one side of a pair, discard both sides.
    if not df.empty and {"ytid", "label"}.issubset(df.columns):
        pair_counts = df.groupby("ytid")["label"].agg(["count", "sum"])
        complete_ids = pair_counts.index[
            (pair_counts["count"] == 2) & (pair_counts["sum"] == 1)
        ]
        df = df[df["ytid"].isin(complete_ids)].copy()
    required = {
        "split",
        "label",
        "processed_path",
        "raw_bandwidth_cutoff_hz",
        *LOCAL_FEATURES,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in feature CSV: {sorted(missing)}")
    # The manifest constructor makes paired rows balanced. Check rather than silently proceed.
    balance = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    if (
        0 not in balance.columns
        or 1 not in balance.columns
        or not (balance[0] == balance[1]).all()
    ):
        raise ValueError(f"Expected balanced classes in every split; got:\n{balance}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_features = df.copy()
    if not args.no_shared_basis:
        train_paths = (
            df.loc[df["split"].eq("train"), "processed_path"].astype(str).tolist()
        )
        if not train_paths:
            raise ValueError("No training clips available to fit shared basis")

        def block_factory():
            for path in train_paths:
                yield read_logmag(path, cfg)

        basis, shared_s = randomized_shared_svd_basis(
            block_factory,
            rank=int(cfgd["shared_basis_rank"]),
            oversample=int(cfgd["shared_basis_oversample"]),
            n_power_iterations=int(cfgd["shared_basis_power_iterations"]),
            seed=args.seed,
        )
        np.save(args.output_dir / "shared_spectral_basis.npy", basis)
        pd.DataFrame({"singular_value_approx": shared_s}).to_csv(
            args.output_dir / "shared_basis_singular_values.csv", index=False
        )
        additions = []
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Projecting clips onto shared basis"
        ):
            lm = read_logmag(str(row["processed_path"]), cfg)
            additions.append(shared_basis_clip_features(lm, basis))
        extra = pd.DataFrame(additions, index=df.index)
        for col in extra.columns:
            shared_features[col] = extra[col]
        shared_features.to_csv(
            args.output_dir / "features_with_shared_basis.csv", index=False
        )
    else:
        shared_features = df.copy()

    result_rows = []
    result_rows.append(
        evaluate_model(df, BASELINE_FEATURES, "bandwidth_only_baseline", args.seed)
    )
    local_cols = [c for c in LOCAL_FEATURES if c in df.columns]
    result_rows.append(
        evaluate_model(df, local_cols, "local_exact_svd_and_subspace_angles", args.seed)
    )
    if not args.no_shared_basis:
        shared_cols = local_cols + [
            "shared_basis_energy_fraction",
            "shared_basis_relative_residual",
        ]
        result_rows.append(
            evaluate_model(
                shared_features, shared_cols, "local_plus_train_shared_basis", args.seed
            )
        )
    result = pd.DataFrame(result_rows)
    result.to_csv(args.output_dir / "metrics.csv", index=False)
    (args.output_dir / "experiment.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "feature_csv": str(args.features),
                "class_counts_by_split": balance.to_dict(),
                "effective_rank": "exp(-sum_i p_i log p_i), p_i = sigma_i/sum_j sigma_j; log-magnitude matrix is globally mean-centered",
                "subspace_stability": "1 - mean principal angle/(pi/2); principal angles are from singular values of U_j.T U_{j+1}",
                "shared_basis": "randomized SVD approximation of [L_train_1 L_train_2 ...], training partition only",
                "models": result_rows,
            },
            indent=2,
            default=int,
        )
    )
    print(
        result[
            [
                "model",
                "n_test",
                "balanced_accuracy",
                "roc_auc",
                "threshold_from_validation",
            ]
        ].to_string(index=False)
    )
    print(f"Results written to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
