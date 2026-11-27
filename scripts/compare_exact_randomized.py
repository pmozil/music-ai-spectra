#!/usr/bin/env python3
"""Compare exact and randomized per-clip SVD features, principal angles and regressions."""
from __future__ import annotations
import argparse, json, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.utils.extmath import randomized_svd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from music_svd_detection.parallel import limit_worker_threads
from music_svd_detection.spectral import SpectralConfig, log_magnitude_spectrogram

def effective_rank(s):
    s = np.asarray(s, dtype=float)
    p = s[s > 0] / np.sum(s[s > 0])
    return float(np.exp(-np.sum(p * np.log(p)))) if len(p) else 0.0

def angle_values(b0, b1):
    c = np.linalg.svd(b0.T @ b1, compute_uv=False)
    return np.arccos(np.clip(c, 0.0, 1.0))

def get_basis(a, k, method, seed):
    if method == "exact":
        u, s, _ = np.linalg.svd(a, full_matrices=False)
        return u[:, :k], s[:k]
    u, s, _ = randomized_svd(a, n_components=k, n_iter=4, random_state=seed)
    return u, s

def features_for(logmag, cfg, k, method, seed):
    u, s = get_basis(logmag, k, method, seed)
    total = float(np.sum(logmag * logmag))
    full_s = np.linalg.svd(logmag, compute_uv=False) if method == "exact" else s
    ratios = [float(s[i] / s[0]) if s[0] else 0.0 for i in range(min(k, len(s)))]
    width = min(cfg.subspace_window_frames, logmag.shape[1])
    starts = list(range(0, logmag.shape[1] - width + 1, cfg.subspace_step_frames)) if width >= 2 else []
    bases = []
    for start in starts:
        b, _ = get_basis(logmag[:, start:start+width], min(k, width), method, seed)
        bases.append(b)
    angles = np.concatenate([angle_values(a, b) for a, b in zip(bases[:-1], bases[1:])]) if len(bases) > 1 else np.array([np.nan])
    out = {
        "effective_rank": effective_rank(full_s),
        "sigma1_over_fro": float(full_s[0] / np.sqrt(total)) if total > 0 else 0.0,
        "topk_energy_fraction": float(np.sum(s*s) / total) if total > 0 else 0.0,
        "principal_angle_mean_deg": float(np.degrees(np.nanmean(angles))),
        "principal_angle_std_deg": float(np.degrees(np.nanstd(angles))),
        "subspace_stability": float(1.0 - np.nanmean(angles)/(np.pi/2)),
    }
    for i, value in enumerate(ratios, 1):
        out[f"sigma_ratio_{i}"] = value
    return out, angles

BASE_FEATURES = ["effective_rank","sigma1_over_fro","topk_energy_fraction",
                 "principal_angle_mean_deg","principal_angle_std_deg","subspace_stability"]

def fit_predict(wide, method, seed):
    """Train on the train split, return (test rows, P(AI)). Threshold is 0.5."""
    cols = [f"{method}_{x}" for x in BASE_FEATURES]
    train, test = wide[wide.split.eq("train")], wide[wide.split.eq("test")]
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed))
    model.fit(train[cols], train.label.astype(int))
    return test, model.predict_proba(test[cols])[:,1]

def regression_metrics(wide, seed):
    metrics = []
    n_train = int(wide.split.eq("train").sum())
    for method in ("exact","randomized"):
        test, prob = fit_predict(wide, method, seed)
        pred = prob >= 0.5
        metrics.append({"model": f"{method}_svd_features_logistic_regression",
            "n_train":n_train,"n_test":len(test),
            "accuracy":accuracy_score(test.label,pred),
            "balanced_accuracy":balanced_accuracy_score(test.label,pred),
            "roc_auc":roc_auc_score(test.label,prob) if test.label.nunique()==2 else np.nan})
    return metrics

def _process_record(task):
    record, cfg, rank, seed = task
    limit_worker_threads()
    try:
        audio, sr = sf.read(str(record["processed_path"]), dtype="float32", always_2d=False)
        if np.asarray(audio).ndim == 2:
            audio = np.asarray(audio).mean(axis=1)
        lm = log_magnitude_spectrogram(audio, int(sr), cfg)
        rows, angle_rows = [], []
        for method in ("exact", "randomized"):
            features, angles = features_for(lm, cfg, rank, method, seed)
            rec = {key: record[key] for key in ("sample_id", "ytid", "label", "class_name", "generator", "split")}
            rows.append({**rec, "method": method, **features})
            for j, angle in enumerate(angles):
                angle_rows.append({"sample_id": record["sample_id"], "ytid": record["ytid"],
                    "label": int(record["label"]), "class_name": record["class_name"],
                    "method": method, "angle_index": j, "angle_deg": float(np.degrees(angle))})
        return rows, angle_rows, None
    except Exception as exc:
        return [], [], f"{record.get('sample_id', '?')}: {type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_harmonized.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/svd_comparison"))
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--limit-pairs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--chunksize", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261112)
    args = ap.parse_args()
    if args.rank < 1 or args.workers < 1 or args.chunksize < 1:
        ap.error("--rank, --workers and --chunksize must be positive")
    c = json.loads(args.config.read_text())
    cfg = SpectralConfig(sample_rate=int(c["sample_rate"]), n_fft=int(c["n_fft"]),
        hop_length=int(c["hop_length"]), log_epsilon=float(c["log_epsilon"]),
        local_top_k=int(c["local_top_k"]), subspace_window_frames=int(c["subspace_window_frames"]),
        subspace_step_frames=int(c["subspace_step_frames"]), max_freq_hz=float(c.get("max_freq_hz", 0)))
    df = pd.read_csv(args.manifest)
    if "processing_status" in df:
        df = df[df["processing_status"].eq("ok")].copy()
    if args.limit_pairs:
        ids = df.ytid.drop_duplicates().head(args.limit_pairs)
        df = df[df.ytid.isin(ids)]
    counts = df.groupby("ytid").label.agg(["count", "sum"])
    good = counts.index[(counts["count"] == 2) & (counts["sum"] == 1)]
    df = df[df.ytid.isin(good)].copy()
    records = df.to_dict("records")
    tasks = ((r, cfg, args.rank, args.seed) for r in records)
    rows, angle_rows, errors = [], [], []
    with tqdm(total=len(records), desc=f"Exact/randomized SVD ({args.workers} workers)") as bar:
        if args.workers == 1:
            iterator = map(_process_record, tasks)
            for result, angles, error in iterator:
                rows.extend(result); angle_rows.extend(angles)
                if error: errors.append(error)
                bar.update(1)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=limit_worker_threads) as pool:
                for result, angles, error in pool.map(_process_record, tasks, chunksize=args.chunksize):
                    rows.extend(result); angle_rows.extend(angles)
                    if error: errors.append(error)
                    bar.update(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feats = pd.DataFrame(rows)
    if feats.empty:
        raise SystemExit("No clips were processed successfully")
    feats.to_csv(args.output_dir / "features_long.csv", index=False)
    pd.DataFrame(angle_rows).to_csv(args.output_dir / "principal_angles.csv", index=False)
    if errors:
        (args.output_dir / "errors.txt").write_text("\n".join(errors) + "\n")
    else:
        (args.output_dir / "errors.txt").unlink(missing_ok=True)

    wide = feats.pivot(index=["sample_id","ytid","label","class_name","generator","split"], columns="method")
    wide.columns = [f"{method}_{feature}" for feature, method in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(args.output_dir / "features_exact_vs_randomized.csv", index=False)
    metrics = regression_metrics(wide, args.seed)
    pd.DataFrame(metrics).to_csv(args.output_dir / "regression_metrics.csv", index=False)
    angles = pd.DataFrame(angle_rows)
    summary = angles.groupby(["method","class_name"]).angle_deg.agg(["count","mean","std","median"]).reset_index()
    summary.to_csv(args.output_dir / "angle_summary.csv", index=False)
    print(pd.DataFrame(metrics).to_string(index=False))
    print("\nPrincipal-angle summary:\n", summary.to_string(index=False))
    print(f"\nProcessed {len(feats) // 2} clips successfully; errors: {len(errors)}")
    print(f"Outputs: {args.output_dir}")

if __name__ == "__main__":
    main()
