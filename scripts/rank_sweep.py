#!/usr/bin/env python3
"""Rank sweep: accuracy of the exact and randomized SVD features for several k.

For every k in --ranks this computes the same 6 features and the same logistic
regression as compare_exact_randomized.py --rank k (at k = 20 the numbers match
that script), but each clip is read and its exact SVD is computed only once:

* exact: one full SVD of the clip and one per sliding window; rank k just takes
  the first k columns. Effective rank and sigma1/||A||_F use all singular values,
  so only top-k energy and the principal angles depend on k.
* randomized: one randomized_svd(n_components=k) per k, as in the compare script.
  Its effective rank sees only the top k singular values, so it is <= k.
  Window bases use min(k, window) columns. When that is the whole window
  (k >= 64 frames), the basis is the window's full column space, which the exact
  SVD gives directly (same subspace, much faster than randomized_svd;
  angles agree with the compare script to about 0.01 degrees).

Outputs in --output-dir:
  rank_K/features_exact_vs_randomized.csv, rank_K/regression_metrics.csv
      (same format as compare_exact_randomized.py, so generator_report.py
       --compare-dir output/rank_K works on them)
  rank_sweep_metrics.csv   accuracy / AUC per k and method, plus effective rank
                           and top-k energy medians per class
  rank_sweep.png           the plot
  features_long.csv        per-clip cache; rerunning the same command resumes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

SCRIPTS = str(Path(__file__).resolve().parent)
sys.path.insert(0, SCRIPTS)
# worker processes (--workers) need to find compare_exact_randomized too
os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (SCRIPTS, os.environ.get("PYTHONPATH", "")) if p)
import compare_exact_randomized as compare  # noqa: E402
from music_svd_detection.spectral import SpectralConfig, log_magnitude_spectrogram  # noqa: E402

ID_COLS = ["sample_id", "ytid", "label", "class_name", "generator", "split"]


def window_starts(n_frames: int, cfg: SpectralConfig) -> tuple[int, list[int]]:
    width = min(cfg.subspace_window_frames, n_frames)
    starts = list(range(0, n_frames - width + 1, cfg.subspace_step_frames)) if width >= 2 else []
    return width, starts


def angle_stats(bases: list[np.ndarray]) -> dict[str, float]:
    if len(bases) > 1:
        angles = np.concatenate([compare.angle_values(a, b) for a, b in zip(bases[:-1], bases[1:])])
    else:
        angles = np.array([np.nan])
    return {
        "principal_angle_mean_deg": float(np.degrees(np.nanmean(angles))),
        "principal_angle_std_deg": float(np.degrees(np.nanstd(angles))),
        "subspace_stability": float(1.0 - np.nanmean(angles) / (np.pi / 2)),
    }


def clip_features(lm: np.ndarray, cfg: SpectralConfig, ranks: list[int], seed: int,
                  window_rank: int = 0) -> list[dict]:
    """Features for every (rank, method) of one log-magnitude spectrogram.

    window_rank > 0 fixes the window-basis rank (principal angles) for every k,
    so only the global top-k features change with k. 0 = use k (compare script)."""
    total = float(np.sum(lm * lm))
    u_full, s_full, _ = np.linalg.svd(lm, full_matrices=False)
    s_all = np.linalg.svd(lm, compute_uv=False)  # what the compare script uses for erank / sigma1
    width, starts = window_starts(lm.shape[1], cfg)
    windows = [lm[:, st:st + width] for st in starts]
    win_u = [np.linalg.svd(w, full_matrices=False)[0] for w in windows]
    out = []
    rand_bases: dict[int, list[np.ndarray]] = {}
    for k in ranks:
        kw = min(window_rank or k, width)
        # exact
        s = s_full[:k]
        rec = {"rank": k, "method": "exact",
               "effective_rank": compare.effective_rank(s_all),
               "sigma1_over_fro": float(s_all[0] / np.sqrt(total)) if total > 0 else 0.0,
               "topk_energy_fraction": float(np.sum(s * s) / total) if total > 0 else 0.0,
               **angle_stats([u[:, :kw] for u in win_u])}
        out.append(rec)
        # randomized
        _, s = compare.get_basis(lm, k, "randomized", seed)
        if kw >= width:
            bases = [u[:, :kw] for u in win_u]
        else:
            if kw not in rand_bases:
                rand_bases[kw] = [compare.get_basis(w, kw, "randomized", seed)[0] for w in windows]
            bases = rand_bases[kw]
        rec = {"rank": k, "method": "randomized",
               "effective_rank": compare.effective_rank(s),
               "sigma1_over_fro": float(s[0] / np.sqrt(total)) if total > 0 else 0.0,
               "topk_energy_fraction": float(np.sum(s * s) / total) if total > 0 else 0.0,
               **angle_stats(bases)}
        out.append(rec)
    return out


def process(row: dict, cfg: SpectralConfig, ranks: list[int], seed: int, window_rank: int = 0) -> list[dict]:
    # Many small SVDs: one BLAS thread per clip is about 5x faster than letting
    # BLAS spread each tiny SVD over all cores. Use --workers for parallelism.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return _process(row, cfg, ranks, seed, window_rank)


def _process(row: dict, cfg: SpectralConfig, ranks: list[int], seed: int, window_rank: int = 0) -> list[dict]:
    try:
        audio, sr = sf.read(str(row["processed_path"]), dtype="float32", always_2d=False)
        if np.asarray(audio).ndim == 2:
            audio = np.asarray(audio).mean(axis=1)
        lm = log_magnitude_spectrogram(audio, int(sr), cfg)
        ids = {c: row[c] for c in ID_COLS}
        return [{**ids, **f} for f in clip_features(lm, cfg, ranks, seed, window_rank)]
    except Exception as exc:  # same behaviour as the compare script: skip and report
        print(f"Skipping {row['sample_id']}: {type(exc).__name__}: {exc}")
        return []


def summarise(feats: pd.DataFrame, ranks: list[int], seed: int, out_dir: Path) -> pd.DataFrame:
    rows = []
    feats = feats.copy()
    for k in ranks:
        sub = feats[feats["rank"].eq(k)].drop(columns="rank")
        wide = sub.pivot(index=ID_COLS, columns="method")
        wide.columns = [f"{m}_{f}" for f, m in wide.columns]
        wide = compare_pairs(wide.reset_index())
        d = out_dir / f"rank_{k}"
        d.mkdir(parents=True, exist_ok=True)
        wide.to_csv(d / "features_exact_vs_randomized.csv", index=False)
        metrics = compare.regression_metrics(wide, seed)
        pd.DataFrame(metrics).to_csv(d / "regression_metrics.csv", index=False)
        for m in metrics:
            method = m["model"].split("_")[0]
            rec = {"rank": k, "method": method, **{x: m[x] for x in ("n_train", "n_test", "accuracy",
                                                                   "balanced_accuracy", "roc_auc")}}
            for lab, name in ((0, "human"), (1, "ai")):
                cls = wide[wide["label"].eq(lab)]
                rec[f"median_effective_rank_{name}"] = float(cls[f"{method}_effective_rank"].median())
                e = cls[f"{method}_topk_energy_fraction"]
                rec[f"mean_topk_energy_{name}"] = float(e.mean())
                # Eckart-Young: ||A - A_k||_F / ||A||_F = sqrt(1 - top-k energy)
                rec[f"mean_rel_frobenius_error_{name}"] = float(np.sqrt(np.clip(1 - e, 0, 1)).mean())
            rows.append(rec)
    return pd.DataFrame(rows)


def compare_pairs(df: pd.DataFrame) -> pd.DataFrame:
    c = df.groupby("ytid")["label"].agg(["count", "sum"])
    return df[df["ytid"].isin(c.index[(c["count"] == 2) & (c["sum"] == 1)])].copy()


def plot(summary: pd.DataFrame, path: Path, window_rank: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"exact": "#2a78d6", "randomized": "#eb6834"}
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    ranks = sorted(summary["rank"].unique())
    for method in ("exact", "randomized"):
        s = summary[summary["method"].eq(method)].sort_values("rank")
        a1.plot(s["rank"], s["accuracy"], marker="o", ms=6, lw=2, color=colors[method], label=method)
        for name, ls in (("human", "-"), ("ai", "--")):
            a2.plot(s["rank"], s[f"median_effective_rank_{name}"], marker="o", ms=5, lw=2, ls=ls,
                    color=colors[method],
                    label=f"{'exact (all singular values)' if method == 'exact' else 'randomized (top k only)'}, "
                          f"{'human' if name == 'human' else 'AI'}")
    a1.axhline(0.5, color=muted, lw=1, ls=":")
    a1.text(ranks[0], 0.505, "chance", color=muted, fontsize=8, va="bottom")
    title = "Logistic regression on 6 SVD features"
    if window_rank:
        title += f"\n(window rank fixed at {window_rank})"
    a1.set(xlabel="rank k", ylabel="test accuracy", title=title)
    lo = min(0.45, float(summary["accuracy"].min()) - 0.03)
    a1.set_ylim(lo, 1.0)
    a2.plot(ranks, ranks, color=muted, lw=1, ls=":", label="erank = k (upper bound for top-k)")
    a2.set(xlabel="rank k", ylabel="median effective rank",
           title="Effective rank: full spectrum vs top-k only")
    for ax in (a1, a2):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(r) for r in ranks])
        ax.grid(True, color=grid, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_harmonized.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/rank_sweep"))
    ap.add_argument("--ranks", default="10,20,40,80")
    ap.add_argument("--window-rank", type=int, default=None,
                    help="fix the sliding-window basis rank; default uses configs/default.json window_rank; "
                         "0 = use each swept k")
    ap.add_argument("--limit-pairs", type=int, default=0)
    ap.add_argument("--include-validation", action="store_true",
                    help="also process validation clips (the classifier does not use them)")
    ap.add_argument("--workers", type=int, default=4, help="parallel processes (1 = no parallelism)")
    ap.add_argument("--seed", type=int, default=20261112)
    args = ap.parse_args()
    try:
        ranks = sorted({int(r) for r in args.ranks.split(",") if r.strip()})
    except ValueError:
        ap.error("--ranks must be comma-separated positive integers")
    if not ranks or min(ranks) < 1 or args.workers < 1:
        ap.error("--ranks and --workers must be positive")

    c = json.loads(args.config.read_text())
    window_rank = int(c.get("window_rank", 0) if args.window_rank is None else args.window_rank)
    if window_rank < 0:
        ap.error("--window-rank must be non-negative")
    cfg = SpectralConfig(sample_rate=int(c["sample_rate"]), n_fft=int(c["n_fft"]),
        hop_length=int(c["hop_length"]), log_epsilon=float(c["log_epsilon"]),
        local_top_k=int(c["local_top_k"]), subspace_window_frames=int(c["subspace_window_frames"]),
        subspace_step_frames=int(c["subspace_step_frames"]),
        max_freq_hz=float(c.get("max_freq_hz", 0)))

    df = pd.read_csv(args.manifest)
    if "processing_status" in df:
        df = df[df["processing_status"].eq("ok")].copy()
    if args.limit_pairs:
        df = df[df["ytid"].isin(df["ytid"].drop_duplicates().head(args.limit_pairs))]
    if not args.include_validation:
        df = df[df["split"].isin(["train", "test"])]
    df = compare_pairs(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "features_long.csv"
    info = args.output_dir / "sweep_settings.json"
    settings = {"ranks": ranks, "window_rank": window_rank, "seed": args.seed, "config": c,
                "manifest": str(args.manifest)}
    done: set[str] = set()
    if cache.exists():
        old = json.loads(info.read_text()) if info.exists() else {}
        old.setdefault("window_rank", 0)
        keys = ("ranks", "window_rank", "seed", "config")
        if {k: old.get(k) for k in keys} != {k: settings[k] for k in keys}:
            raise SystemExit(f"{cache} was made with different settings. Use a new --output-dir or delete that folder.")
        done = set(pd.read_csv(cache, usecols=["sample_id"])["sample_id"])
        print(f"Resuming: {len(done)} clips already in {cache}")
    info.write_text(json.dumps(settings, indent=2))

    todo = df[~df["sample_id"].isin(done)].to_dict("records")
    chunk = 100
    with tqdm(total=len(todo), desc=f"Rank sweep k={ranks}") as bar:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            if args.workers > 1:
                from joblib import Parallel, delayed
                res = Parallel(n_jobs=args.workers)(delayed(process)(r, cfg, ranks, args.seed, window_rank) for r in part)
            else:
                res = [process(r, cfg, ranks, args.seed, window_rank) for r in part]
            rows = [x for r in res for x in r]
            if rows:
                pd.DataFrame(rows).to_csv(cache, mode="a", header=not cache.exists(), index=False)
            bar.update(len(part))

    feats = pd.read_csv(cache)
    feats = feats[feats["sample_id"].isin(df["sample_id"])]
    summary = summarise(feats, ranks, args.seed, args.output_dir)
    summary.to_csv(args.output_dir / "rank_sweep_metrics.csv", index=False)
    plot(summary, args.output_dir / "rank_sweep.png", window_rank)
    cols = ["rank", "method", "n_test", "accuracy", "roc_auc",
            "median_effective_rank_human", "median_effective_rank_ai", "mean_topk_energy_human", "mean_topk_energy_ai"]
    pd.set_option("display.width", 200)
    print(summary[cols].round(3).to_string(index=False))
    print(f"\nOutputs: {args.output_dir}/rank_sweep_metrics.csv, rank_sweep.png, rank_K/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
