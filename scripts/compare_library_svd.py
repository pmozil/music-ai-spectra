#!/usr/bin/env python3
"""Interim report 2: our randomized SVD vs library routines on the stacked matrix.

A = [L_1 L_2 ... L_N] stacks the 6 kHz-cropped, mean-subtracted log-magnitude
spectrograms of N training clips (F rows, about 858 N columns).  For several N we
compute

  * the exact thin SVD with NumPy/LAPACK (np.linalg.svd, the reference),
  * our streaming randomized SVD (music_svd_detection.shared_basis), with target
    rank k, oversampling p and q power iterations,
  * scikit-learn's randomized_svd with the same k, p, q (a library rSVD),

and record

  * relative error of the top-k singular values,
        ||sigma_rand - sigma_exact||_2 / ||sigma_exact||_2   (criterion: <= 1e-2),
  * per-index relative errors |sigma_i^rand - sigma_i| / sigma_i,
  * the largest principal angle between the rank-k left singular subspaces,
  * median wall-clock time over --repeats runs and the predicted flop counts.

Outputs (in --output-dir): library_svd_summary.csv, library_svd_per_index.csv,
settings.json and fig_rsvd_vs_exact.pdf/.png (the figure for the report).

Typical run on the real data (a few minutes):

    python scripts/compare_library_svd.py

Dry run without audio, on synthetic matrices (checks the code only; do NOT put
these numbers in the report):

    python scripts/compare_library_svd.py --synthetic --output-dir artifacts/library_svd_synthetic
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from music_svd_detection.shared_basis import randomized_shared_svd_basis


def load_config(path: Path):
    from music_svd_detection.spectral import SpectralConfig

    c = json.loads(path.read_text())
    cfg = SpectralConfig(
        sample_rate=int(c["sample_rate"]), n_fft=int(c["n_fft"]),
        hop_length=int(c["hop_length"]), log_epsilon=float(c["log_epsilon"]),
        local_top_k=int(c["local_top_k"]),
        subspace_window_frames=int(c["subspace_window_frames"]),
        subspace_step_frames=int(c["subspace_step_frames"]),
        max_freq_hz=float(c.get("max_freq_hz", 0)),
    )
    return cfg


def real_blocks(manifest: Path, config: Path, n_clips: int, split: str, seed: int):
    """Log-magnitude spectrograms of n_clips training clips, half human, half AI."""
    import soundfile as sf
    from music_svd_detection.spectral import log_magnitude_spectrogram

    cfg = load_config(config)
    df = pd.read_csv(manifest)
    if "processing_status" in df:
        df = df[df["processing_status"].eq("ok")]
    df = df[df["split"].eq(split)]
    rng = np.random.default_rng(seed)
    per_class = n_clips // 2
    picked = []
    for label in (0, 1):
        rows = df[df["label"].astype(int).eq(label)]
        idx = rng.permutation(len(rows))[:per_class]
        picked.append(rows.iloc[idx])
    # interleave human/AI so that every prefix of the list is balanced
    h, a = picked
    order = [r for pair in zip(h.itertuples(index=False), a.itertuples(index=False)) for r in pair]
    blocks = []
    for row in order:
        audio, sr = sf.read(str(row.processed_path), dtype="float32", always_2d=False)
        audio = np.asarray(audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        blocks.append(log_magnitude_spectrogram(audio, int(sr), cfg).astype(np.float64))
    return blocks


def synthetic_blocks(n_clips: int, seed: int, f: int = 279, t: int = 858):
    """Matrices with a decaying spectrum plus noise. Code check only."""
    rng = np.random.default_rng(seed)
    r = 60
    basis, _ = np.linalg.qr(rng.standard_normal((f, r)))
    decay = 40.0 * np.exp(-np.arange(r) / 6.0) + 2.0
    blocks = []
    for _ in range(n_clips):
        coeff = rng.standard_normal((r, t)) * decay[:, None]
        blocks.append(basis @ coeff + 0.5 * rng.standard_normal((f, t)))
    return blocks


def timed(fn, repeats: int):
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def max_principal_angle_deg(u_a: np.ndarray, u_b: np.ndarray) -> float:
    c = np.linalg.svd(u_a.T @ u_b, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(c.min(), 0.0, 1.0))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_harmonized.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("artifacts/library_svd"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--clip-counts", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    ap.add_argument("--rank", type=int, default=20)
    ap.add_argument("--oversample", type=int, default=10)
    ap.add_argument("--power-iters", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timing-power-iters", type=int, default=1,
                    help="q used for the timing curves in panel (b)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    from sklearn.utils.extmath import randomized_svd

    n_max = max(args.clip_counts)
    if args.synthetic:
        blocks = synthetic_blocks(n_max, args.seed)
    else:
        blocks = real_blocks(args.manifest, args.config, n_max, args.split, args.seed)
    f = blocks[0].shape[0]
    k, p = args.rank, args.oversample
    ell = k + p

    # warm-up so that BLAS/LAPACK start-up is not charged to the first timing
    np.linalg.svd(np.hstack(blocks[:2]), full_matrices=False)
    randomized_svd(np.hstack(blocks[:2]), n_components=k, n_oversamples=p, random_state=0)

    summary, per_index = [], []
    for n in sorted(args.clip_counts):
        sub = blocks[:n]
        a = np.hstack(sub)
        t_cols = a.shape[1]
        (u_ex, s_ex, _), t_exact = timed(lambda: np.linalg.svd(a, full_matrices=False), args.repeats)
        u_ex, s_ex = u_ex[:, :k], s_ex[:k]
        row_base = {"n_clips": n, "rows_F": f, "cols_T": t_cols, "rank_k": k, "oversample_p": p}
        summary.append({**row_base, "method": "exact_numpy_lapack", "power_iters_q": np.nan,
                        "seconds": t_exact, "sigma_rel_err": 0.0, "sigma_max_index_rel_err": 0.0,
                        "max_principal_angle_deg": 0.0,
                        "predicted_flops": 4.0 * f * f * t_cols + 8.0 * f ** 3})
        for q in args.power_iters:
            (u_own, s_own), t_own = timed(
                lambda: randomized_shared_svd_basis(lambda: iter(sub), rank=k, oversample=p,
                                                    n_power_iterations=q, seed=args.seed),
                args.repeats)
            (u_sk, s_sk, _), t_sk = timed(
                lambda: randomized_svd(a, n_components=k, n_oversamples=p, n_iter=q,
                                       power_iteration_normalizer="QR", random_state=args.seed),
                args.repeats)
            flops = 2.0 * (2 * q + 2) * f * t_cols * ell + 4.0 * f * ell ** 2
            for name, u, s, sec in (("own_randomized", u_own, s_own, t_own),
                                    ("sklearn_randomized", u_sk, s_sk, t_sk)):
                rel = np.abs(s - s_ex) / s_ex
                summary.append({**row_base, "method": name, "power_iters_q": q, "seconds": sec,
                                "sigma_rel_err": float(np.linalg.norm(s - s_ex) / np.linalg.norm(s_ex)),
                                "sigma_max_index_rel_err": float(rel.max()),
                                "max_principal_angle_deg": max_principal_angle_deg(u_ex, np.asarray(u, float)),
                                "predicted_flops": flops})
                for i, (e, se) in enumerate(zip(rel, s_ex), 1):
                    per_index.append({"n_clips": n, "method": name, "power_iters_q": q,
                                      "index": i, "sigma_exact": float(se), "rel_err": float(e)})
        print(f"N={n:4d} clips, A is {f} x {t_cols}: exact {t_exact:.2f}s", flush=True)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    sm, pi = pd.DataFrame(summary), pd.DataFrame(per_index)
    sm.to_csv(out / "library_svd_summary.csv", index=False)
    pi.to_csv(out / "library_svd_per_index.csv", index=False)
    (out / "settings.json").write_text(json.dumps({**{k_: str(v) for k_, v in vars(args).items()},
                                                   "F": f, "ell": ell}, indent=2))
    print(sm.to_string(index=False))
    make_figure(sm, pi, out, n_max, args.timing_power_iters, args.synthetic)


def make_figure(sm, pi, out, n_max, q_time, synthetic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {0: "#9aa5b1", 1: "#1f6fb2", 2: "#0b3d66"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    own = pi[(pi.method == "own_randomized") & (pi.n_clips == n_max)]
    for q, grp in own.groupby("power_iters_q"):
        ax1.semilogy(grp["index"], np.maximum(grp["rel_err"], 1e-16), "o-", ms=3,
                     color=colors.get(int(q), None), label=f"$q={int(q)}$")
    ax1.axhline(1e-2, color="#c0392b", lw=0.8, ls="--")
    ax1.text(1, 1.3e-2, "criterion $10^{-2}$", color="#c0392b", fontsize=7)
    ax1.set_xlabel("index $i$")
    ax1.set_ylabel(r"$|\tilde\sigma_i-\sigma_i|/\sigma_i$")
    ax1.set_title(f"(a) our rSVD vs LAPACK, N={n_max} clips", fontsize=8)
    ax1.legend(fontsize=7, frameon=False)

    styles = {"exact_numpy_lapack": ("NumPy/LAPACK exact", "k", "s"),
              "own_randomized": (f"our rSVD ($q={q_time}$)", "#1f6fb2", "o"),
              "sklearn_randomized": (f"scikit-learn rSVD ($q={q_time}$)", "#e67e22", "^")}
    for method, (label, color, marker) in styles.items():
        g = sm[sm.method == method]
        if method != "exact_numpy_lapack":
            g = g[g.power_iters_q == q_time]
        ax2.loglog(g["cols_T"], g["seconds"], marker=marker, color=color, ms=4, label=label)
    ax2.set_xlabel("columns of $A$ ($\\approx 858N$)")
    ax2.set_ylabel("wall-clock time (s)")
    ax2.set_title("(b) runtime", fontsize=8)
    ax2.legend(fontsize=7, frameon=False)
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, lw=0.5)
    if synthetic:
        fig.suptitle("SYNTHETIC DRY RUN - not for the report", color="#c0392b", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_rsvd_vs_exact.pdf")
    fig.savefig(out / "fig_rsvd_vs_exact.png", dpi=200)
    print(f"Figure: {out / 'fig_rsvd_vs_exact.pdf'}")


if __name__ == "__main__":
    main()
