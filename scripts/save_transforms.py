#!/usr/bin/env python3
"""Save per-clip log-STFT and exact/randomized truncated SVD transforms."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from music_svd_detection.spectral import SpectralConfig, log_magnitude_spectrogram

def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_harmonized.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/default.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/processed/transforms"))
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="First N pairs; 0 means all")
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--save-float32", action="store_true")
    args = ap.parse_args()

    cfgd = json.loads(args.config.read_text())
    cfg = SpectralConfig(
        sample_rate=int(cfgd["sample_rate"]), n_fft=int(cfgd["n_fft"]),
        hop_length=int(cfgd["hop_length"]), log_epsilon=float(cfgd["log_epsilon"]),
        local_top_k=int(cfgd["local_top_k"]),
        subspace_window_frames=int(cfgd["subspace_window_frames"]),
        subspace_step_frames=int(cfgd["subspace_step_frames"]),
    )
    df = pd.read_csv(args.manifest)
    if "processing_status" in df:
        df = df[df["processing_status"].eq("ok")].copy()
    if args.limit:
        ids = df["ytid"].drop_duplicates().head(args.limit)
        df = df[df["ytid"].isin(ids)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Saving transforms"):
        path = Path(str(row.processed_path))
        out = args.output_dir / f"{safe_name(row.sample_id)}.npz"
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            if np.asarray(audio).ndim == 2:
                audio = np.asarray(audio).mean(axis=1)
            logmag = log_magnitude_spectrogram(audio, int(sr), cfg).astype(np.float64)
            k = min(args.rank, *logmag.shape)
            u, s, vh = np.linalg.svd(logmag, full_matrices=False)
            ue, se, vhe = u[:, :k], s[:k], vh[:k, :]
            ur, srnd, vhr = randomized_svd(
                logmag, n_components=k, n_iter=4, random_state=args.seed
            )
            arrays = dict(
                log_magnitude=logmag, U_exact=ue, S_exact=se, Vh_exact=vhe,
                U_randomized=ur, S_randomized=srnd, Vh_randomized=vhr,
                sample_id=np.asarray(str(row.sample_id)),
                ytid=np.asarray(str(row.ytid)),
                label=np.asarray(int(row.label)),
                generator=np.asarray(str(row.generator)),
                sample_rate=np.asarray(int(sr)),
            )
            if args.save_float32:
                for key, value in list(arrays.items()):
                    if isinstance(value, np.ndarray) and value.dtype.kind == "f":
                        arrays[key] = value.astype(np.float32)
            np.savez_compressed(out, **arrays)
            status.append({"sample_id": row.sample_id, "transform_path": str(out), "status": "ok"})
        except Exception as exc:
            status.append({"sample_id": row.sample_id, "transform_path": "", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    pd.DataFrame(status).to_csv(args.output_dir / "transform_status.csv", index=False)
    print(pd.Series([x["status"] for x in status]).value_counts().to_string())
    print(f"Transforms: {args.output_dir}")

if __name__ == "__main__":
    main()
