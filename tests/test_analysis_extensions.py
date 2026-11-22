import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_exact_randomized as compare  # noqa: E402
import generator_report  # noqa: E402
import rank_sweep  # noqa: E402
from music_svd_detection.spectral import SpectralConfig  # noqa: E402

FEATS = compare.BASE_FEATURES


def test_rank_sweep_matches_compare_script_features():
    rng = np.random.default_rng(1)
    # full-rank (like real spectrograms) with a dominant low-rank part; F > window width
    lm = (rng.normal(size=(100, 10)) @ rng.normal(size=(10, 300)) + rng.normal(size=(100, 300))).astype(np.float32)
    cfg = SpectralConfig(subspace_window_frames=64, subspace_step_frames=32)
    ranks = [5, 20, 80]
    rows = rank_sweep.clip_features(lm, cfg, ranks, seed=7)
    for rec in rows:
        ref, _ = compare.features_for(lm, cfg, rec["rank"], rec["method"], 7)
        for f in FEATS:
            # k >= window: exact window basis instead of randomized_svd; same subspace,
            # angles agree to ~0.01 degrees (arccos near 0 amplifies rounding)
            tol = 0.05 if rec["rank"] >= 64 and rec["method"] == "randomized" else 1e-4
            assert np.isclose(rec[f], ref[f], rtol=1e-4, atol=tol), (rec["rank"], rec["method"], f)
    # randomized effective rank only sees the top k singular values
    for rec in rows:
        if rec["method"] == "randomized":
            assert rec["effective_rank"] <= rec["rank"] + 1e-9


def test_breakdown_pairs_each_generator_with_its_humans():
    test = pd.DataFrame({
        "ytid": ["a", "a", "b", "b", "c", "c"],
        "label": [0, 1, 0, 1, 0, 1],
        "generator": ["human_source", "g1", "human_source", "g2", "human_source", "g1"],
    })
    prob = np.array([0.1, 0.9, 0.2, 0.4, 0.6, 0.7])
    overall, rows = generator_report.breakdown("m", test, prob, (prob >= 0.5).astype(int))
    by = {r["generator"]: r for r in rows}
    assert overall["n_test"] == 6
    assert by["g1"]["n_test_pairs"] == 2 and by["g2"]["n_test_pairs"] == 1
    assert np.isclose(by["g1"]["accuracy"], 0.75)  # a: both right, c: human wrong
    assert np.isclose(by["g2"]["accuracy"], 0.5)   # b: AI missed


def test_bandwidth_filter_drops_whole_pairs():
    df = pd.DataFrame({"sample_id": ["h1", "f1", "h2", "f2"], "ytid": ["1", "1", "2", "2"],
                       "label": [0, 1, 0, 1]})
    bw = pd.Series({"h1": 15000.0, "f1": 3000.0, "h2": 15000.0, "f2": 7750.0})
    kept = generator_report.apply_bandwidth_filter(df, bw, 6000)
    assert list(kept["sample_id"]) == ["h2", "f2"]
