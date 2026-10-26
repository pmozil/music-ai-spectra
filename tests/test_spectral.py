import numpy as np
from music_svd_detection.spectral import effective_rank, principal_angle_features, singular_value_features
from music_svd_detection.shared_basis import randomized_shared_svd_basis, shared_basis_clip_features


def test_effective_rank_extremes():
    assert abs(effective_rank(np.array([1.0, 0.0, 0.0])) - 1.0) < 1e-8
    assert abs(effective_rank(np.ones(4)) - 4.0) < 1e-8


def test_svd_features_and_angles_finite():
    rng = np.random.default_rng(10)
    # Low-rank-ish 40-frequency spectrogram with sufficient time for several windows.
    x = rng.normal(size=(40, 160)).astype(np.float32)
    feat = singular_value_features(x, top_k=5)
    angle = principal_angle_features(x, top_k=4, window_frames=32, step_frames=16)
    assert 1 <= feat["effective_rank"] <= min(x.shape)
    assert 0 <= feat["topk_energy_fraction"] <= 1
    assert 0 <= angle["subspace_stability"] <= 1
    assert angle["n_window_pairs"] > 0


def test_streaming_shared_basis_captures_rank_one():
    rng = np.random.default_rng(42)
    u = rng.normal(size=(20, 1))
    blocks = [u @ rng.normal(size=(1, 15)), u @ rng.normal(size=(1, 23)), u @ rng.normal(size=(1, 11))]
    factory = lambda: iter(blocks)
    basis, values = randomized_shared_svd_basis(factory, rank=1, oversample=1, n_power_iterations=1, seed=3)
    assert basis.shape == (20, 1)
    assert values.shape == (1,)
    projection = shared_basis_clip_features(blocks[0], basis)
    assert projection["shared_basis_energy_fraction"] > 0.999
