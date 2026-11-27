import numpy as np

from music_svd_detection.spectral import (
    effective_rank,
    joint_pair_svd_features,
    principal_angle_features,
    singular_value_features,
    svd_clip_features,
)
from music_svd_detection.shared_basis import (
    randomized_shared_svd_basis,
    randomized_svd_matrix,
    shared_basis_clip_features,
    write_stacked_matrix,
)


def test_effective_rank_extremes():
    assert abs(effective_rank(np.array([1.0, 0.0, 0.0])) - 1.0) < 1e-8
    assert abs(effective_rank(np.ones(4)) - 4.0) < 1e-8


def test_svd_features_and_angles_finite():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(40, 160)).astype(np.float32)
    feat = singular_value_features(x, top_k=5)
    all_feat = svd_clip_features(x, top_k=5, vector_modes=4)
    angle = principal_angle_features(x, top_k=4, window_frames=32, step_frames=16)
    assert 1 <= feat["effective_rank"] <= min(x.shape)
    assert 0 <= feat["topk_energy_fraction"] <= 1
    assert 0 <= angle["subspace_stability"] <= 1
    assert angle["n_window_pairs"] > 0
    assert 0.0 <= all_feat["u_subspace_freq_entropy"] <= 1.0
    assert 0.0 <= all_feat["v_subspace_time_entropy"] <= 1.0
    assert 0.0 <= all_feat["u_mode1_freq_centroid"] <= 1.0


def test_vector_descriptors_are_sign_invariant_and_joint_svd_is_side_specific():
    rng = np.random.default_rng(22)
    shared = rng.normal(size=(24, 5))
    human = shared @ rng.normal(size=(5, 70)) + 0.2 * rng.normal(size=(24, 70))
    ai = shared @ rng.normal(size=(5, 70)) + 0.5 * rng.normal(size=(24, 70))
    h, a = joint_pair_svd_features(human, ai, top_k=6)
    assert len([k for k in h if k.startswith("pair_side_mode_share_")]) == 6
    assert 0 <= h["pair_side_projection_energy"] <= 1
    assert 0 <= a["pair_side_projection_energy"] <= 1
    assert np.isclose(h["pair_side_weighted_mode_share"] + a["pair_side_weighted_mode_share"], 1.0, atol=1e-5)
    h_swapped, a_swapped = joint_pair_svd_features(ai, human, top_k=6)
    for i in range(1, 7):
        assert np.isclose(h[f"pair_side_mode_share_{i}"], a_swapped[f"pair_side_mode_share_{i}"], atol=1e-6)
        assert np.isclose(a[f"pair_side_mode_share_{i}"], h_swapped[f"pair_side_mode_share_{i}"], atol=1e-6)


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


def test_randomized_svd_of_explicit_stacked_memmap(tmp_path):
    rng = np.random.default_rng(7)
    left, _ = np.linalg.qr(rng.standard_normal((32, 6)))
    right, _ = np.linalg.qr(rng.standard_normal((180, 6)))
    exact_values = np.array([10, 7, 4, 2, 1, 0.5])
    full = (left * exact_values) @ right.T
    blocks = [full[:, :61], full[:, 61:127], full[:, 127:]]
    path = tmp_path / "stacked.dat"
    matrix = write_stacked_matrix(lambda: iter(blocks), path)
    assert matrix.shape == full.shape
    basis, values = randomized_svd_matrix(
        matrix, rank=6, oversample=8, n_power_iterations=2, seed=4, block_columns=37
    )
    assert np.allclose(values, exact_values, rtol=1e-4, atol=1e-5)
    cosines = np.linalg.svd(left.T @ basis, compute_uv=False)
    assert np.min(cosines) > 0.999
    del matrix
