"""STFT log-magnitude features and SVD/subspace diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft, welch


@dataclass(frozen=True)
class SpectralConfig:
    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    log_epsilon: float = 1e-8
    local_top_k: int = 8
    subspace_window_frames: int = 64
    subspace_step_frames: int = 32
    max_freq_hz: float = 0.0


def log_magnitude_spectrogram(
    audio: np.ndarray, sample_rate: int, config: SpectralConfig | None = None
) -> np.ndarray:
    """Return centered log-magnitude STFT as an F x T float32 matrix."""
    cfg = config or SpectralConfig()
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != cfg.sample_rate:
        raise ValueError(f"Expected {cfg.sample_rate} Hz audio, got {sample_rate} Hz")
    if x.size < cfg.n_fft:
        x = np.pad(x, (0, cfg.n_fft - x.size))
    freqs, _, z = stft(
        x, fs=sample_rate, window="hann", nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - cfg.hop_length, nfft=cfg.n_fft,
        boundary=None, padded=False, return_onesided=True,
    )
    if cfg.max_freq_hz and cfg.max_freq_hz > 0:
        z = z[freqs <= cfg.max_freq_hz, :]
    mag = np.abs(z).astype(np.float64, copy=False)
    logmag = np.log(np.maximum(mag, cfg.log_epsilon))
    logmag -= float(logmag.mean())
    return np.asarray(logmag, dtype=np.float32)


def effective_rank(singular_values: np.ndarray) -> float:
    """Entropy effective rank exp(H(p)), p_i=sigma_i/sum_j sigma_j."""
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[np.isfinite(s) & (s > 0)]
    if not s.size:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def _distribution_features(weights: np.ndarray, prefix: str) -> dict[str, float]:
    """Sign-invariant centroid, spread and normalized entropy of squared loadings."""
    p = np.asarray(weights, dtype=np.float64).reshape(-1)
    p = np.maximum(p, 0.0)
    total = float(p.sum())
    if not p.size or total <= 0:
        return {f"{prefix}_{name}": 0.0 for name in ("centroid", "spread", "entropy")}
    p /= total
    axis = np.linspace(0.0, 1.0, p.size) if p.size > 1 else np.zeros(1)
    centroid = float(np.sum(p * axis))
    spread = float(np.sqrt(np.sum(p * (axis - centroid) ** 2)))
    positive = p[p > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    entropy /= float(np.log(p.size)) if p.size > 1 else 1.0
    return {
        f"{prefix}_centroid": centroid,
        f"{prefix}_spread": spread,
        f"{prefix}_entropy": float(np.clip(entropy, 0.0, 1.0)),
    }


def _svd_value_features(s: np.ndarray, top_k: int) -> dict[str, float]:
    energy = float(np.sum(s * s))
    k = min(max(1, int(top_k)), s.size)
    p2 = (s * s / energy) if energy > 0 else np.zeros_like(s)
    out: dict[str, float] = {
        "effective_rank": effective_rank(s),
        "n_singular_values": float(s.size),
        "sigma1_over_fro": float(s[0] / np.sqrt(energy)) if energy > 0 else 0.0,
        "topk_energy_fraction": float(np.sum(s[:k] ** 2) / energy) if energy > 0 else 0.0,
        "spectral_entropy_bits": float(-np.sum(p2[p2 > 0] * np.log2(p2[p2 > 0]))),
    }
    for i in range(min(8, s.size)):
        out[f"sigma_ratio_{i+1}"] = float(s[i] / s[0]) if s[0] > 0 else 0.0
    return out


def _svd_vector_features(u: np.ndarray, vt: np.ndarray, top_k: int = 4) -> dict[str, float]:
    """Describe U/V loadings without their arbitrary signs or rotations in a subspace."""
    k = min(max(1, int(top_k)), u.shape[1], vt.shape[0])
    out: dict[str, float] = {}
    for i in range(k):
        out.update(_distribution_features(u[:, i] ** 2, f"u_mode{i+1}_freq"))
        out.update(_distribution_features(vt[i, :] ** 2, f"v_mode{i+1}_time"))
    # Diagonals of the two projectors are invariant to sign flips and rotations
    # within the retained singular subspace.
    out.update(_distribution_features(np.sum(u[:, :k] ** 2, axis=1), "u_subspace_freq"))
    out.update(_distribution_features(np.sum(vt[:k, :] ** 2, axis=0), "v_subspace_time"))
    return out


def svd_clip_features(
    logmag: np.ndarray, top_k: int = 8, vector_modes: int = 4
) -> dict[str, float]:
    """Compute singular-value and singular-vector features from one exact SVD."""
    a = np.asarray(logmag, dtype=np.float64)
    if a.ndim != 2 or not a.size:
        raise ValueError("logmag must be a non-empty 2-D matrix")
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    out = _svd_value_features(s, top_k)
    out.update(_svd_vector_features(u, vt, vector_modes))
    return out


def singular_value_features(logmag: np.ndarray, top_k: int = 8) -> dict[str, float]:
    """Backwards-compatible singular-value features (without vector descriptors)."""
    a = np.asarray(logmag, dtype=np.float64)
    if a.ndim != 2 or not a.size:
        raise ValueError("logmag must be a non-empty 2-D matrix")
    s = np.linalg.svd(a, full_matrices=False, compute_uv=False)
    return _svd_value_features(s, top_k)


def singular_vector_features(logmag: np.ndarray, top_k: int = 4) -> dict[str, float]:
    """Sign-invariant descriptors of leading left (frequency) and right (time) vectors."""
    a = np.asarray(logmag, dtype=np.float64)
    if a.ndim != 2 or not a.size:
        raise ValueError("logmag must be a non-empty 2-D matrix")
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    return _svd_vector_features(u, vt, top_k)


def principal_angle_features(
    logmag: np.ndarray, top_k: int = 8, window_frames: int = 64, step_frames: int = 32
) -> dict[str, float]:
    """Adjacent-window principal-angle summaries for leading left singular subspaces."""
    a = np.asarray(logmag, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("logmag must be a 2-D matrix")
    f, t = a.shape
    width = min(int(window_frames), t)
    starts = list(range(0, t - width + 1, max(1, int(step_frames)))) if width >= 2 else []
    if len(starts) < 2:
        return {
            "principal_angle_mean_deg": float("nan"),
            "principal_angle_std_deg": float("nan"),
            "subspace_stability": float("nan"),
            "n_window_pairs": 0.0,
        }
    k = min(max(1, int(top_k)), f, width)
    bases = []
    for start in starts:
        u, _, _ = np.linalg.svd(a[:, start : start + width], full_matrices=False)
        bases.append(u[:, :k])
    angles = []
    for u0, u1 in zip(bases[:-1], bases[1:]):
        cosines = np.linalg.svd(u0.T @ u1, full_matrices=False, compute_uv=False)
        angles.append(np.arccos(np.clip(cosines, 0.0, 1.0)))
    theta = np.concatenate(angles)
    mean_angle = float(theta.mean())
    return {
        "principal_angle_mean_deg": float(np.degrees(mean_angle)),
        "principal_angle_std_deg": float(np.degrees(theta.std())),
        "subspace_stability": float(np.clip(1.0 - mean_angle / (np.pi / 2), 0.0, 1.0)),
        "n_window_pairs": float(len(starts) - 1),
    }


def raw_bandwidth_cutoff_hz(
    audio: np.ndarray, sample_rate: int, band_hz: float = 250.0, relative_db: float = -60.0
) -> float:
    """Return upper edge of highest PSD band within relative_db of the strongest band."""
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    nyquist = float(sample_rate) / 2.0
    if x.size < 16 or sample_rate <= 0:
        return nyquist
    freq, psd = welch(x, fs=sample_rate, nperseg=min(8192, x.size), scaling="density")
    if not np.any(psd > 0):
        return 0.0
    edges = np.arange(0.0, nyquist + band_hz, band_hz)
    means, uppers = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freq >= lo) & (freq < hi)
        if np.any(mask):
            means.append(float(np.mean(psd[mask])))
            uppers.append(float(min(hi, nyquist)))
    if not means:
        return nyquist
    b = np.asarray(means)
    active = np.flatnonzero(b >= float(np.max(b)) * 10.0 ** (relative_db / 10.0))
    return float(uppers[int(active[-1])]) if active.size else 0.0


def joint_pair_svd_features(
    human_logmag: np.ndarray, ai_logmag: np.ndarray, top_k: int = 8
) -> tuple[dict[str, float], dict[str, float]]:
    """One SVD of [L_human L_AI], returning side-specific descriptors for each clip.

    Each side's returned features use its own block of each joint right singular
    vector. The shared singular spectrum is included for context; block energy
    shares distinguish which side contributes to each joint mode.
    """
    human = np.asarray(human_logmag, dtype=np.float64)
    ai = np.asarray(ai_logmag, dtype=np.float64)
    if human.ndim != 2 or ai.ndim != 2 or not human.size or not ai.size:
        raise ValueError("Both pair members must be non-empty 2-D matrices")
    if human.shape[0] != ai.shape[0]:
        raise ValueError("Paired spectrograms must have the same frequency dimension")
    joint = np.concatenate((human, ai), axis=1)
    u, s, vt = np.linalg.svd(joint, full_matrices=False)
    k = min(max(1, int(top_k)), s.size)
    h_share = np.sum(vt[:k, : human.shape[1]] ** 2, axis=1)
    a_share = np.sum(vt[:k, human.shape[1] :] ** 2, axis=1)
    weights = s[:k] ** 2
    total_weight = float(weights.sum())
    h_energy = float(np.sum(weights * h_share) / total_weight) if total_weight else 0.0
    a_energy = float(np.sum(weights * a_share) / total_weight) if total_weight else 0.0

    def side_features(block: np.ndarray, shares: np.ndarray, side: str) -> dict[str, float]:
        denom = float(np.sum(block * block))
        projected = float(np.sum((u[:, :k].T @ block) ** 2))
        captured = float(np.clip(projected / denom, 0.0, 1.0)) if denom > 0 else 0.0
        result: dict[str, float] = {
            "pair_side_projection_energy": captured,
            "pair_side_relative_residual": float(np.sqrt(max(0.0, 1.0 - captured))),
            "pair_side_total_energy_fraction": float(
                denom / (np.sum(human * human) + np.sum(ai * ai))
            ) if denom > 0 else 0.0,
            "pair_side_weighted_mode_share": h_energy if side == "human" else a_energy,
            "pair_joint_topk_energy_fraction": float(
                np.sum(s[:k] ** 2) / np.sum(joint * joint)
            ) if np.sum(joint * joint) > 0 else 0.0,
        }
        for i in range(k):
            result[f"pair_joint_sigma_ratio_{i+1}"] = float(s[i] / s[0]) if s[0] else 0.0
            result[f"pair_side_mode_share_{i+1}"] = float(np.clip(shares[i], 0.0, 1.0))
        return result

    return side_features(human, h_share, "human"), side_features(ai, a_share, "ai")
