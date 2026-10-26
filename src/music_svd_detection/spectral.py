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


def log_magnitude_spectrogram(
    audio: np.ndarray, sample_rate: int, config: SpectralConfig | None = None
) -> np.ndarray:
    """Return centered log-magnitude STFT as an F x T float32 matrix"""
    cfg = config or SpectralConfig()
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != cfg.sample_rate:
        raise ValueError(f"Expected {cfg.sample_rate} Hz audio, got {sample_rate} Hz")

    if x.size < cfg.n_fft:
        x = np.pad(x, (0, cfg.n_fft - x.size))

    _, _, z = stft(
        x,
        fs=sample_rate,
        window="hann",
        nperseg=cfg.n_fft,
        noverlap=cfg.n_fft - cfg.hop_length,
        nfft=cfg.n_fft,
        boundary=None,
        padded=False,
        return_onesided=True,
    )
    mag = np.abs(z).astype(np.float64, copy=False)
    logmag = np.log(np.maximum(mag, cfg.log_epsilon))
    logmag -= float(logmag.mean())

    return np.asarray(logmag, dtype=np.float32)


def effective_rank(singular_values: np.ndarray) -> float:
    """Entropy effective rank exp(H(p)), with p_i=sigma_i/sum_j sigma_j.

    This uses the normalized-singular-value convention
    """
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[np.isfinite(s) & (s > 0)]
    if not s.size:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def singular_value_features(logmag: np.ndarray, top_k: int = 8) -> dict[str, float]:
    """Exact per-clip SVD; spectrogram matrices are intentionally small."""
    a = np.asarray(logmag, dtype=np.float64)
    if a.ndim != 2 or not a.size:
        raise ValueError("logmag must be a non-empty 2-D matrix")
    s = np.linalg.svd(a, full_matrices=False, compute_uv=False)
    energy = float(np.sum(s * s))
    k = min(max(1, int(top_k)), s.size)
    p2 = (s * s / energy) if energy > 0 else np.zeros_like(s)
    out: dict[str, float] = {
        "effective_rank": effective_rank(s),
        "n_singular_values": float(s.size),
        "sigma1_over_fro": float(s[0] / np.sqrt(energy)) if energy > 0 else 0.0,
        "topk_energy_fraction": (
            float(np.sum(s[:k] ** 2) / energy) if energy > 0 else 0.0
        ),
        "spectral_entropy_bits": float(-np.sum(p2[p2 > 0] * np.log2(p2[p2 > 0]))),
    }
    for i in range(min(8, s.size)):
        out[f"sigma_ratio_{i+1}"] = float(s[i] / s[0]) if s[0] > 0 else 0.0
    return out


def principal_angle_features(
    logmag: np.ndarray, top_k: int = 8, window_frames: int = 64, step_frames: int = 32
) -> dict[str, float]:
    """
    For each F x window_frames window j, U_j contains its leading k left
    singular vectors. cos(theta_i)=svdvals(U_j.T @ U_{j+1})_i, with theta_i
    in [0, pi/2]. Stability = 1 - mean(theta)/(pi/2), so 1 means stable and
    0 means maximally different. Aggregate all angles from adjacent windows.
    """
    a = np.asarray(logmag, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("logmag must be a 2-D matrix")

    f, t = a.shape
    width = min(int(window_frames), t)
    starts = (
        list(range(0, t - width + 1, max(1, int(step_frames)))) if width >= 2 else []
    )

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
    audio: np.ndarray,
    sample_rate: int,
    band_hz: float = 250.0,
    relative_db: float = -60.0,
) -> float:
    """
    Return the upper edge of the highest 250-Hz band whose mean PSD is within
    -60 dB of the strongest band. This is a provenance shortcut baseline, not
    a perceptual bandwidth estimate. Measure on raw decoded audio only.
    """
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
