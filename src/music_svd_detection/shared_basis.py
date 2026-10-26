"""Streaming randomized SVD for a shared basis over stacked clip frames."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

BlockFactory = Callable[[], Iterable[np.ndarray]]


def randomized_shared_svd_basis(
    block_factory: BlockFactory,
    rank: int = 16,
    oversample: int = 8,
    n_power_iterations: int = 1,
    seed: int = 20261026,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate leading left singular vectors of A=[A_1 A_2 ...] without storing A.

    Each A_j is an F x T_j log-magnitude spectrogram; all have the same F.
    The restartable block_factory is iterated a few times to do a randomized range
    finder and small projected covariance eigendecomposition.

    Returns basis and approximate leading singular values.
    Use training clips only to avoid leakage.
    """
    if rank < 1 or oversample < 0 or n_power_iterations < 0:
        raise ValueError("rank >= 1, oversample >= 0, n_power_iterations >= 0 required")

    first = next(iter(block_factory()), None)

    if first is None:
        raise ValueError("No spectrogram blocks supplied")

    f = int(first.shape[0])
    ell = min(f, rank + oversample)

    rng = np.random.default_rng(seed)
    y = np.zeros((f, ell), dtype=np.float64)

    frame_count = 0

    # Construct sketch matrix
    for block in block_factory():
        a = np.asarray(block, dtype=np.float64)
        if a.shape[0] != f:
            raise ValueError("All blocks must have equal frequency dimension")
        y += a @ rng.standard_normal((a.shape[1], ell))
        frame_count += a.shape[1]

    if frame_count == 0:
        raise ValueError("No frames supplied")

    # Power iteration to better approximate subspace
    q, _ = np.linalg.qr(y, mode="reduced")
    for _ in range(n_power_iterations):
        z = np.zeros_like(q, dtype=np.float64)
        for block in block_factory():
            a = np.asarray(block, dtype=np.float64)
            z += a @ (a.T @ q)
        q, _ = np.linalg.qr(z, mode="reduced")

    c = np.zeros((q.shape[1], q.shape[1]), dtype=np.float64)
    for block in block_factory():
        a = np.asarray(block, dtype=np.float64)
        b = q.T @ a
        c += b @ b.T

    evals, evecs = np.linalg.eigh((c + c.T) * 0.5)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]
    k = min(rank, evecs.shape[1])

    return (q @ evecs[:, :k]).astype(np.float32), np.sqrt(evals[:k])


def shared_basis_clip_features(
    logmag: np.ndarray, basis: np.ndarray
) -> dict[str, float]:
    """Projection energy and relative residual against a basis learned on train data."""
    a, q = np.asarray(logmag, dtype=np.float64), np.asarray(basis, dtype=np.float64)
    if a.shape[0] != q.shape[0]:
        raise ValueError("Basis and spectrogram frequency dimensions differ")
    total = float(np.sum(a * a))
    captured = float(np.sum((q.T @ a) ** 2) / total) if total > 0 else 0.0
    captured = float(np.clip(captured, 0.0, 1.0))
    return {
        "shared_basis_energy_fraction": captured,
        "shared_basis_relative_residual": float(np.sqrt(max(0.0, 1.0 - captured))),
    }
