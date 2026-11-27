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


def write_stacked_matrix(
    block_factory: BlockFactory,
    path,
    dtype=np.float32,
):
    """Materialize A=[A_1 ... A_N] as a Fortran-order disk-backed matrix.

    The factory is read twice, so callers can regenerate spectrograms without
    retaining the full dataset in RAM. The returned memmap owns the file at path.
    """
    from pathlib import Path

    path = Path(path)
    first = next(iter(block_factory()), None)
    if first is None:
        raise ValueError("No spectrogram blocks supplied")
    f = int(np.asarray(first).shape[0])
    total_t = 0
    count = 0
    for block in block_factory():
        a = np.asarray(block)
        if a.ndim != 2 or a.shape[0] != f or a.shape[1] == 0:
            raise ValueError("Blocks must be non-empty 2-D arrays with equal row count")
        total_t += int(a.shape[1])
        count += 1
    if total_t == 0:
        raise ValueError("No frames supplied")
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.memmap(path, mode="w+", dtype=dtype, shape=(f, total_t), order="F")
    cursor = 0
    written = 0
    for block in block_factory():
        a = np.asarray(block, dtype=dtype)
        if a.ndim != 2 or a.shape[0] != f:
            raise ValueError("Block shape changed between passes")
        end = cursor + a.shape[1]
        if end > total_t:
            raise ValueError("Block factory yielded different data on second pass")
        matrix[:, cursor:end] = a
        cursor = end
        written += 1
    if cursor != total_t or written != count:
        del matrix
        path.unlink(missing_ok=True)
        raise ValueError("Block factory yielded a different number of frames on second pass")
    matrix.flush()
    return matrix


def randomized_svd_matrix(
    matrix: np.ndarray,
    rank: int = 16,
    oversample: int = 8,
    n_power_iterations: int = 2,
    seed: int = 20261026,
    block_columns: int = 65_536,
) -> tuple[np.ndarray, np.ndarray]:
    """Truncated randomized SVD of an explicit stacked matrix A (F x T).

    A may be an ndarray or memmap. Products are accumulated by column blocks,
    avoiding a large right sketch or a dense copy of A. Returns left singular
    vectors and singular values; right vectors are not materialized.
    """
    a = matrix
    if getattr(a, "ndim", None) != 2 or min(a.shape) < 1:
        raise ValueError("matrix must be a non-empty 2-D array")
    f, t = map(int, a.shape)
    if rank < 1 or oversample < 0 or n_power_iterations < 0 or block_columns < 1:
        raise ValueError("Invalid randomized SVD parameter")
    ell = min(f, int(rank) + int(oversample))
    rng = np.random.default_rng(seed)
    y = np.zeros((f, ell), dtype=np.float64)
    for start in range(0, t, block_columns):
        end = min(t, start + block_columns)
        ab = np.asarray(a[:, start:end], dtype=np.float64)
        omega = rng.standard_normal((end - start, ell))
        y += ab @ omega
    q, _ = np.linalg.qr(y, mode="reduced")
    for _ in range(n_power_iterations):
        z = np.zeros_like(q)
        for start in range(0, t, block_columns):
            end = min(t, start + block_columns)
            ab = np.asarray(a[:, start:end], dtype=np.float64)
            z += ab @ (ab.T @ q)
        q, _ = np.linalg.qr(z, mode="reduced")
    covariance = np.zeros((q.shape[1], q.shape[1]), dtype=np.float64)
    for start in range(0, t, block_columns):
        end = min(t, start + block_columns)
        b = q.T @ np.asarray(a[:, start:end], dtype=np.float64)
        covariance += b @ b.T
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    order = np.argsort(eigenvalues)[::-1][: min(rank, eigenvalues.size)]
    values = np.sqrt(np.maximum(eigenvalues[order], 0.0))
    basis = q @ eigenvectors[:, order]
    return basis.astype(np.float32), values


def compare_class_bases(
    human_basis: np.ndarray,
    ai_basis: np.ndarray,
    human_singular_values: np.ndarray | None = None,
    ai_singular_values: np.ndarray | None = None,
) -> tuple[dict[str, float | int], list[dict[str, float | int]], list[dict[str, float | int]]]:
    """Compare left singular-vector spaces learned from human and AI stacks.

    Principal angles compare the retained subspaces and are invariant to signs
    and rotations within nearly-degenerate singular subspaces. A Hungarian
    assignment on absolute dot products additionally gives an interpretable,
    sign-invariant mode-to-mode matching (but is less stable near repeated
    singular values).
    """
    from scipy.optimize import linear_sum_assignment

    h = np.asarray(human_basis, dtype=np.float64)
    a = np.asarray(ai_basis, dtype=np.float64)
    if h.ndim != 2 or a.ndim != 2 or h.shape[0] != a.shape[0]:
        raise ValueError("Bases must be 2-D and share the same feature dimension")
    if h.shape[1] < 1 or a.shape[1] < 1:
        raise ValueError("Both bases must contain at least one vector")
    # QR removes small numerical loss of orthogonality from randomized SVD.
    h, _ = np.linalg.qr(h, mode="reduced")
    a, _ = np.linalg.qr(a, mode="reduced")
    overlap = h.T @ a
    principal_cosines = np.linalg.svd(overlap, compute_uv=False)
    principal_cosines = np.clip(principal_cosines, 0.0, 1.0)
    principal_angles = np.degrees(np.arccos(principal_cosines))

    rows, cols = linear_sum_assignment(-np.abs(overlap))
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    matched = np.abs(overlap[rows, cols])

    weighted_match = float("nan")
    if human_singular_values is not None and ai_singular_values is not None:
        hs = np.asarray(human_singular_values, dtype=np.float64).reshape(-1)
        av = np.asarray(ai_singular_values, dtype=np.float64).reshape(-1)
        if hs.size >= h.shape[1] and av.size >= a.shape[1] and np.all(hs[:h.shape[1]] >= 0) and np.all(av[:a.shape[1]] >= 0):
            hw = hs[rows] ** 2
            aw = av[cols] ** 2
            w = np.sqrt((hw / hw.sum()) * (aw / aw.sum())) if hw.sum() and aw.sum() else np.zeros_like(hw)
            if w.sum() > 0:
                weighted_match = float(np.sum(w * matched) / w.sum())

    chordal_sq = float(np.sum(1.0 - principal_cosines ** 2))
    summary: dict[str, float | int] = {
        "human_rank": int(h.shape[1]),
        "ai_rank": int(a.shape[1]),
        "shared_rank": int(principal_cosines.size),
        "mean_principal_angle_deg": float(np.mean(principal_angles)),
        "median_principal_angle_deg": float(np.median(principal_angles)),
        "max_principal_angle_deg": float(np.max(principal_angles)),
        "mean_squared_principal_cosine": float(np.mean(principal_cosines ** 2)),
        "normalized_chordal_distance": float(np.sqrt(chordal_sq / principal_cosines.size)),
        "mean_matched_abs_cosine": float(np.mean(matched)),
        "median_matched_abs_cosine": float(np.median(matched)),
        "energy_weighted_matched_abs_cosine": weighted_match,
    }
    matched_rows = [
        {
            "human_mode": int(i + 1),
            "ai_mode": int(j + 1),
            "absolute_cosine_similarity": float(value),
        }
        for i, j, value in zip(rows, cols, matched)
    ]
    angle_rows = [
        {"principal_component": int(i + 1), "cosine": float(c), "angle_deg": float(theta)}
        for i, (c, theta) in enumerate(zip(principal_cosines, principal_angles))
    ]
    return summary, matched_rows, angle_rows
