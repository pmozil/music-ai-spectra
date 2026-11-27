"""Small multiprocessing helpers used by the command-line analysis scripts."""

from __future__ import annotations

# Keep the controller alive for the lifetime of each worker process.  Without
# this, a process pool can start several BLAS/OpenMP thread pools per worker and
# become slower (and much more memory hungry) than serial execution.
_THREAD_LIMITER = None


def limit_worker_threads() -> None:
    """Limit BLAS/OpenMP to one thread inside a multiprocessing worker."""
    global _THREAD_LIMITER
    try:
        from threadpoolctl import threadpool_limits

        _THREAD_LIMITER = threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover - sklearn normally installs this
        _THREAD_LIMITER = None
