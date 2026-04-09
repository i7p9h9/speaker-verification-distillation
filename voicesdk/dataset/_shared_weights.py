from __future__ import annotations

import typing as tp

import torch
import torch.multiprocessing as mp


class SharedWeights:
    """
    Normalized weights + cumulative probabilities stored in torch shared memory.

    Safe to read from DataLoader worker processes forked after ``__init__``,
    including ``persistent_workers=True`` mode.

    The underlying tensors live in OS shared memory (via ``share_memory_()``),
    so all forked workers see writes made by the main process without any IPC.

    Write path (main process only)
    --------------------------------
    ``update()`` acquires a ``mp.Lock``, copies new values atomically, then
    releases the lock.  Workers never call ``update()``.

    Read path (worker processes)
    ----------------------------
    ``cum_probs`` and ``weights`` are read lock-free.  On x86/ARM, 64-bit
    aligned stores are atomic, so a worker will see either the old or the new
    tensor — never a torn intermediate state.

    Parameters
    ----------
    weights:
        Initial normalized weights (must sum to 1.0).
    """

    def __init__(self, weights: tp.List[float]) -> None:
        assert len(weights) > 0, "weights must be non-empty"

        # Allocate in shared memory BEFORE any DataLoader fork
        self._weights: torch.Tensor = (
            torch.tensor(weights, dtype=torch.float64).share_memory_()
        )
        self._cum_probs: torch.Tensor = (
            torch.cumsum(self._weights, dim=0).share_memory_()
        )
        # Lock protects concurrent writes from the main process only
        self._lock: mp.Lock = mp.Lock()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Write API (main process)
    # ------------------------------------------------------------------

    def update(self, weights: tp.List[float]) -> None:
        """
        Atomically replace weights and recompute cumulative probabilities.

        Parameters
        ----------
        weights:
            New normalized weights. Must have the same length as the
            original list passed to ``__init__``.
        """
        assert len(weights) == len(self._weights), (
            f"Expected {len(self._weights)} weights, got {len(weights)}"
        )
        t = torch.tensor(weights, dtype=torch.float64)
        with self._lock:
            self._weights.copy_(t)
            self._cum_probs.copy_(torch.cumsum(t, dim=0))

    # ------------------------------------------------------------------
    # Read API (worker processes)
    # ------------------------------------------------------------------

    @property
    def cum_probs(self) -> torch.Tensor:
        """Cumulative probability tensor (shared memory, read-only for workers)."""
        return self._cum_probs

    @property
    def weights(self) -> torch.Tensor:
        """Raw weight tensor (shared memory, read-only for workers)."""
        return self._weights

    def __len__(self) -> int:
        return len(self._weights)

    def __repr__(self) -> str:
        w = self._weights.tolist()
        return f"SharedWeights([{', '.join(f'{v:.4f}' for v in w)}])"
