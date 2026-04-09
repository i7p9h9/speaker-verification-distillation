from __future__ import annotations

import typing as tp

import torch
from torch.utils.data import Dataset

from ._registry import _NameRegistry
from ._shared_weights import SharedWeights
from ._type import WeightedDataset


class AggregatedDataset(Dataset):
    """
    Aggregator dataset that samples from multiple datasets
    with configurable sampling probabilities.

    Multiprocessing safety
    ----------------------
    Sampling weights are stored in :class:`SharedWeights`, which uses
    ``torch.Tensor.share_memory_()`` to place tensors in OS shared memory.
    This means:

    * DataLoader worker processes forked **after** ``__init__`` see weight
      updates made by the main process immediately — no copies, no IPC lag.
    * ``persistent_workers=True`` is fully supported: the shared tensors
      persist across batches, and workers always read the latest weights.
    * Weight updates (via :meth:`set_weights`) are protected by a
      ``multiprocessing.Lock`` so concurrent writes from the main process
      are safe.

    Accepts either:
    - A list of Dataset objects → equal probability (1 / N) per dataset
    - A list of WeightedDataset dataclasses where:
        - All weights are None  → weights proportional to ``len(dataset)``
        - All weights are set (positive floats) → used directly (normalized)

    In all cases weights are resolved into a unified list of WeightedDataset
    with explicit normalized floats before being written to shared memory.

    Parameters
    ----------
    sources:
        Non-empty list of Dataset or WeightedDataset objects.
    name:
        Optional unique name for this dataset. Auto-generated if not provided.
    """

    def __init__(
        self,
        sources: tp.Union[tp.List[Dataset], tp.List[WeightedDataset]],
        name: tp.Optional[str] = None,
    ) -> None:
        assert len(sources) > 0, "At least one dataset must be provided"

        self._name: str = _NameRegistry.register(name, type(self).__name__)
        self._weighted: tp.List[WeightedDataset] = self._normalize(sources)
        self._datasets: tp.List[Dataset] = [s.dataset for s in self._weighted]
        self._lengths: tp.List[int] = [len(ds) for ds in self._datasets]  # type: ignore[arg-type]
        self._total: int = sum(self._lengths)

        # Allocate shared memory BEFORE any DataLoader fork.
        # All worker processes forked later will share the same physical pages.
        self._shared = SharedWeights([float(s.weight) for s in self._weighted])  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Name
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Normalization pipeline
    # ------------------------------------------------------------------

    @property
    def cum_probs(self):
        return self._shared.cum_probs

    @staticmethod
    def _normalize(
        sources: tp.Union[tp.List[Dataset], tp.List[WeightedDataset]]
    ) -> tp.List[WeightedDataset]:
        wrapped = AggregatedDataset._wrap(sources)
        return AggregatedDataset._resolve_weights(wrapped)

    @staticmethod
    def _wrap(
        sources: tp.Union[tp.List[Dataset], tp.List[WeightedDataset]]
    ) -> tp.List[WeightedDataset]:
        if not isinstance(sources[0], WeightedDataset):
            return [WeightedDataset(dataset=ds, weight=None) for ds in sources]  # type: ignore[arg-type]

        weighted: tp.List[WeightedDataset] = sources  # type: ignore[assignment]
        weights_none = [s.weight is None for s in weighted]
        assert all(weights_none) or not any(weights_none), (
            "Either all weights must be None or all must be set"
        )
        if not any(weights_none):
            assert all(s.weight > 0 for s in weighted), (  # type: ignore[operator]
                "All weights must be positive"
            )
        return weighted

    @staticmethod
    def _resolve_weights(sources: tp.List[WeightedDataset]) -> tp.List[WeightedDataset]:
        if all(s.weight is None for s in sources):
            lengths = [len(s.dataset) for s in sources]  # type: ignore[arg-type]
            total = sum(lengths)
            resolved = [ln / total for ln in lengths]
        else:
            total_w = sum(s.weight for s in sources)  # type: ignore[misc]
            resolved = [s.weight / total_w for s in sources]  # type: ignore[operator]

        result: tp.List[WeightedDataset] = []
        for s, w in zip(sources, resolved):
            child_name = (
                s.name
                if s.name is not None
                else getattr(s.dataset, "name", None)
            )
            if child_name is None:
                child_name = _NameRegistry.register(None, type(s.dataset).__name__)
            result.append(WeightedDataset(dataset=s.dataset, weight=w, name=child_name))
        return result

    # ------------------------------------------------------------------
    # Internal: sync WeightedDataset list → SharedWeights
    # ------------------------------------------------------------------

    def _flush_to_shared(self) -> None:
        """
        Push current ``_weighted`` float weights into shared memory.

        Must be called after every in-place mutation of ``_weighted[i].weight``.
        Worker processes will see updated values on their very next
        ``__getitem__`` call.
        """
        self._shared.update([float(wd.weight) for wd in self._weighted])  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Weight API
    # ------------------------------------------------------------------

    def get_weights(self) -> tp.Dict[str, float]:
        """
        Return a flat dict of all (normalized) weights keyed by path.

        Paths use ``'/'`` as separator, e.g.
        ``"AggregatedDataset_0/TensorDataset_1"``.
        Inner :class:`AggregatedDataset` children are expanded recursively.
        """
        out: tp.Dict[str, float] = {}
        self._collect_weights(prefix=self._name, out=out)
        return out

    def _collect_weights(self, prefix: str, out: tp.Dict[str, float]) -> None:
        for wd in self._weighted:
            child_name = wd.name or ""
            path = f"{prefix}/{child_name}"
            if isinstance(wd.dataset, AggregatedDataset):
                child_weights = wd.dataset.get_weights()
                for sub_path, sub_w in child_weights.items():
                    rel = sub_path[len(wd.dataset.name):]
                    out[f"{path}{rel}"] = float(wd.weight) * sub_w  # type: ignore[operator]
            else:
                out[path] = float(wd.weight)  # type: ignore[arg-type]

    def set_weights(self, weights: tp.Dict[str, float]) -> None:
        """
        Update weights from a flat path-keyed dict.

        All values must be positive floats (not None).
        All keys in *weights* must correspond to existing paths.
        Weights at each level are re-normalized independently after update.

        The new weights are immediately written to shared memory so that
        DataLoader worker processes read the updated probabilities on their
        very next ``__getitem__`` call — no restart required.

        Example
        -------
        ::

            ds.set_weights({
                "root/source_a/sub_1": 2.0,
                "root/source_a/sub_2": 1.0,
                "root/source_b":       3.0,
            })
        """
        assert all(v is not None and v > 0 for v in weights.values()), (
            "All weight values must be positive floats (not None, not <= 0)"
        )
        self._apply_weights(prefix=self._name, weights=weights)

    def _apply_weights(self, prefix: str, weights: tp.Dict[str, float]) -> None:
        """
        Recursively apply *weights* starting from *prefix*.

        For each child of this node:
          (a) direct key ``prefix/child`` present in *weights*
              → use that value as the new raw weight for this child.
          (b) descendant keys ``prefix/child/...`` present in *weights*
              → sum their values as the proportional weight for this child,
                then recurse into the child for fine-grained update.

        After collecting raw weights for every addressed child at this level,
        they are re-normalized to sum=1 and written to ``_weighted`` and
        :class:`SharedWeights` atomically.
        Unaddressed children keep their current weight.
        """
        raw: tp.Dict[int, float] = {}  # index → new unnormalized weight

        for i, wd in enumerate(self._weighted):
            child_name = wd.name or ""
            direct_key = f"{prefix}/{child_name}"

            # (a) direct child key
            if direct_key in weights:
                raw[i] = weights[direct_key]

            # (b) descendant keys
            desc_keys = {
                k: v for k, v in weights.items()
                if k.startswith(direct_key + "/")
            }
            if desc_keys:
                if i not in raw:
                    raw[i] = sum(desc_keys.values())
                if isinstance(wd.dataset, AggregatedDataset):
                    wd.dataset._apply_weights(direct_key, desc_keys)

        if not raw:
            return  # nothing addressed at this level

        current = [float(wd.weight) for wd in self._weighted]  # type: ignore[arg-type]
        for i, v in raw.items():
            current[i] = v

        assert all(w > 0 for w in current), (
            "After update all weights must remain positive"
        )
        total = sum(current)
        normalized = [w / total for w in current]

        for i, wd in enumerate(self._weighted):
            wd.weight = normalized[i]

        # Single atomic write to shared memory — visible to all workers
        self._flush_to_shared()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index: int) -> tp.Any:
        """
        Select a dataset proportionally to its weight (read from shared memory),
        then pick a uniformly random item from it.

        Workers forked by DataLoader read ``_shared.cum_probs`` directly from
        OS shared memory, so they always see the latest weights set by the
        main process via :meth:`set_weights`.
        """
        r = torch.rand(1).item()
        # Read cumulative probs from shared memory (lock-free on x86/ARM)
        cum_probs = self.cum_probs
        dataset_idx = int(
            torch.searchsorted(cum_probs, torch.tensor(r)).clamp(
                0, len(self._datasets) - 1
            )
        )
        inner_idx = int(
            torch.randint(0, self._lengths[dataset_idx], (1,)).item()
        )
        return self._datasets[dataset_idx][inner_idx]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def datasets(self) -> tp.List[Dataset]:
        return self._datasets

    @property
    def weighted(self) -> tp.List[WeightedDataset]:
        return self._weighted

    @property
    def probabilities(self) -> tp.List[float]:
        """Current normalized weights read from shared memory."""
        return self._shared.weights.tolist()

    @property
    def shared_weights(self) -> SharedWeights:
        """Direct access to the underlying :class:`SharedWeights` object."""
        return self._shared

    def __repr__(self) -> str:
        parts = [
            f"  [{i}] {wd.name}(len={self._lengths[i]}, p={wd.weight:.4f})"
            for i, wd in enumerate(self._weighted)
        ]
        return (
            f"{type(self).__name__}(name={self._name!r},\n"
            + "\n".join(parts)
            + "\n)"
        )
