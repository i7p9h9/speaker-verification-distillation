from __future__ import annotations

import typing as tp

if tp.TYPE_CHECKING:
    from ..labeled import LabeledAggregatedDataset


def _get_direct_child_paths(dataset: "LabeledAggregatedDataset") -> tp.List[str]:
    """Return ``"root/child_name"`` paths for each direct child of *dataset*."""
    return [f"{dataset.name}/{wd.name}" for wd in dataset.weighted]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class WeightingStrategy:
    """
    Abstract base class for dataset sampling weight strategies.

    A strategy owns a reference to a :class:`LabeledAggregatedDataset` and
    is responsible for calling ``dataset.set_weights()`` in response to
    training signals.

    Subclasses must implement :meth:`update`.  All other methods have sensible
    defaults but may be overridden.

    Parameters
    ----------
    dataset:
        The dataset whose weights this strategy manages.
    """

    def __init__(self, dataset: LabeledAggregatedDataset) -> None:
        self._dataset = dataset
        self._keys: tp.List[str] = list(dataset.get_weights().keys())

    @property
    def keys(self) -> tp.List[str]:
        """Ordered list of source paths the strategy operates on."""
        return list(self._keys)

    @property
    def dataset(self) -> LabeledAggregatedDataset:
        return self._dataset

    def update(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        """
        Receive a training signal and update dataset weights accordingly.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def get_weights(self) -> tp.Dict[str, float]:
        """Current normalized weights at the source level."""
        raise NotImplementedError

    def state_dict(self) -> tp.Dict[str, tp.Any]:
        """Return serialisable strategy state for checkpointing."""
        raise NotImplementedError

    def load_state_dict(self, state: tp.Dict[str, tp.Any]) -> None:
        """Restore strategy state from a checkpoint."""
        raise NotImplementedError
