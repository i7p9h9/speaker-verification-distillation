"""Core dataflow containers and abstract augmentation base class."""

import abc
import dataclasses
import typing as tp

import torch


@dataclasses.dataclass
class PersistEntry:
    """Snapshot of one stage's config and/or output signal for replay or debugging.

    Attributes:
        config: Sampled augmentation parameters. None if not persisted.
        signal: Output signal snapshot at this stage. None if not persisted.
    """

    config: tp.Optional["AugConfig"] = None
    signal: tp.Optional[torch.Tensor] = None


@dataclasses.dataclass
class AugConfig:
    """Base dataclass for all augmentation parameter configs.

    All concrete configs must inherit from this class.
    """


@dataclasses.dataclass
class DataSample:
    """Container for a single audio sample flowing through the augmentation pipeline.

    Attributes:
        signal: Current working signal, shape ``(C, T)``. Updated after each stage.
        original: Frozen copy of the input signal set once at pipeline entry.
        sample_rate: Sample rate in Hz.
        target: Optional label — speaker id, class index, etc.
        output: Result of the latest augmentation stage. Initialised to *signal*.
        persist: Per-stage snapshots keyed by stage name.
    """

    signal: torch.Tensor
    original: torch.Tensor
    sample_rate: int
    target: tp.Optional[tp.Union[torch.Tensor, int, str]] = None
    output: tp.Optional[torch.Tensor] = dataclasses.field(default=None)
    persist: tp.Dict[str, PersistEntry] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output is None:
            self.output = self.signal

    def to_device(self, device: tp.Union[str, torch.device]) -> "DataSample":
        """Return a new DataSample with all tensors moved to *device*."""
        return DataSample(
            signal=self.signal.to(device),
            original=self.original.to(device),
            sample_rate=self.sample_rate,
            target=(
                self.target.to(device)
                if isinstance(self.target, torch.Tensor)
                else self.target
            ),
            output=self.output.to(device) if self.output is not None else None,
            persist={
                k: PersistEntry(
                    config=v.config,
                    signal=v.signal.to(device) if v.signal is not None else None,
                )
                for k, v in self.persist.items()
            },
        )

    def to_numpy(self) -> tp.Dict[str, tp.Any]:
        """Return a plain dict with tensors converted to numpy arrays."""
        return {
            "signal": self.signal.cpu().numpy(),
            "original": self.original.cpu().numpy(),
            "sample_rate": self.sample_rate,
            "target": (
                self.target.cpu().numpy()
                if isinstance(self.target, torch.Tensor)
                else self.target
            ),
            "output": self.output.cpu().numpy() if self.output is not None else None,
        }

    def clone(self) -> "DataSample":
        """Return a deep clone: all tensors are copied, persist entries are shallow-copied."""
        return DataSample(
            signal=self.signal.clone(),
            original=self.original.clone(),
            sample_rate=self.sample_rate,
            target=(
                self.target.clone()
                if isinstance(self.target, torch.Tensor)
                else self.target
            ),
            output=self.output.clone() if self.output is not None else None,
            persist={
                k: PersistEntry(
                    config=dataclasses.replace(v.config) if v.config is not None else None,
                    signal=v.signal.clone() if v.signal is not None else None,
                )
                for k, v in self.persist.items()
            },
        )


@dataclasses.dataclass
class BatchSample(DataSample):
    """Batched DataSample with tensors of shape ``(B, C, T)``.

    ``DataSample`` is treated as ``BatchSample`` with ``B=1`` where applicable.
    """

    def __len__(self) -> int:
        """Return batch size B."""
        return self.signal.shape[0]

    def __getitem__(self, i: int) -> DataSample:
        """Return a ``DataSample`` for the i-th element of the batch."""
        if isinstance(self.target, torch.Tensor):
            item_target: tp.Optional[tp.Union[torch.Tensor, int, str]] = self.target[i]
        elif isinstance(self.target, (list, tuple)):
            item_target = self.target[i]  # type: ignore[assignment]
        else:
            item_target = self.target

        return DataSample(
            signal=self.signal[i],
            original=self.original[i],
            sample_rate=self.sample_rate,
            target=item_target,
            output=self.output[i] if self.output is not None else None,
            persist=dict(self.persist),
        )

    @classmethod
    def from_samples(cls, samples: tp.List[DataSample]) -> "BatchSample":
        """Create a ``BatchSample`` by stacking a list of ``DataSample`` instances.

        Args:
            samples: Non-empty list of ``DataSample`` objects with identical
                ``sample_rate`` and compatible tensor shapes.

        Returns:
            A ``BatchSample`` with tensors of shape ``(B, C, T)``.
        """
        if not samples:
            raise ValueError("samples list must not be empty")

        signal = torch.stack([s.signal for s in samples])
        original = torch.stack([s.original for s in samples])
        sample_rate = samples[0].sample_rate

        # Aggregate targets into a batch-level representation
        raw_targets = [s.target for s in samples]
        if all(t is None for t in raw_targets):
            batch_target: tp.Optional[tp.Any] = None
        elif all(isinstance(t, torch.Tensor) for t in raw_targets):
            batch_target = torch.stack(raw_targets)  # type: ignore[arg-type]
        elif all(isinstance(t, int) for t in raw_targets):
            batch_target = torch.tensor(raw_targets, dtype=torch.long)
        else:
            batch_target = raw_targets  # list of str or mixed types

        # Stack outputs
        raw_outputs = [s.output for s in samples]
        if all(o is not None for o in raw_outputs):
            batch_output: tp.Optional[torch.Tensor] = torch.stack(
                raw_outputs  # type: ignore[arg-type]
            )
        else:
            batch_output = signal

        # Merge persist dicts; later samples win on key collision
        merged_persist: tp.Dict[str, PersistEntry] = {}
        for s in samples:
            merged_persist.update(s.persist)

        return cls(
            signal=signal,
            original=original,
            sample_rate=sample_rate,
            target=batch_target,
            output=batch_output,
            persist=merged_persist,
        )


class AugmentationBase(abc.ABC):
    """Abstract base class for all augmentation stages.

    Attributes:
        name: Unique stage identifier. Convention: ``'{index:02d}-{class_name_lower}'``.
        p: Probability of applying this augmentation. Default ``1.0`` (always apply).
        persist: If ``True``, save both config and signal. Wins over the other flags.
        persist_config: If ``True``, save only the sampled config.
        persist_signal: If ``True``, save only the output signal snapshot.
    """

    def __init__(
        self,
        name: str,
        p: float = 1.0,
        persist: bool = False,
        persist_config: bool = False,
        persist_signal: bool = False,
    ) -> None:
        self.name = name
        self.p = p
        self.persist = persist
        self.persist_config = persist_config
        self.persist_signal = persist_signal

    @abc.abstractmethod
    def sample(self, data: DataSample) -> AugConfig:
        """Draw random augmentation parameters. Must be stateless w.r.t. *self*."""
        ...

    @abc.abstractmethod
    def apply(self, data: DataSample, config: AugConfig) -> DataSample:
        """Apply augmentation deterministically given *config*.

        Implementations must return a **new** ``DataSample`` (use
        ``dataclasses.replace``) with ``signal`` and ``output`` updated.
        The ``persist`` dict must be a shallow copy of ``data.persist``.
        """
        ...

    def _should_persist_config(self) -> bool:
        """Return True if the sampled config should be saved to ``PersistEntry``."""
        return self.persist or self.persist_config

    def _should_persist_signal(self) -> bool:
        """Return True if the output signal should be saved to ``PersistEntry``."""
        return self.persist or self.persist_signal

    def __call__(
        self,
        data: DataSample,
        config: tp.Optional[AugConfig] = None,
    ) -> DataSample:
        """Run the augmentation: probability gate → sample → apply → persist.

        Args:
            data: Input sample (single or batch).
            config: Pre-sampled config for deterministic replay. If provided,
                the probability gate is bypassed.

        Returns:
            Updated ``DataSample`` (or ``BatchSample``).
        """
        # Skip probability gate when replaying from a provided config
        if config is None:
            if torch.rand(1).item() > self.p:
                return data
            config = self.sample(data)

        result = self.apply(data, config)

        if self._should_persist_config() or self._should_persist_signal():
            entry = PersistEntry(
                config=config if self._should_persist_config() else None,
                signal=(
                    result.output.clone()
                    if self._should_persist_signal() and result.output is not None
                    else None
                ),
            )
            new_persist = dict(result.persist)
            new_persist[self.name] = entry
            result = dataclasses.replace(result, persist=new_persist)

        return result
