# Audimentation — Architecture Overview

## Goal

A PyTorch-native audio augmentation library with a composable, reproducible, and batch-aware pipeline.  
Stack: Python + PyTorch + torchaudio + soundfile.

---

## Design Principles

- Everything is a `torch.Tensor` operation where possible — no hidden numpy conversions mid-pipeline
- Pipeline is reproducible: if all stages persist their sampled configs, the full augmentation can be replayed deterministically
- Batch and single samples flow through the same pipeline (`BatchSample` is the general case, `DataSample` is a degenerate batch of size 1)
- Augmentation implementations are decoupled from data providers via Dependency Injection (`Provider` protocol)
- All typing uses `import typing as tp`; all code and comments in English

---

## Layer 1 — Dataflow Containers

### DataSample
```
signal       : Tensor  (C, T) — current working signal, mutated by pipeline
original     : Tensor  (C, T) — frozen copy, set once at pipeline entry, never modified
sample_rate  : int
target       : Optional[Tensor | int | str]  — speaker id, class label, etc.
output       : Tensor  — result of the latest augmentation stage
persist      : Dict[str, PersistEntry]  — keyed by stage name
```

### BatchSample(DataSample)
Same fields, but tensors have shape `(B, C, T)`.  
`DataSample` is treated as `BatchSample` with `B=1` — one flow handles both.

### PersistEntry
```
config : Optional[AugConfig]   — sampled parameters for the stage
signal : Optional[Tensor]      — snapshot of output at this stage
```

### Persist flags (per stage, evaluated left-to-right, `persist` wins)
```
persist=False            # default: save nothing
persist=True             # save config + signal
persist_config=True      # save config only
persist_signal=True      # save signal only
```

### Replay
If every stage has `persist_config=True` and configs contain no live tensors
(use ids/paths instead of raw tensors), the entire flow can be replayed:
```python
replay(sample, flow)  # re-runs each stage with its recorded config
```

---

## Layer 2 — Base Class

```
AugmentationBase
    name  : str    # e.g. 'stage_03-reverb_artificial'
    p     : float  # probability of application, default 1.0

    sample(data) -> AugConfig          # draw random parameters
    apply(data, config) -> DataSample  # deterministic application
    __call__(data, config=None)        # sample if config is None, then apply + persist
```

All classes inherit from `AugmentationBase`.

---

## Layer 3 — Compose

### SequentialCompose
Stages applied in order; output of each stage is input to the next.

### ParallelCompose
All stages receive the same input; outputs are combined via a `MixStrategy`.
```
reduction : MixStrategy  # 'sum' | 'mean' | MixUpMix (future)
```
`ParallelCompose` is the designated extension point for MixUp.

---

## Layer 4 — Samplers

Samplers wrap an augmentation and control *how* it is applied across a batch.  
All three are `AugmentationBase` subclasses.

### BatchWiseSampler
One `sample()` call → same config applied to the entire batch tensor at once.

### SampleWiseSampler
One `sample()` call per element in the batch → independent augmentation per sample.

### IdWiseSampler
One `sample()` call per unique value in `target` → all samples sharing a target id
get the same augmentation config. Requires `target` to be present.

---

## Layer 5 — Providers (Dependency Injection)

Augmentation implementations depend on `Provider` protocols, not concrete data sources.  
This decouples *what* augmentation to apply from *where* the material comes from.

### NoiseProvider (Protocol)
```
sample(duration: float, sample_rate: int) -> Tensor
```

### Implementations

| Class | Description |
|---|---|
| `FileListNoiseProvider` | List of file paths; reads from disk on every call |
| `InMemoryNoiseProvider` | Loads all files into memory at init; samples from RAM |
| `QueueNoiseProvider` | Producer/consumer queue; one process writes, another reads; sample is dropped after read |
| `TaggedQueueNoiseProvider` | Like queue, but samples are not dropped — instead a read counter is incremented. Higher counter → higher replacement priority when queue is replenished, lower sampling probability when drawing. |

`RIRProvider` (Room Impulse Response) uses the identical protocol and the same four implementations.

### TaggedQueueNoiseProvider — detailed semantics
- Internal structure: priority queue ordered by `read_count` descending
- On `sample()`: draw with probability ∝ `1 / (1 + read_count)` (softmax or similar weighting); increment `read_count` of drawn item
- On replenishment: items with highest `read_count` are replaced first
- Thread-safe; designed for streaming / online augmentation scenarios

---

## Layer 6 — Augmentation Implementations

### AddNoise
```
noise_provider : NoiseProvider
snr_range      : Tuple[float, float]  # dB
```
Config: `AddNoiseConfig(snr_db, noise_id_or_tensor)`

### Reverb
```
rir_provider : RIRProvider
wet_dry_range : Tuple[float, float]
```
Convolution via `torchaudio.functional.fftconvolve`.  
Config: `ReverbConfig(rir_id_or_tensor, wet_dry)`

---

## Layer 7 — MixUp (future, architecture stub only)

```python
class MixStrategy(Protocol):
    def mix(
        self,
        signals: tp.List[Tensor],
        targets: tp.Optional[tp.List[tp.Any]]
    ) -> Tensor: ...
```

Planned implementations:
- `MeanMix`, `SumMix` — trivial reductions
- `MixUpMix` — weighted blend with lambda; batch mode uses `torch.roll(batch, 1, 0)` for shifted targets

`ParallelCompose` accepts `MixStrategy` — no interface changes needed when MixUp is added.

---

## Stage Naming Convention

```
{sequential_index:02d}-{class_name_lower}[_{user_suffix}]
```
Examples: `'00-add_noise'`, `'01-reverb_artificial'`, `'02-add_noise_street'`

Default name is auto-generated from class name + position in Compose; user can override at construction.

---

## File / Module Layout (proposed)

```
audimentation/
    __init__.py
    base.py              # AugmentationBase, AugConfig, DataSample, BatchSample, PersistEntry
    compose.py           # SequentialCompose, ParallelCompose
    samplers.py          # BatchWiseSampler, SampleWiseSampler, IdWiseSampler
    providers/
        __init__.py
        base.py          # NoiseProvider, RIRProvider protocols
        file_list.py     # FileListNoiseProvider, FileListRIRProvider
        in_memory.py     # InMemoryNoiseProvider, InMemoryRIRProvider
        queue.py         # QueueNoiseProvider, QueueRIRProvider
        tagged_queue.py  # TaggedQueueNoiseProvider, TaggedQueueRIRProvider
    augmentations/
        __init__.py
        noise.py         # AddNoise, AddNoiseConfig
        reverb.py        # Reverb, ReverbConfig
    mix_strategy.py      # MixStrategy protocol + stub implementations
    utils.py             # audio I/O helpers, snr calculation, etc.
```
