# Task: Implement `audimentation` — PyTorch Audio Augmentation Library

## Context

Read `audimentation.md` for the full architecture description before writing any code.  
Read `CLAUDE.md` for project-wide Python style guidelines before writing any code.

You are implementing a PyTorch-native audio augmentation library.  
Stack: **Python + PyTorch + torchaudio + soundfile**.

---

## Mandatory style rules (in addition to CLAUDE.md)

- `import typing as tp` everywhere — never `from typing import List, Optional, ...`
- All code, comments, docstrings, log messages, and exception texts in **English**
- Dataclasses for all config/data containers
- No hidden numpy conversions mid-pipeline; stay in torch tensors
- Thread-safety required for all Provider implementations

---

## Step 0 — Read before coding

1. Read `audimentation.md` (architecture overview)
2. Read `CLAUDE.md` (style guide)
3. Only then proceed to Step 1

---

## Step 1 — Write an implementation plan

Before writing any code, produce a file `PLAN.md` containing:

- List of all modules/files to create with one-line description each
- Ordered list of implementation steps (which module first, why)
- Interface decisions: any ambiguities in the architecture you resolved and how
- Open questions (if any) you would normally ask, plus your chosen default answer

Do **not** write code in Step 1. Only `PLAN.md`.

---

## Step 2 — Implement in order

Follow your plan from Step 1. Implement module by module, fully, before moving to the next.

### 2.1 — Core dataflow (`audimentation/base.py`)

Implement:

- `PersistEntry` dataclass
- `DataSample` dataclass with fields:
  - `signal: torch.Tensor` — shape `(C, T)`, current working signal
  - `original: torch.Tensor` — frozen, set once, never mutated
  - `sample_rate: int`
  - `target: tp.Optional[...]`
  - `output: torch.Tensor` — result of latest stage
  - `persist: tp.Dict[str, PersistEntry]`
  - Helper methods: `to_device(device)`, `to_numpy()`, `clone()`
- `BatchSample(DataSample)` — same fields, tensors shaped `(B, C, T)`
  - `__len__` returns `B`
  - `__getitem__` returns a `DataSample` for index `i`
  - `from_samples(cls, samples: tp.List[DataSample])` classmethod
- `AugConfig` — empty base dataclass (all concrete configs inherit from it)
- `AugmentationBase` abstract class:
  - `name: str`
  - `p: float = 1.0`
  - `persist: bool = False`
  - `persist_config: bool = False`
  - `persist_signal: bool = False`
  - Abstract `sample(data: DataSample) -> AugConfig`
  - Abstract `apply(data: DataSample, config: AugConfig) -> DataSample`
  - Concrete `__call__(data, config=None)` — handles probability, calls sample+apply, writes PersistEntry
  - Helper `_should_persist_config() -> bool` and `_should_persist_signal() -> bool` respecting flag priority

### 2.2 — Compose (`audimentation/compose.py`)

Implement:

- `SequentialCompose(AugmentationBase)` — list of stages, passes output forward
- `ParallelCompose(AugmentationBase)` — all stages share same input, outputs combined via `MixStrategy`
- Both must correctly propagate `persist` entries from inner stages up to the top-level sample

### 2.3 — Mix strategy stub (`audimentation/mix_strategy.py`)

Implement:

- `MixStrategy` Protocol with `mix(signals, targets) -> Tensor`
- `SumMix`, `MeanMix` concrete implementations
- `MixUpMix` stub — raises `NotImplementedError` with a clear message explaining it is reserved for future implementation

### 2.4 — Samplers (`audimentation/samplers.py`)

Implement:

- `BatchWiseSampler(AugmentationBase)` — one `sample()`, `apply()` on full batch tensor
- `SampleWiseSampler(AugmentationBase)` — one `sample()` per element, loop over `B`
- `IdWiseSampler(AugmentationBase)` — one `sample()` per unique `target` value; raises `ValueError` if `target` is None

### 2.5 — Provider protocol (`audimentation/providers/base.py`)

Implement:

- `AudioProvider` Protocol:
  ```python
  def sample(self, duration: float, sample_rate: int) -> torch.Tensor: ...
  ```
- `NoiseProvider = AudioProvider` (type alias)
- `RIRProvider = AudioProvider` (type alias)

### 2.6 — Provider: FileList (`audimentation/providers/file_list.py`)

Implement `FileListAudioProvider`:
- Constructor: `paths: tp.List[tp.Union[str, Path]]`, `sample_mode: str = 'random'`
- On `sample()`: pick a random file, load with `soundfile`, resample if needed with `torchaudio`, crop or pad to `duration`
- Thread-safe (reads are stateless)

### 2.7 — Provider: InMemory (`audimentation/providers/in_memory.py`)

Implement `InMemoryAudioProvider`:
- Constructor: same as FileList
- Loads all files into memory at `__init__` time as tensors
- On `sample()`: pick from in-memory pool, crop/pad to `duration`
- Thread-safe (read-only after init)

### 2.8 — Provider: Queue (`audimentation/providers/queue.py`)

Implement `QueueAudioProvider`:
- Backed by `queue.Queue`
- Consumer: `sample()` — blocks until an item is available, item is **removed** from queue after read
- Producer interface: `put(tensor: Tensor)` — external process calls this to fill the queue
- Thread-safe by design (stdlib queue)

### 2.9 — Provider: TaggedQueue (`audimentation/providers/tagged_queue.py`)

Implement `TaggedQueueAudioProvider`:

Internal item structure:
```python
@dataclass
class TaggedItem:
    tensor: torch.Tensor
    read_count: int = 0
```

Behavior:
- `sample()` — draw with probability ∝ `softmax(-read_count * temperature)` (lower read_count = higher probability); increment `read_count` of drawn item; item is **not removed**
- `put(tensor)` — add new item; if at capacity, evict the item with highest `read_count` first
- Constructor params: `maxsize: int`, `temperature: float = 1.0`
- Thread-safe via `threading.Lock`

### 2.10 — Augmentation: AddNoise (`audimentation/augmentations/noise.py`)

Implement:

```python
@dataclass
class AddNoiseConfig(AugConfig):
    snr_db: float
    noise_id: str  # identifier for replay (path or queue slot id); no raw tensor stored by default
    # noise_tensor is transient, not persisted unless explicitly requested
```

`AddNoise(AugmentationBase)`:
- Constructor: `noise_provider: NoiseProvider`, `snr_range: tp.Tuple[float, float] = (5.0, 30.0)`, `store_noise_tensor: bool = False`
- `sample()` — draw `snr_db` uniformly from `snr_range`, call `noise_provider.sample()`, return config
- `apply()` — compute RMS-based SNR scaling, add noise to `data.signal`, return updated `DataSample`
- Helper: `_compute_snr_scale(signal, noise, target_snr_db) -> float`

### 2.11 — Augmentation: Reverb (`audimentation/augmentations/reverb.py`)

Implement:

```python
@dataclass
class ReverbConfig(AugConfig):
    wet_dry: float
    rir_id: str
```

`Reverb(AugmentationBase)`:
- Constructor: `rir_provider: RIRProvider`, `wet_dry_range: tp.Tuple[float, float] = (0.3, 1.0)`
- `sample()` — draw `wet_dry`, call `rir_provider.sample()`
- `apply()` — convolve signal with RIR using `torchaudio.functional.fftconvolve`; mix wet/dry; trim output to original length
- Handle mono and multi-channel inputs

### 2.12 — Package init and utils

- `audimentation/__init__.py` — re-export main public API
- `audimentation/utils.py`:
  - `load_audio(path, target_sr) -> Tuple[Tensor, int]`
  - `crop_or_pad(tensor, target_length) -> Tensor`
  - `rms(tensor) -> Tensor`

---

## Step 3 — Tests

Write `tests/test_pipeline.py` covering:

1. `DataSample` construction, `to_device`, `clone`
2. `BatchSample` from list of `DataSample`, indexing
3. `SequentialCompose` with two dummy augmentations
4. `ParallelCompose` with `SumMix`
5. `BatchWiseSampler` vs `SampleWiseSampler` — verify configs differ per sample in SampleWise
6. `IdWiseSampler` — same target id → same config
7. `AddNoise` with `InMemoryAudioProvider` — output SNR within tolerance
8. `Reverb` with `FileListAudioProvider` — output length equals input length
9. `TaggedQueueAudioProvider` — higher `read_count` items sampled less frequently (statistical test)
10. Persist flag logic — `persist=True` saves both; `persist_config=True` saves only config
11. Full replay — run pipeline, collect configs, replay, assert outputs are identical

---

## Deliverables

```
PLAN.md
audimentation/
    __init__.py
    base.py
    compose.py
    mix_strategy.py
    samplers.py
    providers/
        __init__.py
        base.py
        file_list.py
        in_memory.py
        queue.py
        tagged_queue.py
    augmentations/
        __init__.py
        noise.py
        reverb.py
    utils.py
tests/
    test_pipeline.py
```

---

## Notes for the agent

- Do not implement MixUp logic — only the `MixUpMix` stub with `NotImplementedError`
- Do not add CLI, training loop, or dataset classes — out of scope
- If a design decision is not covered by `audimentation.md`, choose the simplest correct solution and document it in a comment
- Prefer explicit over implicit; avoid magic
- All public classes and functions must have docstrings
