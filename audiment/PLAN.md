# Implementation Plan — audimentation

## Module List

| File | Description |
|------|-------------|
| `audimentation/utils.py` | Pure helpers: `load_audio`, `crop_or_pad`, `rms` |
| `audimentation/base.py` | `PersistEntry`, `DataSample`, `BatchSample`, `AugConfig`, `AugmentationBase` |
| `audimentation/mix_strategy.py` | `MixStrategy` protocol, `SumMix`, `MeanMix`, `MixUpMix` stub |
| `audimentation/compose.py` | `SequentialCompose`, `ParallelCompose` |
| `audimentation/samplers.py` | `BatchWiseSampler`, `SampleWiseSampler`, `IdWiseSampler` |
| `audimentation/providers/base.py` | `AudioProvider` protocol, `NoiseProvider`/`RIRProvider` aliases |
| `audimentation/providers/file_list.py` | `FileListAudioProvider` — reads from disk on each call |
| `audimentation/providers/in_memory.py` | `InMemoryAudioProvider` — loads all files at init |
| `audimentation/providers/queue.py` | `QueueAudioProvider` — blocking consumer/producer queue |
| `audimentation/providers/tagged_queue.py` | `TaggedQueueAudioProvider` — soft-eviction priority queue |
| `audimentation/augmentations/noise.py` | `AddNoiseConfig`, `AddNoise` |
| `audimentation/augmentations/reverb.py` | `ReverbConfig`, `Reverb` |
| `audimentation/__init__.py` | Public API re-exports |
| `audimentation/providers/__init__.py` | Provider re-exports |
| `audimentation/augmentations/__init__.py` | Augmentation re-exports |
| `tests/test_pipeline.py` | 11 test cases |

---

## Implementation Order

1. **`utils.py`** — no dependencies; used by providers
2. **`base.py`** — everything depends on it; implement `PersistEntry`, `AugConfig`, `DataSample`, `BatchSample`, `AugmentationBase`
3. **`mix_strategy.py`** — needed by `ParallelCompose`
4. **`compose.py`** — depends on `base`, `mix_strategy`
5. **`samplers.py`** — depends on `base`
6. **`providers/base.py`** — protocol definitions only; no deps
7. **`providers/file_list.py`** — depends on `utils`, `providers/base`
8. **`providers/in_memory.py`** — depends on `utils`, `providers/base`
9. **`providers/queue.py`** — depends on `providers/base`
10. **`providers/tagged_queue.py`** — depends on `providers/base`
11. **`augmentations/noise.py`** — depends on `base`, `providers/base`, `utils`
12. **`augmentations/reverb.py`** — depends on `base`, `providers/base`, `utils`
13. All `__init__.py` files
14. **`tests/test_pipeline.py`**

---

## Interface Decisions

### AugmentationBase is not a dataclass
`@dataclass` inheritance in Python causes field-ordering errors when subclass fields have no defaults but parent fields do. `AugmentationBase` and all concrete augmentations use regular `__init__`. Only data containers (`DataSample`, `BatchSample`, `AugConfig` subclasses, `PersistEntry`, `TaggedItem`) use `@dataclass`.

### `apply()` convention
Every `apply()` implementation must return a new `DataSample`/`BatchSample` via `dataclasses.replace(data, signal=out, output=out, persist=dict(data.persist))`. This ensures:
- `signal` is updated for the next stage in `SequentialCompose`
- `persist` is a new dict (no accidental shared mutation)

### Probability gate bypassed when config is provided
`AugmentationBase.__call__` skips the `p` check if `config is not None`. This enables deterministic replay from persisted configs.

### Noise/RIR tensor stored transiently in config
`AddNoiseConfig.noise_tensor` and `ReverbConfig.rir_tensor` hold the tensor for use in `apply()`. They are not persisted unless `store_noise_tensor=True`. `noise_id`/`rir_id` is a UUID string for replay identification.

### `BatchWiseSampler` and `SampleWiseSampler` override `__call__`
These samplers call `augmentation.sample()` and `augmentation.apply()` element-wise internally. They override `__call__` rather than `apply()` to control how many times `sample()` is called.

### `IdWiseSampler` converts tensor targets to Python scalars
To use target values as dict keys for grouping, tensor targets are converted via `.tolist()`.

### `InMemoryAudioProvider` loads at native SR
Files are stored as `(tensor, native_sr)` pairs; resampling happens at `sample()` time using the requested `sample_rate`.

### `TaggedQueueAudioProvider` sampling
Uses `softmax(-read_count * temperature)` as specified in the task (overrides the `1/(1+read_count)` formula from the architecture doc).

---

## Open Questions

| Question | Chosen default |
|----------|---------------|
| Should `SequentialCompose` update `signal` or `output` between stages? | Each `apply()` sets both `signal=out` and `output=out`; no extra step needed |
| What `noise_id` value to use for non-file providers? | `str(uuid.uuid4())` — unique per sample call |
| Should `ParallelCompose` mix `output` or `signal` from each stage? | Mix the `output` field (result of each stage) |
| Should `QueueAudioProvider.sample()` resample? | Crop/pad only; caller is responsible for correct SR |
| `DataSample.output` initial value? | Set to `signal` in `__post_init__` |
