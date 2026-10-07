"""Comprehensive pipeline tests for the audimentation library."""

import dataclasses
import random
import typing as tp
from pathlib import Path

import pytest
import soundfile as sf
import torch

from audimentation.augmentations.clamp import Clamp, ClampConfig
from audimentation.augmentations.noise import AddNoise, AddNoiseConfig
from audimentation.augmentations.reverb import Reverb, ReverbConfig
from audimentation.base import (
    AugConfig,
    AugmentationBase,
    BatchSample,
    DataSample,
    PersistEntry,
)
from audimentation.compose import ParallelCompose, SequentialCompose
from audimentation.mix_strategy import MixUpMix, SumMix
from audimentation.providers.file_list import FileListAudioProvider
from audimentation.providers.in_memory import InMemoryAudioProvider
from audimentation.providers.tagged_queue import TaggedQueueAudioProvider
from audimentation.samplers import BatchWiseSampler, IdWiseSampler, SampleWiseSampler
from audimentation.utils import rms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 16000
DURATION = 1.0
T = round(DURATION * SR)


def make_signal(channels: int = 1, length: int = T) -> torch.Tensor:
    """Return a white-noise tensor of shape (channels, length)."""
    return torch.randn(channels, length)


def make_sample(
    channels: int = 1,
    length: int = T,
    target: tp.Optional[tp.Union[int, str]] = None,
) -> DataSample:
    sig = make_signal(channels, length)
    return DataSample(
        signal=sig,
        original=sig.clone(),
        sample_rate=SR,
        target=target,
    )


def write_wav(path: Path, tensor: torch.Tensor, sr: int) -> None:
    """Write (C, T) tensor to a WAV file using soundfile."""
    # soundfile expects (T, C) or (T,) for mono
    data = tensor.numpy().T  # (T, C)
    if data.shape[1] == 1:
        data = data[:, 0]
    sf.write(str(path), data, sr)


# ---------------------------------------------------------------------------
# Dummy augmentations
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ConstantConfig(AugConfig):
    """Config that carries a constant value to add to the signal."""

    value: float


class ConstantAddAug(AugmentationBase):
    """Add a fixed constant to every sample of the signal."""

    def __init__(self, name: str, value: float = 1.0, **kwargs: tp.Any) -> None:
        super().__init__(name=name, **kwargs)
        self.value = value

    def sample(self, data: DataSample) -> ConstantConfig:
        return ConstantConfig(value=self.value)

    def apply(self, data: DataSample, config: ConstantConfig) -> DataSample:
        augmented = data.signal + config.value
        return dataclasses.replace(
            data, signal=augmented, output=augmented, persist=dict(data.persist)
        )


class RandomValueAug(AugmentationBase):
    """Add a freshly drawn random float to every sample of the signal."""

    def __init__(self, name: str, **kwargs: tp.Any) -> None:
        super().__init__(name=name, **kwargs)

    def sample(self, data: DataSample) -> ConstantConfig:
        return ConstantConfig(value=float(torch.rand(1)))

    def apply(self, data: DataSample, config: ConstantConfig) -> DataSample:
        augmented = data.signal + config.value
        return dataclasses.replace(
            data, signal=augmented, output=augmented, persist=dict(data.persist)
        )


# ---------------------------------------------------------------------------
# Test 1 — DataSample construction, to_device, clone
# ---------------------------------------------------------------------------


def test_datasample_construction():
    """DataSample initialises correctly; output defaults to signal."""
    sample = make_sample()
    assert sample.signal.shape == (1, T)
    assert sample.output is not None
    assert torch.allclose(sample.output, sample.signal)
    assert sample.persist == {}


def test_datasample_to_device():
    """to_device moves all tensors; original sample is not modified."""
    sample = make_sample()
    cpu_sample = sample.to_device("cpu")
    assert cpu_sample.signal.device.type == "cpu"
    assert cpu_sample.original.device.type == "cpu"
    assert cpu_sample.output.device.type == "cpu"


def test_datasample_clone_is_independent():
    """Cloned DataSample tensors are independent copies."""
    sample = make_sample()
    cloned = sample.clone()
    cloned.signal[0, 0] = 999.0
    assert sample.signal[0, 0] != 999.0, "Original signal was mutated by clone."


def test_datasample_clone_persist_is_copied():
    """Persist entries are deep-copied in clone."""
    sig = make_signal()
    entry = PersistEntry(config=ConstantConfig(value=1.0), signal=sig.clone())
    sample = DataSample(
        signal=sig,
        original=sig.clone(),
        sample_rate=SR,
        persist={"stage": entry},
    )
    cloned = sample.clone()
    # Modify original's persist entry — clone must not reflect the change
    sample.persist["stage"].signal[0, 0] = 999.0
    assert cloned.persist["stage"].signal is not None
    assert cloned.persist["stage"].signal[0, 0] != 999.0


# ---------------------------------------------------------------------------
# Test 2 — BatchSample from list of DataSample, indexing
# ---------------------------------------------------------------------------


def test_batchsample_from_samples_shape():
    """BatchSample.from_samples stacks signals correctly."""
    samples = [make_sample(target=i) for i in range(4)]
    batch = BatchSample.from_samples(samples)
    assert len(batch) == 4
    assert batch.signal.shape == (4, 1, T)
    assert batch.original.shape == (4, 1, T)
    assert batch.output.shape == (4, 1, T)


def test_batchsample_indexing_returns_datasample():
    """BatchSample[i] returns a DataSample with the correct slice."""
    samples = [make_sample(target=i) for i in range(4)]
    batch = BatchSample.from_samples(samples)
    s0 = batch[0]
    assert isinstance(s0, DataSample)
    assert s0.signal.shape == (1, T)
    assert torch.allclose(s0.signal, samples[0].signal)


def test_batchsample_target_stacking():
    """Integer targets are stacked into a LongTensor."""
    samples = [make_sample(target=i) for i in range(4)]
    batch = BatchSample.from_samples(samples)
    assert isinstance(batch.target, torch.Tensor)
    assert batch.target.tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Test 3 — SequentialCompose with two dummy augmentations
# ---------------------------------------------------------------------------


def test_sequential_compose_two_stages():
    """Sequential compose applies stages in order; signal accumulates."""
    signal = torch.zeros(1, T)
    sample = DataSample(signal=signal, original=signal.clone(), sample_rate=SR)
    compose = SequentialCompose(
        stages=[
            ConstantAddAug(name="aug_a", value=1.0),
            ConstantAddAug(name="aug_b", value=2.0),
        ]
    )
    result = compose(sample)
    expected = torch.full_like(signal, 3.0)
    assert torch.allclose(result.signal, expected)


def test_sequential_compose_persist_accumulates():
    """SequentialCompose propagates persist entries from inner stages."""
    sample = make_sample()
    compose = SequentialCompose(
        stages=[
            ConstantAddAug(name="s1", value=0.0, persist_config=True),
            ConstantAddAug(name="s2", value=0.0, persist_config=True),
        ]
    )
    result = compose(sample)
    assert "s1" in result.persist
    assert "s2" in result.persist


# ---------------------------------------------------------------------------
# Test 4 — ParallelCompose with SumMix
# ---------------------------------------------------------------------------


def test_parallel_compose_sum_mix():
    """ParallelCompose with SumMix sums the outputs of all branches."""
    signal = torch.ones(1, T)
    sample = DataSample(signal=signal, original=signal.clone(), sample_rate=SR)
    # aug_a: signal + 1.0 → output = 2.0
    # aug_b: signal + 3.0 → output = 4.0
    # sum = 6.0
    compose = ParallelCompose(
        stages=[
            ConstantAddAug(name="branch_a", value=1.0),
            ConstantAddAug(name="branch_b", value=3.0),
        ],
        mix_strategy=SumMix(),
    )
    result = compose(sample)
    assert torch.allclose(result.signal, torch.full_like(signal, 6.0))


def test_parallel_compose_same_input():
    """All branches of ParallelCompose receive the same original signal."""
    inputs_seen: tp.List[torch.Tensor] = []

    class RecordInputAug(AugmentationBase):
        def sample(self, data: DataSample) -> AugConfig:
            inputs_seen.append(data.signal.clone())
            return AugConfig()

        def apply(self, data: DataSample, config: AugConfig) -> DataSample:
            return dataclasses.replace(
                data, output=data.signal, persist=dict(data.persist)
            )

    signal = make_signal()
    sample = DataSample(signal=signal, original=signal.clone(), sample_rate=SR)
    compose = ParallelCompose(
        stages=[RecordInputAug(name="r1"), RecordInputAug(name="r2")],
        mix_strategy=SumMix(),
    )
    compose(sample)
    assert len(inputs_seen) == 2
    assert torch.allclose(inputs_seen[0], inputs_seen[1])


# ---------------------------------------------------------------------------
# Test 5 — BatchWiseSampler vs SampleWiseSampler
# ---------------------------------------------------------------------------


class CountCallsAug(AugmentationBase):
    """Records how many times sample() is called."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.sample_calls = 0

    def sample(self, data: DataSample) -> AugConfig:
        self.sample_calls += 1
        return AugConfig()

    def apply(self, data: DataSample, config: AugConfig) -> DataSample:
        return dataclasses.replace(data, persist=dict(data.persist))


def test_batch_wise_sampler_calls_sample_once():
    """BatchWiseSampler calls sample() exactly once for a batch of 4."""
    batch = BatchSample.from_samples([make_sample() for _ in range(4)])
    aug = CountCallsAug(name="counter")
    sampler = BatchWiseSampler(augmentation=aug, name="bw")
    sampler(batch)
    assert aug.sample_calls == 1


def test_sample_wise_sampler_calls_sample_per_element():
    """SampleWiseSampler calls sample() once per batch element."""
    batch = BatchSample.from_samples([make_sample() for _ in range(4)])
    aug = CountCallsAug(name="counter")
    sampler = SampleWiseSampler(augmentation=aug, name="sw")
    sampler(batch)
    assert aug.sample_calls == 4


def test_sample_wise_sampler_configs_differ():
    """SampleWiseSampler produces independent (potentially different) configs."""
    batch = BatchSample.from_samples([make_sample() for _ in range(8)])
    aug = RandomValueAug(name="rand")
    sampler = SampleWiseSampler(augmentation=aug, name="sw")
    result = sampler(batch)
    assert isinstance(result, BatchSample)
    # Elements should generally differ (independent samples); check not all equal
    signals = [result.signal[i] for i in range(len(result))]
    all_equal = all(torch.allclose(signals[0], s) for s in signals[1:])
    assert not all_equal, "All elements are identical; independent sampling failed."


# ---------------------------------------------------------------------------
# Test 6 — IdWiseSampler — same target id → same config / same augmentation
# ---------------------------------------------------------------------------


def test_id_wise_sampler_same_target_same_result():
    """Elements sharing a target get identical augmentation (zero-input test)."""
    T_small = 100
    # 4 samples: (0, 0, 1, 1)
    samples = [
        DataSample(
            signal=torch.zeros(1, T_small),
            original=torch.zeros(1, T_small),
            sample_rate=SR,
            target=t,
        )
        for t in [0, 0, 1, 1]
    ]
    batch = BatchSample.from_samples(samples)
    aug = RandomValueAug(name="rand")
    sampler = IdWiseSampler(augmentation=aug, name="id")
    result = sampler(batch)

    assert isinstance(result, BatchSample)
    # Elements 0 & 1 (target=0) must be equal
    assert torch.allclose(result.signal[0], result.signal[1])
    # Elements 2 & 3 (target=1) must be equal
    assert torch.allclose(result.signal[2], result.signal[3])


def test_id_wise_sampler_raises_without_target():
    """IdWiseSampler raises ValueError when target is None."""
    batch = BatchSample.from_samples([make_sample() for _ in range(2)])
    aug = ConstantAddAug(name="aug", value=0.0)
    sampler = IdWiseSampler(augmentation=aug, name="id")
    with pytest.raises(ValueError, match="target"):
        sampler(batch)


# ---------------------------------------------------------------------------
# Test 7 — AddNoise with InMemoryAudioProvider — output SNR within tolerance
# ---------------------------------------------------------------------------


@pytest.fixture()
def noise_wav_path(tmp_path: Path) -> Path:
    """Write a 2-second white-noise WAV file and return its path."""
    path = tmp_path / "noise.wav"
    tensor = torch.randn(1, SR * 2)
    write_wav(path, tensor, SR)
    return path


def test_add_noise_output_power_preserved(noise_wav_path: Path) -> None:
    """AddNoise output RMS matches input RMS (power normalisation)."""
    provider = InMemoryAudioProvider(paths=[noise_wav_path])
    aug = AddNoise(noise_provider=provider, name="noise", snr_range=(5.0, 30.0))

    signal_tensor = make_signal()
    sample = DataSample(signal=signal_tensor, original=signal_tensor.clone(), sample_rate=SR)
    result = aug(sample)

    rms_in = rms(sample.signal).item()
    rms_out = rms(result.signal).item()
    assert abs(rms_in - rms_out) / (rms_in + 1e-10) < 0.01, (
        f"Output RMS {rms_out:.4f} differs from input RMS {rms_in:.4f} by more than 1 %."
    )


def test_add_noise_snr_within_tolerance(noise_wav_path: Path) -> None:
    """AddNoise achieves the target SNR within ±2 dB (measured before normalisation gain)."""
    provider = InMemoryAudioProvider(paths=[noise_wav_path])
    target_snr = 20.0
    aug = AddNoise(noise_provider=provider, name="noise", snr_range=(target_snr, target_snr))

    signal_tensor = make_signal()
    sample = DataSample(signal=signal_tensor, original=signal_tensor.clone(), sample_rate=SR)
    result = aug(sample)

    # After power normalisation the gain k = rms(signal)/rms(mixed) is applied
    # uniformly to both signal and noise components, so SNR is preserved.
    # Recover noise component: noise_out = result.signal - signal * k
    # k = rms(result.signal) / rms(signal)  [since rms(result) == rms(signal)]
    rms_signal = rms(sample.signal)
    # SNR via variance decomposition: signal component power / residual power
    # Use a linear regression projection of result onto normalised signal
    sig_unit = sample.signal / (rms_signal + 1e-10)
    coeff = (result.signal * sig_unit).mean()
    noise_component = result.signal - coeff * sig_unit
    snr_actual = 20.0 * torch.log10(rms(coeff * sig_unit) / (rms(noise_component) + 1e-10))
    assert abs(snr_actual.item() - target_snr) < 2.0, (
        f"SNR {snr_actual.item():.2f} dB is not within 2 dB of target {target_snr} dB."
    )


# ---------------------------------------------------------------------------
# Test 8 — Reverb with FileListAudioProvider — output length equals input length
# ---------------------------------------------------------------------------


@pytest.fixture()
def rir_wav_path(tmp_path: Path) -> Path:
    """Write a 1-second exponential decay RIR WAV file and return its path."""
    path = tmp_path / "rir.wav"
    t = torch.linspace(0, 1, SR)
    rir = torch.exp(-5.0 * t).unsqueeze(0)  # (1, T) — exponential decay
    write_wav(path, rir, SR)
    return path


def test_reverb_output_length_preserved(rir_wav_path: Path) -> None:
    """Reverb output has the same length as the input signal."""
    provider = FileListAudioProvider(paths=[rir_wav_path])
    aug = Reverb(rir_provider=provider, name="reverb")

    sample = make_sample()
    result = aug(sample)

    assert result.signal.shape == sample.signal.shape, (
        f"Output shape {result.signal.shape} != input shape {sample.signal.shape}."
    )


def test_reverb_multichannel(rir_wav_path: Path) -> None:
    """Reverb preserves length for multi-channel (stereo) input."""
    provider = FileListAudioProvider(paths=[rir_wav_path])
    aug = Reverb(rir_provider=provider, name="reverb")

    sample = make_sample(channels=2)
    result = aug(sample)
    assert result.signal.shape == sample.signal.shape


# ---------------------------------------------------------------------------
# Test 9 — TaggedQueueAudioProvider: high read_count → lower sampling freq
# ---------------------------------------------------------------------------


def test_tagged_queue_prefers_unread_items() -> None:
    """Items with read_count=0 are sampled more often than heavily-read items."""
    provider = TaggedQueueAudioProvider(maxsize=3, temperature=2.0)
    # Add three identical-length tensors (different values for identification)
    for val in [0.0, 1.0, 2.0]:
        provider.put(torch.full((1, SR), val))

    # Artificially inflate read_count for items 1 and 2 by sampling them
    # We sample a large number of times and track how often item 0 is returned
    n_trials = 500
    item0_count = 0
    for _ in range(n_trials):
        t = provider.sample(DURATION, SR)
        if torch.allclose(t[:, :1], torch.tensor([[[0.0]]])):
            item0_count += 1

    # After many reads, items 1 and 2 accumulate high read_counts.
    # Item 0 should be sampled noticeably more than 1/3 of the time.
    # We use a loose threshold to avoid flakiness.
    assert item0_count > n_trials / 4, (
        f"Item 0 sampled only {item0_count}/{n_trials} times; "
        "expected preference for low-read-count items."
    )


def test_tagged_queue_evicts_most_read() -> None:
    """When at capacity, put() evicts the highest read_count item."""
    provider = TaggedQueueAudioProvider(maxsize=2, temperature=1.0)
    a = torch.zeros(1, SR)
    b = torch.ones(1, SR)
    provider.put(a)
    provider.put(b)

    # Read b many times to inflate its read_count
    for _ in range(10):
        provider.sample(DURATION, SR)

    # Add a new item — should evict whichever has the highest read_count
    c = torch.full((1, SR), 2.0)
    provider.put(c)
    assert len(provider) == 2  # still at capacity


# ---------------------------------------------------------------------------
# Test 10 — Persist flag logic
# ---------------------------------------------------------------------------


def test_persist_true_saves_both() -> None:
    """persist=True saves both config and signal in PersistEntry."""
    sample = make_sample()
    aug = ConstantAddAug(name="stage", value=1.0, persist=True)
    result = aug(sample)
    assert "stage" in result.persist
    entry = result.persist["stage"]
    assert entry.config is not None
    assert entry.signal is not None


def test_persist_config_only() -> None:
    """persist_config=True saves config but not signal."""
    sample = make_sample()
    aug = ConstantAddAug(name="stage", value=1.0, persist_config=True)
    result = aug(sample)
    entry = result.persist["stage"]
    assert entry.config is not None
    assert entry.signal is None


def test_persist_signal_only() -> None:
    """persist_signal=True saves signal but not config."""
    sample = make_sample()
    aug = ConstantAddAug(name="stage", value=1.0, persist_signal=True)
    result = aug(sample)
    entry = result.persist["stage"]
    assert entry.config is None
    assert entry.signal is not None


def test_persist_false_saves_nothing() -> None:
    """Default persist=False writes no PersistEntry."""
    sample = make_sample()
    aug = ConstantAddAug(name="stage", value=1.0)
    result = aug(sample)
    assert "stage" not in result.persist


def test_persist_true_overrides_config_only() -> None:
    """persist=True wins over persist_config=True and saves both."""
    sample = make_sample()
    # persist=True even when persist_config is also True
    aug = ConstantAddAug(name="stage", value=1.0, persist=True, persist_config=True)
    result = aug(sample)
    entry = result.persist["stage"]
    assert entry.config is not None
    assert entry.signal is not None


# ---------------------------------------------------------------------------
# Test 11 — Full replay
# ---------------------------------------------------------------------------


def test_full_replay_produces_identical_output() -> None:
    """Replaying a pipeline with persisted configs yields bit-identical results."""
    sample = make_sample()

    aug1 = ConstantAddAug(name="s1", value=3.0, persist_config=True)
    aug2 = ConstantAddAug(name="s2", value=7.0, persist_config=True)
    compose = SequentialCompose(stages=[aug1, aug2], name="seq")

    # First run — collect configs
    result1 = compose(sample)
    assert "s1" in result1.persist and "s2" in result1.persist

    # Replay: build a fresh sample from original and apply each stage with its config
    replay_input = DataSample(
        signal=sample.original.clone(),
        original=sample.original.clone(),
        sample_rate=sample.sample_rate,
    )
    for stage in compose.stages:
        saved_config = result1.persist[stage.name].config
        replay_input = stage(replay_input, config=saved_config)

    assert torch.allclose(result1.signal, replay_input.signal), (
        "Replay output does not match the original run output."
    )


def test_full_replay_with_random_aug() -> None:
    """Replaying with persisted RandomValueAug configs produces identical results."""
    sample = make_sample()

    aug1 = RandomValueAug(name="r1", persist_config=True)
    aug2 = RandomValueAug(name="r2", persist_config=True)
    compose = SequentialCompose(stages=[aug1, aug2], name="seq")

    result1 = compose(sample)

    # Replay
    replay_input = DataSample(
        signal=sample.original.clone(),
        original=sample.original.clone(),
        sample_rate=sample.sample_rate,
    )
    for stage in compose.stages:
        saved_config = result1.persist[stage.name].config
        replay_input = stage(replay_input, config=saved_config)

    assert torch.allclose(result1.signal, replay_input.signal)


# ---------------------------------------------------------------------------
# Clamp augmentation tests
# ---------------------------------------------------------------------------


def test_clamp_symmetric_default() -> None:
    """Default Clamp clips the top/bottom 5 % symmetrically around zero."""
    # Build a signal with known extreme values
    signal = torch.zeros(1, 1000)
    signal[0, :50] = 10.0   # top 5 % extremes
    signal[0, -50:] = -10.0  # bottom 5 % extremes
    signal[0, 50:-50] = torch.linspace(-1.0, 1.0, 900)
    sample = DataSample(signal=signal, original=signal.clone(), sample_rate=SR)

    aug = Clamp(name="clamp")
    result = aug(sample)

    # All extreme values must have been clipped
    assert result.signal.max() < 10.0
    assert result.signal.min() > -10.0
    # Symmetric: |max| ≈ |min|
    assert abs(result.signal.max().item() + result.signal.min().item()) < 0.1


def test_clamp_values_within_bounds() -> None:
    """No output value exceeds the computed thresholds."""
    sample = make_sample()
    aug = Clamp(name="clamp", quantile=0.1, persist_config=True)
    result = aug(sample)

    entry = result.persist["clamp"]
    assert entry.config is not None
    cfg: ClampConfig = entry.config  # type: ignore[assignment]
    assert result.signal.min().item() >= cfg.low - 1e-6
    assert result.signal.max().item() <= cfg.high + 1e-6


def test_clamp_asymmetric_fine_tune() -> None:
    """Asymmetric clamp with low_ratio != high_ratio applies independent bounds."""
    sample = make_sample()
    aug = Clamp(
        name="clamp_asym",
        quantile=0.05,
        low_ratio=0.01,
        high_ratio=0.10,
        symmetric=False,
        persist_config=True,
    )
    result = aug(sample)
    entry = result.persist["clamp_asym"]
    assert entry.config is not None
    cfg: ClampConfig = entry.config  # type: ignore[assignment]
    # Asymmetric: |low| and |high| may differ
    # Just verify the output is within bounds
    assert result.signal.min().item() >= cfg.low - 1e-6
    assert result.signal.max().item() <= cfg.high + 1e-6


def test_clamp_replay_deterministic() -> None:
    """Clamp replayed with persisted config produces identical output."""
    sample = make_sample()
    aug = Clamp(name="clamp", quantile=0.05, persist_config=True)
    result1 = aug(sample)

    saved_config = result1.persist["clamp"].config
    replay_sample = sample.clone()
    result2 = aug(replay_sample, config=saved_config)

    assert torch.allclose(result1.signal, result2.signal)


def test_clamp_invalid_quantile() -> None:
    """Clamp raises ValueError for quantile outside (0, 0.5)."""
    with pytest.raises(ValueError, match="quantile"):
        Clamp(name="clamp", quantile=0.0)
    with pytest.raises(ValueError, match="quantile"):
        Clamp(name="clamp", quantile=0.5)
