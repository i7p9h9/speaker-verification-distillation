from pathlib import Path

import torch

from audimentation import (
    AddNoise,
    DataSample,
    FileListAudioProvider,
    OneOf,
    Reverb,
    SequentialCompose,
)

rir_provider_a = FileListAudioProvider(paths=[Path("data/rir/room_a.wav")])
rir_provider_b = FileListAudioProvider(paths=[Path("data/rir/room_b.wav")])
rir_provider_c = FileListAudioProvider(paths=[Path("data/rir/room_c.wav")])

noise_provider_a = FileListAudioProvider(paths=[Path("data/noise/street.wav")])
noise_provider_b = FileListAudioProvider(paths=[Path("data/noise/cafe.wav")])
noise_provider_c = FileListAudioProvider(paths=[Path("data/noise/office.wav")])

pipeline = SequentialCompose(
    stages=[
        OneOf(
            stages=[
                Reverb(rir_provider=rir_provider_a, name="reverb_a"),
                Reverb(rir_provider=rir_provider_b, name="reverb_b"),
                Reverb(rir_provider=rir_provider_c, name="reverb_c"),
            ],
            weights=[10, 5, 5],
            name="reverb",
        ),
        OneOf(
            stages=[
                AddNoise(noise_provider=noise_provider_a, name="noise_a", snr_range=(3.0, 10.0)),
                AddNoise(noise_provider=noise_provider_b, name="noise_b", snr_range=(3.0, 10.0)),
                AddNoise(noise_provider=noise_provider_c, name="noise_c", snr_range=(3.0, 10.0)),
            ],
            weights=[4, 4, 2],
            name="noise",
        ),
    ],
)

signal = torch.randn(1, 16_000)
sample = DataSample(signal=signal, original=signal.clone(), sample_rate=16_000)

result = pipeline(sample)

print(f"Input shape:  {sample.signal.shape}")
print(f"Output shape: {result.signal.shape}")
