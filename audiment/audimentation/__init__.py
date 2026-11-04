"""audimentation — PyTorch-native audio augmentation library.

Public API
----------
Core dataflow:
    DataSample, BatchSample, AugConfig, AugmentationBase, PersistEntry

Compose:
    SequentialCompose, ParallelCompose

Samplers:
    BatchWiseSampler, SampleWiseSampler, IdWiseSampler

Mix strategies:
    MixStrategy, SumMix, MeanMix, MixUpMix

Providers:
    AudioProvider, NoiseProvider, RIRProvider,
    FileListAudioProvider, InMemoryAudioProvider,
    QueueAudioProvider, TaggedQueueAudioProvider

Augmentations:
    AddNoise, AddNoiseConfig, Clamp, ClampConfig, Reverb, ReverbConfig

Utilities:
    load_audio, crop_or_pad, rms
"""

from audimentation.base import (
    AugConfig,
    AugmentationBase,
    BatchSample,
    DataSample,
    PersistEntry,
)
from audimentation.compose import OneOf, OneOfConfig, ParallelCompose, SequentialCompose
from audimentation.mix_strategy import MeanMix, MixStrategy, MixUpMix, SumMix
from audimentation.samplers import BatchWiseSampler, IdWiseSampler, SampleWiseSampler
from audimentation.providers.base import AudioProvider, NoiseProvider, RIRProvider
from audimentation.providers.file_list import FileListAudioProvider
from audimentation.providers.in_memory import InMemoryAudioProvider
from audimentation.providers.queue import QueueAudioProvider
from audimentation.providers.tagged_queue import TaggedItem, TaggedQueueAudioProvider
from audimentation.augmentations.clamp import Clamp, ClampConfig
from audimentation.augmentations.noise import AddNoise, AddNoiseConfig
from audimentation.augmentations.reverb import Reverb, ReverbConfig
from audimentation.utils import crop_or_pad, load_audio, rms

__all__ = [
    # Core
    "AugConfig",
    "AugmentationBase",
    "BatchSample",
    "DataSample",
    "PersistEntry",
    # Compose
    "OneOf",
    "OneOfConfig",
    "ParallelCompose",
    "SequentialCompose",
    # Mix
    "MeanMix",
    "MixStrategy",
    "MixUpMix",
    "SumMix",
    # Samplers
    "BatchWiseSampler",
    "IdWiseSampler",
    "SampleWiseSampler",
    # Providers
    "AudioProvider",
    "NoiseProvider",
    "RIRProvider",
    "FileListAudioProvider",
    "InMemoryAudioProvider",
    "QueueAudioProvider",
    "TaggedItem",
    "TaggedQueueAudioProvider",
    # Augmentations
    "AddNoise",
    "AddNoiseConfig",
    "Clamp",
    "ClampConfig",
    "Reverb",
    "ReverbConfig",
    # Utils
    "crop_or_pad",
    "load_audio",
    "rms",
]
