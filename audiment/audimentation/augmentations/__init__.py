"""Built-in augmentation implementations."""

from audimentation.augmentations.clamp import Clamp, ClampConfig
from audimentation.augmentations.noise import AddNoise, AddNoiseConfig
from audimentation.augmentations.reverb import Reverb, ReverbConfig

__all__ = [
    "AddNoise",
    "AddNoiseConfig",
    "Clamp",
    "ClampConfig",
    "Reverb",
    "ReverbConfig",
]
