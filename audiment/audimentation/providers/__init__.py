"""Audio provider implementations."""

from audimentation.providers.base import AudioProvider, NoiseProvider, RIRProvider
from audimentation.providers.file_list import FileListAudioProvider
from audimentation.providers.in_memory import InMemoryAudioProvider
from audimentation.providers.queue import QueueAudioProvider
from audimentation.providers.tagged_queue import TaggedQueueAudioProvider, TaggedItem

__all__ = [
    "AudioProvider",
    "NoiseProvider",
    "RIRProvider",
    "FileListAudioProvider",
    "InMemoryAudioProvider",
    "QueueAudioProvider",
    "TaggedQueueAudioProvider",
    "TaggedItem",
]
