"""Blocking queue-backed audio provider: items are consumed on read."""

import queue
import typing as tp

import torch

from audimentation.utils import crop_or_pad


class QueueAudioProvider:
    """Producer/consumer queue where each sampled item is **removed** after read.

    The producer calls ``put()`` to enqueue tensors; the consumer calls
    ``sample()``, which blocks until an item is available.  Thread-safe by
    virtue of ``queue.Queue``.

    Args:
        maxsize: Maximum queue capacity. ``0`` means unlimited.
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: queue.Queue[torch.Tensor] = queue.Queue(maxsize=maxsize)

    def put(self, tensor: torch.Tensor) -> None:
        """Enqueue *tensor* for future consumption.

        Args:
            tensor: Audio tensor of shape ``(C, T)`` at whatever sample rate
                the consumer expects.  The producer is responsible for ensuring
                correct sample rate.
        """
        self._queue.put(tensor)

    def sample(self, duration: float, sample_rate: int) -> torch.Tensor:
        """Dequeue and return the next item, blocking until one is available.

        The item is **removed** from the queue after this call.

        Args:
            duration: Desired duration in seconds (used for crop/pad only).
            sample_rate: Target sample rate (used to compute target length).

        Returns:
            Tensor of shape ``(C, T)`` with ``T = round(duration * sample_rate)``.
        """
        tensor = self._queue.get()
        target_length = round(duration * sample_rate)
        return crop_or_pad(tensor, target_length)
