"""Tagged queue provider: items persist and are sampled by inverse read frequency."""

import dataclasses
import threading
import typing as tp

import torch

from audimentation.utils import crop_or_pad


@dataclasses.dataclass
class TaggedItem:
    """An item in the tagged queue with a read counter.

    Attributes:
        tensor: Stored audio tensor of shape ``(C, T)``.
        read_count: Number of times this item has been drawn via ``sample()``.
    """

    tensor: torch.Tensor
    read_count: int = 0


class TaggedQueueAudioProvider:
    """Soft-eviction priority queue where items are **not** removed after read.

    Sampling probability is proportional to ``softmax(-read_count * temperature)``:
    items with fewer reads are drawn more often.  When the pool is at capacity
    and a new item is added, the item with the highest ``read_count`` is evicted.

    Thread-safe via ``threading.Lock``.

    Args:
        maxsize: Maximum number of items to keep in the pool.
        temperature: Controls how strongly read counts influence sampling.
            Higher temperature → more aggressive preference for unread items.
    """

    def __init__(self, maxsize: int, temperature: float = 1.0) -> None:
        if maxsize <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")
        self.maxsize = maxsize
        self.temperature = temperature
        self._items: tp.List[TaggedItem] = []
        self._lock = threading.Lock()

    def put(self, tensor: torch.Tensor) -> None:
        """Add *tensor* to the pool.

        If the pool is at capacity, the item with the highest ``read_count``
        is evicted first (i.e. the most-consumed item is replaced).

        Args:
            tensor: Audio tensor of shape ``(C, T)``.
        """
        with self._lock:
            if len(self._items) < self.maxsize:
                self._items.append(TaggedItem(tensor=tensor))
            else:
                # Evict most-read item
                evict_idx = max(
                    range(len(self._items)),
                    key=lambda i: self._items[i].read_count,
                )
                self._items[evict_idx] = TaggedItem(tensor=tensor)

    def sample(self, duration: float, sample_rate: int) -> torch.Tensor:
        """Draw an item with probability proportional to softmax(-read_count * T).

        The drawn item's ``read_count`` is incremented; it remains in the pool.

        Args:
            duration: Desired duration in seconds (used for crop/pad).
            sample_rate: Target sample rate (used to compute target length).

        Returns:
            Tensor of shape ``(C, T)`` with ``T = round(duration * sample_rate)``.

        Raises:
            RuntimeError: If the pool is empty.
        """
        with self._lock:
            if not self._items:
                raise RuntimeError(
                    "TaggedQueueAudioProvider is empty. "
                    "Call put() before sample()."
                )
            counts = torch.tensor(
                [item.read_count for item in self._items],
                dtype=torch.float32,
            )
            weights = torch.softmax(-counts * self.temperature, dim=0)
            idx = int(torch.multinomial(weights, num_samples=1).item())
            self._items[idx].read_count += 1
            tensor = self._items[idx].tensor

        target_length = round(duration * sample_rate)
        return crop_or_pad(tensor, target_length)

    def __len__(self) -> int:
        """Return the current number of items in the pool."""
        with self._lock:
            return len(self._items)
