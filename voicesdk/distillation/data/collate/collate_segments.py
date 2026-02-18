import typing as tp
from dataclasses import dataclass

import numpy as np
import torch

from ..pipe import AudioSegments


@dataclass
class BatchSegments:
    """
    Collated batch of AudioSegments.

    segments          : (N, segment_length) — all segments concatenated
    segments_weights  : (N,) — weight per segment
    segment_to_sample : (N,) — maps each segment back to its sample index in the batch
    batch_size        : number of original samples in the batch
    target            : (B,) optional labels, one per sample
    """

    segments: torch.Tensor           # (N, segment_length)
    segments_weights: torch.Tensor   # (N,)
    segment_to_sample: torch.Tensor  # (N,) int64
    batch_size: int
    target: tp.Optional[torch.Tensor] = None  # (B,)


def collate_batch_segments_fn(
    batch: tp.List[tp.Union[AudioSegments, tp.Tuple[AudioSegments, tp.Any]]],
) -> BatchSegments:
    """
    Collate a list of AudioSegments (or (AudioSegments, target) tuples) into BatchSegments.

    Supports two dataset output formats:
        - AudioSegments                  (no target)
        - (AudioSegments, target)        (with target)
    """
    has_target = isinstance(batch[0], tuple)

    all_segments: tp.List[np.ndarray] = []
    all_weights: tp.List[float] = []
    all_segment_to_sample: tp.List[int] = []
    all_targets: tp.List[tp.Any] = []

    for sample_idx, item in enumerate(batch):
        if has_target:
            audio_seg, target = item
            all_targets.append(target)
        else:
            audio_seg = item

        all_segments.append(audio_seg.segments)
        all_weights.extend(audio_seg.segments_weights)
        all_segment_to_sample.extend([sample_idx] * audio_seg.segments.shape[0])

    return BatchSegments(
        segments=torch.from_numpy(np.concatenate(all_segments, axis=0)),
        segments_weights=torch.tensor(all_weights, dtype=torch.float32),
        segment_to_sample=torch.tensor(all_segment_to_sample, dtype=torch.int64),
        batch_size=len(batch),
        target=torch.tensor(all_targets) if has_target else None,
    )
