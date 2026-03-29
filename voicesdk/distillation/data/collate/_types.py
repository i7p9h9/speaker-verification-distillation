import typing as tp
from dataclasses import dataclass

import torch


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


@dataclass
class BatchLabeledSegments:
    """
    Collated batch of LabeledSample(sample=AudioSegments, ...) items.

    segments          : (N, segment_length) — all segments concatenated across all samples
    segments_weights  : (N,) — weight per segment
    segment_to_sample : (N,) — maps each segment back to its sample index in the batch
    batch_size        : number of original samples in the batch
    labels            : (B,) — integer class label per sample
    dataset_names     : (B,) — dataset path per sample
    target            : (B,) — optional extra target per sample
    """

    segments: torch.Tensor            # (N, segment_length)
    segments_weights: torch.Tensor    # (N,)
    segment_to_sample: torch.Tensor   # (N,) int64
    batch_size: int
    labels: torch.Tensor              # (B,) int64
    dataset_names: tp.List[str]       # (B,)
    target: tp.Optional[torch.Tensor] = None  # (B,)
