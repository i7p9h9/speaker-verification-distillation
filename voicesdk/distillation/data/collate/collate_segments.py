import typing as tp

import numpy as np
import torch

from voicesdk.dataset import LabeledSample

from ..pipe import AudioSegments
from ._types import BatchLabeledSegments, BatchSegments


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
    all_dataset_names: tp.List[str] = []

    for sample_idx, item in enumerate(batch):
        if has_target:
            audio_seg, target = item
            all_targets.append(target)
        else:
            audio_seg: AudioSegments = item.sample

        all_segments.append(audio_seg.segments)
        all_weights.extend(audio_seg.segments_weights)
        all_segment_to_sample.extend([sample_idx] * audio_seg.segments.shape[0])
        all_dataset_names.append(item.dataset_name)

    return BatchSegments(
        segments=torch.from_numpy(np.concatenate(all_segments, axis=0)),
        segments_weights=torch.tensor(all_weights, dtype=torch.float32),
        segment_to_sample=torch.tensor(all_segment_to_sample, dtype=torch.int64),
        batch_size=len(batch),
        target=torch.tensor(all_targets) if has_target else None,
    )


def collate_batch_labeled_segments_fn(
    batch: tp.List[LabeledSample],
) -> BatchLabeledSegments:
    """
    Collate a list of LabeledSample(sample=AudioSegments, label=int, dataset_name=str)
    into BatchLabeledSegments.
    """
    all_segments: tp.List[np.ndarray] = []
    all_weights: tp.List[float] = []
    all_segment_to_sample: tp.List[int] = []
    all_labels: tp.List[int] = []
    all_dataset_names: tp.List[str] = []
    all_targets: tp.List[tp.Any] = []
    has_target = False

    for sample_idx, item in enumerate(batch):
        audio_seg: AudioSegments = item.sample
        all_labels.append(item.label)
        all_dataset_names.append(item.dataset_name)

        all_segments.append(audio_seg.segments)
        all_weights.extend(audio_seg.segments_weights)
        all_segment_to_sample.extend([sample_idx] * audio_seg.segments.shape[0])

    return BatchLabeledSegments(
        segments=torch.from_numpy(np.concatenate(all_segments, axis=0)),
        segments_weights=torch.tensor(all_weights, dtype=torch.float32),
        segment_to_sample=torch.tensor(all_segment_to_sample, dtype=torch.int64),
        batch_size=len(batch),
        labels=torch.tensor(all_labels, dtype=torch.int64),
        dataset_names=all_dataset_names,
        target=torch.tensor(all_targets) if has_target else None,
    )
