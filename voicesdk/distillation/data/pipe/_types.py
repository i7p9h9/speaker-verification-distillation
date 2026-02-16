import typing as tp
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioSegments:
    """Dataclass for storing audio segments and metadata."""

    segments: np.ndarray  # [num_segments, segment_length]
    segments_duration: tp.List[float]  # Actual durations in seconds
    segments_weights: tp.List[float]  # Segment weights
    total_duration: float  # Total duration in seconds
