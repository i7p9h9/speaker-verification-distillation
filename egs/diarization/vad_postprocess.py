"""Silero VAD probabilities -> speech segments: hysteresis thresholding + merging.

The default parameters are the benchmark's ``average`` VAD (the mean of the
per-dataset tuned Silero thresholds, results/vad/average.json); the same code as
``ecapa/diarize.py`` (hysteresis, mask_to_segments, merge_and_filter), numpy only.

    python egs/diarization/vad_postprocess.py
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

import numpy as np

Segment = tp.Tuple[float, float]  # (start, end), s


@dataclass(frozen=True)
class HysteresisParams:
    onset: float = 0.2088  # speech starts when prob > onset
    offset: float = 0.1428  # ... and lasts while prob >= offset (offset <= onset)


@dataclass(frozen=True)
class MergeParams:
    merge_gap: float = 0.721  # merge segments closer than this, s
    min_dur: float = 0.3839  # then drop segments shorter than this, s


@dataclass(frozen=True)
class VadParams:
    hysteresis: HysteresisParams = HysteresisParams()
    merge: MergeParams = MergeParams()


class Hysteresis:
    """Frame probabilities -> frame mask -> ``[(start, end)]``."""

    def __init__(self, params: HysteresisParams = HysteresisParams()) -> None:
        self.params = params

    def mask(self, prob: np.ndarray) -> np.ndarray:
        onset, offset = self.params.onset, self.params.offset
        if offset <= onset:
            # every frame is an "on" event, an "off" event or keeps the last event's state
            event = np.where(prob > onset, 1, np.where(prob < offset, 0, -1))
            last = np.maximum.accumulate(np.where(event >= 0, np.arange(len(prob)), -1))
            return (last >= 0) & (event[np.maximum(last, 0)] == 1)
        active = np.zeros(len(prob), dtype=bool)
        on = False
        for i, p in enumerate(prob):
            if not on and p > onset:
                on = True
            elif on and p < offset:
                on = False
            active[i] = on
        return active

    def __call__(self, prob: np.ndarray, frame_shift: float) -> tp.List[Segment]:
        """Frame ``i`` covers ``[i, i + 1) * frame_shift`` seconds."""
        mask = self.mask(prob)
        edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        return [(float(s * frame_shift), float(e * frame_shift)) for s, e in zip(starts, ends)]


class SegmentMerger:
    """Merge segments with a gap <= ``merge_gap``, then drop those shorter than ``min_dur``."""

    def __init__(self, params: MergeParams = MergeParams()) -> None:
        self.params = params

    def __call__(self, segments: tp.Iterable[Segment]) -> tp.List[Segment]:
        merged: tp.List[tp.List[float]] = []
        for start, end in sorted(segments):
            if merged and start - merged[-1][1] <= self.params.merge_gap:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(s, e) for s, e in merged if e - s >= self.params.min_dur]


class Vad:
    """Hysteresis followed by merging."""

    def __init__(self, params: VadParams = VadParams()) -> None:
        self.hysteresis = Hysteresis(params.hysteresis)
        self.merger = SegmentMerger(params.merge)

    def __call__(self, prob: np.ndarray, frame_shift: float) -> tp.List[Segment]:
        return self.merger(self.hysteresis(prob, frame_shift))


if __name__ == "__main__":
    shift = 0.032  # Silero frame shift, s (512 samples at 16 kHz)
    rng = np.random.default_rng(0)
    # 20 s of synthetic probabilities: speech 1-6 s with a 0.5 s dip, a 0.2 s blip, speech 12-15 s
    prob = rng.uniform(0.0, 0.1, int(20 / shift))
    prob[int(1 / shift) : int(6 / shift)] = rng.uniform(0.6, 0.95, int(5 / shift))
    prob[int(3 / shift) : int(3.5 / shift)] = 0.05
    prob[int(8 / shift) : int(8.2 / shift)] = 0.9
    prob[int(12 / shift) : int(15 / shift)] = rng.uniform(0.15, 0.4, int(3 / shift))  # between offset and onset

    vad = Vad()  # the average parameters
    print("params   :", vad.hysteresis.params, vad.merger.params)
    raw = vad.hysteresis(prob, shift)
    print("raw      :", [(round(s, 2), round(e, 2)) for s, e in raw])
    print("merged   :", [(round(s, 2), round(e, 2)) for s, e in vad.merger(raw)])
    print("vad      :", [(round(s, 2), round(e, 2)) for s, e in vad(prob, shift)])

    strict = Vad(VadParams(HysteresisParams(0.5, 0.35), MergeParams(0.1, 0.25)))
    print("strict   :", [(round(s, 2), round(e, 2)) for s, e in strict(prob, shift)])
