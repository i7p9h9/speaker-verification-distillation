"""Single-step speaker diarization from extracted embeddings and VAD logits.

Second stage of the pipeline from Thienpondt and Demuynck (Odyssey 2024),
Sections 4-5, applied to the ``.npz`` files written by ``extract.py``:

  1. VAD: hysteresis thresholding of the overlap-averaged frame logits ``v_t``
     (start above ``--onset``, stop below ``--offset``), then segments closer
     than ``--merge-gap`` are merged and segments shorter than ``--min-dur``
     are removed.  With ``--vad-lda`` the LDA speech score of the frame
     features (``lda_vad.py``) is thresholded instead of ``v_t``.  With
     ``--vad-silero`` the Silero VAD segments (``silero_vad.py``) are the speech,
     with ``--vad-silero-prob`` its probabilities are thresholded as above.
  2. Windows without detected speech are discarded.
  3. Spectral clustering of the remaining window embeddings: cosine affinity,
     top-k pruning per row (k=10), normalized Laplacian, number of speakers
     from the eigengap, k-means on the row-normalized eigenvectors.
  4. Every detected speech frame gets the label of the nearest speech window
     (by window center); runs of equal labels become RTTM turns.

The paper tunes all thresholds on the dev partitions.  VAD logits are not
probabilities and their scale depends on the model, so tune ``--onset`` /
``--offset`` with ``score_der.py`` on a dev set.

Usage:
    python egs/diarization/ecapa/diarize.py --npz-dir data/npz --out-dir data/rttm \
        --onset 0.0 --offset -0.05
"""

from __future__ import annotations

import argparse
import typing as tp
from pathlib import Path

import numpy as np

try:
    from lda_vad import apply_lda, load_lda
    from rttm import Turn, read_rttm_dir, write_rttm
    from silero_vad import SileroCache
except ImportError:
    from .lda_vad import apply_lda, load_lda
    from .rttm import Turn, read_rttm_dir, write_rttm
    from .silero_vad import SileroCache

from sklearn.cluster import KMeans
from tqdm import tqdm

Segment = tp.Tuple[float, float]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True, help="one <rel>.rttm per input .npz")
    # VAD
    p.add_argument("--onset", type=float, default=0.0, help="hysteresis start threshold on v_t")
    p.add_argument("--offset", type=float, default=-0.05, help="hysteresis stop threshold on v_t")
    p.add_argument("--merge-gap", type=float, default=0.3, help="merge speech segments closer than this, s")
    p.add_argument("--min-dur", type=float, default=0.2, help="drop speech segments shorter than this, s")
    p.add_argument(
        "--min-window-speech",
        type=float,
        default=0.0,
        help="keep windows with more detected speech than this, s",
    )
    p.add_argument(
        "--vad-lda",
        type=Path,
        default=None,
        help="LDA from lda_vad.py; threshold its score on the frame features instead of v_t",
    )
    p.add_argument(
        "--vad-silero",
        type=Path,
        default=None,
        help="silero_vad.py output dir; use its speech segments instead of v_t",
    )
    p.add_argument(
        "--vad-silero-prob",
        action="store_true",
        help="with --vad-silero: threshold its probabilities with --onset/--offset/--merge-gap/--min-dur",
    )
    p.add_argument(
        "--oracle-vad-dir",
        type=Path,
        default=None,
        help="reference RTTM dir; use its speech regions instead of model VAD",
    )
    # clustering
    p.add_argument("--top-k", type=int, default=10, help="affinity pruning: neighbours kept per row")
    p.add_argument("--max-speakers", type=int, default=10)
    p.add_argument("--num-speakers", type=int, default=None, help="fixed number of speakers")
    p.add_argument(
        "--oracle-speakers-dir",
        type=Path,
        default=None,
        help="reference RTTM dir; take the number of speakers from it",
    )
    p.add_argument("--seed", type=int, default=0)
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------
def hysteresis(scores: np.ndarray, onset: float, offset: float) -> np.ndarray:
    """Speech starts when ``score > onset`` and lasts while ``score >= offset``."""
    if offset <= onset:
        # every frame is an "on" event, an "off" event or keeps the last event's state
        event = np.where(scores > onset, 1, np.where(scores < offset, 0, -1))
        last = np.maximum.accumulate(np.where(event >= 0, np.arange(len(scores)), -1))
        return (last >= 0) & (event[np.maximum(last, 0)] == 1)
    active = np.zeros(len(scores), dtype=bool)
    on = False
    for i, score in enumerate(scores):
        if not on and score > onset:
            on = True
        elif on and score < offset:
            on = False
        active[i] = on
    return active


def mask_to_segments(mask: np.ndarray, frame_shift: float) -> tp.List[Segment]:
    """Frame mask -> ``[(start, end)]`` with frame ``i`` covering ``[i, i + 1) * frame_shift``."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(s * frame_shift, e * frame_shift) for s, e in zip(starts, ends)]


def merge_and_filter(segments: tp.List[Segment], merge_gap: float, min_dur: float) -> tp.List[Segment]:
    merged: tp.List[tp.List[float]] = []
    for start, end in segments:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if e - s >= min_dur]


def segments_to_mask(segments: tp.Iterable[Segment], num_frames: int, frame_shift: float) -> np.ndarray:
    mask = np.zeros(num_frames, dtype=bool)
    for start, end in segments:
        mask[int(round(start / frame_shift)) : int(round(end / frame_shift))] = True
    return mask


# ---------------------------------------------------------------------------
# Spectral clustering
# ---------------------------------------------------------------------------
def pruned_affinity(emb: np.ndarray, top_k: int) -> np.ndarray:
    """Cosine affinity with only the ``top_k`` largest entries kept per row, symmetrized."""
    x = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    affinity = x @ x.T
    np.fill_diagonal(affinity, 0.0)
    k = min(top_k, len(affinity) - 1)
    drop = np.argsort(affinity, axis=1)[:, : len(affinity) - k]
    np.put_along_axis(affinity, drop, 0.0, axis=1)
    affinity = np.clip(affinity, 0.0, None)
    return 0.5 * (affinity + affinity.T)


def spectral_cluster(
    emb: np.ndarray,
    top_k: int,
    max_speakers: int,
    num_speakers: int | None,
    seed: int,
) -> np.ndarray:
    if len(emb) < 2:
        return np.zeros(len(emb), dtype=int)
    affinity = pruned_affinity(emb, top_k)
    degree = affinity.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-10))
    laplacian = np.eye(len(affinity)) - d_inv_sqrt[:, None] * affinity * d_inv_sqrt[None, :]
    eigvals, eigvecs = np.linalg.eigh(laplacian)

    if num_speakers is None:
        m = min(max_speakers, len(eigvals) - 1)
        num_speakers = int(np.argmax(np.diff(eigvals[: m + 1]))) + 1
    num_speakers = max(1, min(num_speakers, len(emb)))
    if num_speakers == 1:
        return np.zeros(len(emb), dtype=int)

    u = eigvecs[:, :num_speakers]
    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-10)
    return KMeans(n_clusters=num_speakers, n_init=10, random_state=seed).fit_predict(u)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def diarize(
    data: tp.Mapping[str, np.ndarray],
    args: argparse.Namespace,
    oracle_speech: tp.List[Segment] | None,
    num_speakers: int | None,
) -> tp.List[Turn]:
    """``data``: the npz fields, with ``vad`` being the frame scores to threshold
    (see :func:`frame_scores`); ``oracle_speech`` replaces the thresholding."""
    vad = data["vad"]
    shift = float(data["frame_shift"])
    if oracle_speech is None:
        segments = mask_to_segments(hysteresis(vad, args.onset, args.offset), shift)
        segments = merge_and_filter(segments, args.merge_gap, args.min_dur)
    else:
        segments = oracle_speech
    speech = segments_to_mask(segments, len(vad), shift)
    if not speech.any():
        return []

    # Detected speech inside every window.
    cum = np.concatenate(([0], np.cumsum(speech)))
    first = np.clip(np.round(data["window_starts"] / shift).astype(int), 0, len(vad))
    last = np.clip(np.round(data["window_ends"] / shift).astype(int), 0, len(vad))
    window_speech = (cum[last] - cum[first]) * shift
    keep = window_speech > args.min_window_speech
    if not keep.any():
        keep = window_speech > 0
    if not keep.any():
        return []

    labels = spectral_cluster(data["embeddings"][keep], args.top_k, args.max_speakers, num_speakers, args.seed)

    # Each speech frame takes the label of the nearest speech window.
    centers = 0.5 * (data["window_starts"] + data["window_ends"])[keep]
    frames = np.flatnonzero(speech)
    frame_times = (frames + 0.5) * shift
    nearest = np.abs(frame_times[:, None] - centers[None, :]).argmin(axis=1)
    frame_labels = np.full(len(vad), -1, dtype=int)
    frame_labels[frames] = labels[nearest]

    turns: tp.List[Turn] = []
    change = np.flatnonzero(np.diff(np.concatenate(([-1], frame_labels, [-1]))) != 0)
    for start, end in zip(change[:-1], change[1:]):
        label = frame_labels[start]
        if label >= 0:
            turns.append((start * shift, end * shift, f"spk{label}"))
    return turns


NPZ_KEYS = ("vad", "frame_shift", "window_starts", "window_ends", "embeddings")


def frame_scores(
    npz: tp.Mapping[str, np.ndarray],
    file_id: str,
    args: argparse.Namespace,
    lda: tp.Mapping[str, np.ndarray] | None,
    silero: SileroCache | None,
) -> np.ndarray:
    """Frame scores thresholded by the hysteresis: the model's ``v_t``, the LDA
    score of the frame features or the Silero speech probability."""
    if lda is not None:
        if "frames" not in npz:
            raise SystemExit(f"{file_id}: no frame features for --vad-lda, extract with --save-frames")
        return apply_lda(npz["frames"], lda)
    if silero is not None and args.vad_silero_prob:
        return silero.frame_prob(file_id, len(npz["vad"]), float(npz["frame_shift"]))
    return npz["vad"]


def load_vad_sources(args: argparse.Namespace) -> tp.Tuple[dict | None, SileroCache | None]:
    if args.vad_lda and args.vad_silero:
        raise SystemExit("--vad-lda and --vad-silero are exclusive")
    if args.vad_silero_prob and not args.vad_silero:
        raise SystemExit("--vad-silero-prob needs --vad-silero")
    lda = load_lda(args.vad_lda) if args.vad_lda else None
    silero = SileroCache(args.vad_silero) if args.vad_silero else None
    return lda, silero


def main() -> None:
    args = parse_args()
    files = sorted(args.npz_dir.rglob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files in {args.npz_dir}")

    oracle_vad = read_rttm_dir(args.oracle_vad_dir) if args.oracle_vad_dir else None
    oracle_spk = read_rttm_dir(args.oracle_speakers_dir) if args.oracle_speakers_dir else None
    lda, silero = load_vad_sources(args)

    for path in tqdm(files, desc="diarize"):
        file_id = path.stem
        oracle_speech = None
        if oracle_vad is not None:
            ref = oracle_vad.get(file_id, [])
            oracle_speech = merge_and_filter(sorted((s, e) for s, e, _ in ref), 0.0, 0.0)
        elif silero is not None and not args.vad_silero_prob:
            oracle_speech = silero.segments(file_id)
        num_speakers = args.num_speakers
        if oracle_spk is not None:
            num_speakers = len({spk for _, _, spk in oracle_spk.get(file_id, [])}) or None

        with np.load(path) as npz:
            data = {k: npz[k] for k in NPZ_KEYS}
            data["vad"] = frame_scores(npz, file_id, args, lda, silero)
        turns = diarize(data, args, oracle_speech, num_speakers)
        out_path = (args.out_dir / path.relative_to(args.npz_dir)).with_suffix(".rttm")
        write_rttm(out_path, file_id, turns)


if __name__ == "__main__":
    main()
