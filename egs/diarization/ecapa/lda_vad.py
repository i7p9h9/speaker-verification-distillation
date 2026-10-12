"""Two-class LDA on frame-level features as a VAD.

Trains a linear speech / non-speech classifier on the overlap-averaged
frame-level features ``H`` written by ``extract.py`` (``frames`` in the
``.npz``), as an alternative to the attention logits ``v_t``.

The classes are Gaussians with a shared covariance, so the score

    s_t = w^T (h_t - (mu_speech + mu_nonspeech) / 2),   w = Sigma_w^{-1} (mu_speech - mu_nonspeech)

is the speech / non-speech log-likelihood ratio: 0 is the equal-prior
decision threshold, and ``--use-prior`` adds ``log(n_speech / n_nonspeech)``
to get the posterior log-odds.  ``diarize.py --vad-lda`` uses ``s_t`` in place
of ``v_t``, so its ``--onset`` / ``--offset`` apply to it.

Training only accumulates per-class counts, sums and scatter matrices
(``C x C`` in float64), so any amount of data fits in memory.  The scatter is
computed on ``--device`` in float32 per file and summed in float64.

Labels are matched to ``<npz-dir>/**/<file_id>.npz`` by file id (the stem):

* ``--labels-rttm-dir``: RTTM turns of any speaker, rasterized on the frame grid;
* ``--labels-npy-dir``: ``<file_id>.npy`` with 0/1 (or probability, > 0.5 is
  speech) values of any length, e.g. per sample, resampled to the number of
  frames with zero order hold; a 2-D ``(speakers, length)`` array is reduced
  to activity of any speaker.

Files without labels are skipped.  If a dev set is given, the frame error at
threshold 0 and the ROC AUC of the LDA score and of the model's ``v_t`` are
printed for it, otherwise for the training set.

Writes ``--out`` (``.npz``) with

    w               (C,) float32  projection
    b               ()   float32  bias, so that s_t = frames @ w + b
    means           (2, C) float64 class means (non-speech, speech)
    counts          (2,) int64    training frames per class
    covariance      (C, C) float64 shared within-class covariance (before shrinkage)

Usage:
    python egs/diarization/ecapa/lda_vad.py --npz-dir data/npz/train \
        --labels-rttm-dir data/rttm/train --out data/lda_vad.npz \
        [--dev-npz-dir data/npz/dev --dev-labels-rttm-dir data/rttm/dev]
"""

from __future__ import annotations

import argparse
import typing as tp
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
    from rttm import read_rttm_dir, rttm_to_vad
except ImportError:
    from .rttm import read_rttm_dir, rttm_to_vad

LabelFn = tp.Callable[[str, int, float], tp.Optional[np.ndarray]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz-dir", type=Path, required=True)
    labels = p.add_mutually_exclusive_group(required=True)
    labels.add_argument("--labels-rttm-dir", type=Path, help="reference RTTMs, file id = npz stem")
    labels.add_argument("--labels-npy-dir", type=Path, help="<file_id>.npy activity arrays")
    p.add_argument("--dev-npz-dir", type=Path, default=None)
    dev_labels = p.add_mutually_exclusive_group()
    dev_labels.add_argument("--dev-labels-rttm-dir", type=Path, default=None)
    dev_labels.add_argument("--dev-labels-npy-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frame-stride", type=int, default=1, help="use every n-th frame for training")
    p.add_argument(
        "--shrinkage",
        type=float,
        default=1e-3,
        help="add shrinkage * mean eigenvalue to the diagonal of the within-class covariance",
    )
    p.add_argument("--use-prior", action="store_true", help="add the log prior odds of the training data to b")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def resample_labels(labels: np.ndarray, num_frames: int) -> np.ndarray:
    """Zero order hold resampling of the last axis to ``num_frames``."""
    index = np.round(np.linspace(0, labels.shape[-1] - 1, num_frames)).astype(int)
    return labels[..., index]


def rttm_labels(directory: Path) -> LabelFn:
    turns = read_rttm_dir(directory)

    def get(file_id: str, num_frames: int, frame_shift: float) -> np.ndarray | None:
        if file_id not in turns:
            return None
        return rttm_to_vad(turns[file_id], 1.0 / frame_shift, num_frames)

    return get


def npy_labels(directory: Path) -> LabelFn:
    paths = {path.stem: path for path in sorted(directory.rglob("*.npy"))}

    def get(file_id: str, num_frames: int, frame_shift: float) -> np.ndarray | None:
        if file_id not in paths:
            return None
        labels = np.load(paths[file_id]).astype(np.float32) > 0.5
        if labels.ndim == 2:
            labels = labels.any(axis=0)
        if labels.ndim != 1 or not labels.size:
            raise ValueError(f"{paths[file_id]}: expected a non-empty 1-D or 2-D array, got {labels.shape}")
        return resample_labels(labels, num_frames)

    return get


def iter_files(npz_dir: Path, get_labels: LabelFn, desc: str) -> tp.Iterator[tp.Tuple[np.lib.npyio.NpzFile, np.ndarray]]:
    """Yield ``(npz, labels (N,) bool)`` for every ``.npz`` that has labels."""
    files = sorted(npz_dir.rglob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files in {npz_dir}")
    skipped = []
    for path in tqdm(files, desc=desc):
        with np.load(path) as data:
            if "frames" not in data:
                raise SystemExit(f"{path}: no 'frames', re-run extract.py with --save-frames")
            labels = get_labels(path.stem, len(data["frames"]), float(data["frame_shift"]))
            if labels is None:
                skipped.append(path.stem)
                continue
            yield data, labels
    if skipped:
        print(f"{desc}: {len(skipped)} files without labels skipped: {' '.join(skipped[:10])}{' ...' * (len(skipped) > 10)}")


# ---------------------------------------------------------------------------
# LDA
# ---------------------------------------------------------------------------
class ScatterAccumulator:
    """Per-class count, sum and scatter of features, shifted by a reference point.

    The shift (the mean of the first file) only reduces cancellation in the
    float32 scatter; the statistics are corrected for it in :meth:`finalize`.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.shift: torch.Tensor | None = None
        self.count = torch.zeros(2, dtype=torch.float64, device=device)
        self.sum: torch.Tensor | None = None
        self.scatter: torch.Tensor | None = None

    def add(self, frames: np.ndarray, labels: np.ndarray) -> None:
        x = torch.from_numpy(np.asarray(frames, dtype=np.float32)).to(self.device)
        if self.shift is None:
            dim = x.shape[1]
            self.shift = x.mean(dim=0)
            self.sum = torch.zeros(2, dim, dtype=torch.float64, device=self.device)
            self.scatter = torch.zeros(2, dim, dim, dtype=torch.float64, device=self.device)
        x = x - self.shift
        y = torch.from_numpy(labels).to(self.device)
        for c in range(2):
            xc = x[y == bool(c)]
            self.count[c] += len(xc)
            self.sum[c] += xc.sum(dim=0).double()
            self.scatter[c] += (xc.T @ xc).double()

    def finalize(self) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(means (2, C), within-class covariance (C, C), counts (2,))``."""
        if self.shift is None or (self.count == 0).any():
            raise SystemExit(f"need frames of both classes, got non-speech/speech {self.count.tolist()}")
        mean = self.sum / self.count[:, None]
        within = (self.scatter - self.count[:, None, None] * mean[:, :, None] * mean[:, None, :]).sum(dim=0)
        within = within / self.count.sum()
        means = mean + self.shift.double()
        return means.cpu().numpy(), within.cpu().numpy(), self.count.cpu().numpy().astype(np.int64)


def fit_lda(
    means: np.ndarray, covariance: np.ndarray, counts: np.ndarray, shrinkage: float, use_prior: bool
) -> tp.Tuple[np.ndarray, float]:
    """Projection ``w`` and bias ``b`` of the LLR ``s = x @ w + b``."""
    dim = len(covariance)
    sigma = covariance + shrinkage * np.trace(covariance) / dim * np.eye(dim)
    w = np.linalg.solve(sigma, means[1] - means[0])
    b = -0.5 * w @ (means[0] + means[1])
    if use_prior:
        b += np.log(counts[1] / counts[0])
    return w, float(b)


def apply_lda(frames: np.ndarray, lda: tp.Mapping[str, np.ndarray]) -> np.ndarray:
    """LDA speech score ``s_t`` of ``frames (N, C)``, float32."""
    return np.asarray(frames, dtype=np.float32) @ lda["w"] + lda["b"]


def load_lda(path: Path) -> tp.Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {"w": data["w"].astype(np.float32), "b": data["b"].astype(np.float32)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability that a speech frame scores above a non-speech frame (ties count half)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    # average ranks of ties
    _, first, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    ranks[order] = np.repeat(first + 0.5 * (counts - 1), counts) + 1
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(npz_dir: Path, get_labels: LabelFn, lda: tp.Mapping[str, np.ndarray], desc: str) -> None:
    lda_scores, vad_scores, all_labels = [], [], []
    for data, labels in iter_files(npz_dir, get_labels, desc):
        lda_scores.append(apply_lda(data["frames"], lda))
        vad_scores.append(data["vad"].astype(np.float32))
        all_labels.append(labels)
    if not all_labels:
        print(f"{desc}: no labelled files")
        return
    labels = np.concatenate(all_labels)
    lda_s, vad_s = np.concatenate(lda_scores), np.concatenate(vad_scores)
    print(f"{desc}: {len(labels)} frames, speech {labels.mean():.1%}")
    print(f"  LDA score: frame error at 0 {np.mean((lda_s > 0) != labels):.2%}, AUC {roc_auc(lda_s, labels):.4f}")
    print(f"  model v_t: AUC {roc_auc(vad_s, labels):.4f}")


def make_labels(rttm_dir: Path | None, npy_dir: Path | None) -> LabelFn:
    return rttm_labels(rttm_dir) if rttm_dir is not None else npy_labels(npy_dir)


def main() -> None:
    args = parse_args()
    get_labels = make_labels(args.labels_rttm_dir, args.labels_npy_dir)

    acc = ScatterAccumulator(args.device)
    with torch.inference_mode():
        for data, labels in iter_files(args.npz_dir, get_labels, "train"):
            stride = slice(None, None, args.frame_stride)
            acc.add(data["frames"][stride], labels[stride])
    means, covariance, counts = acc.finalize()
    w, b = fit_lda(means, covariance, counts, args.shrinkage, args.use_prior)
    print(f"train: {counts[0]} non-speech / {counts[1]} speech frames, dim {len(w)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        w=w.astype(np.float32),
        b=np.float32(b),
        means=means,
        counts=counts,
        covariance=covariance,
    )
    lda = load_lda(args.out)

    if args.dev_npz_dir is not None:
        dev_rttm, dev_npy = args.dev_labels_rttm_dir, args.dev_labels_npy_dir
        if dev_rttm is None and dev_npy is None:
            raise SystemExit("--dev-npz-dir needs --dev-labels-rttm-dir or --dev-labels-npy-dir")
        evaluate(args.dev_npz_dir, make_labels(dev_rttm, dev_npy), lda, "dev")
    else:
        evaluate(args.npz_dir, get_labels, lda, "train")


if __name__ == "__main__":
    main()
