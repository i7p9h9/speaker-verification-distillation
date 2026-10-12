"""AHC + VB-HMM clustering of x-vectors with a PLDA model (VBx).

Ported from VBx (``VBx/vbhmm.py``, ``VBx/VBx.py``, ``VBx/diarization_lib.py``,
``VBx/kaldi_utils.py``; Lukas Burget, Mireia Diez, Federico Landini, Jan
Profant; Apache License 2.0), PLDA scoring only:

    F. Landini, J. Profant, M. Diez, L. Burget: Bayesian HMM clustering of
    x-vector sequences (VBx) in speaker diarization: theory, implementation
    and analysis on standard tasks.  Computer Speech & Language, 2022.

Differences from the reference, none of which changes the result beyond
rounding: the forward-backward pass uses the structure of the transition
matrix (``loopP * I + (1 - loopP) * pi + eps``) and runs in numba, the AHC
uses scipy instead of fastcluster, and the score calibration of the AHC
threshold accumulates the EM statistics in one numba pass.
"""

from __future__ import annotations

import struct
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import h5py
import numba
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import eigh
from scipy.spatial.distance import squareform
from scipy.special import logsumexp, softmax


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PLDA:
    """Within-class covariance is identity and across-class is ``diag(psi)``
    after ``(x - mean) @ transform.T``."""

    mean: np.ndarray
    transform: np.ndarray
    psi: np.ndarray


def _read_kaldi_vector(fd) -> np.ndarray:
    kind = fd.read(3)
    dtype = {b"FV ": "<f4", b"DV ": "<f8"}[kind]
    assert fd.read(1) == b"\4"
    (size,) = struct.unpack("<i", fd.read(4))
    return np.frombuffer(fd.read(size * np.dtype(dtype).itemsize), dtype=dtype).astype(np.float64)


def _read_kaldi_matrix(fd) -> np.ndarray:
    kind = fd.read(3)
    if kind not in (b"FM ", b"DM "):
        raise ValueError(f"unsupported kaldi matrix type {kind!r}")
    dtype = "<f4" if kind == b"FM " else "<f8"
    assert fd.read(1) == b"\4"
    (rows,) = struct.unpack("<i", fd.read(4))
    assert fd.read(1) == b"\4"
    (cols,) = struct.unpack("<i", fd.read(4))
    data = np.frombuffer(fd.read(rows * cols * np.dtype(dtype).itemsize), dtype=dtype)
    return data.reshape(rows, cols).astype(np.float64)


def read_kaldi_plda(path: Path) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mean, transform, psi)`` of a Kaldi ``<Plda>`` file (binary or text)."""
    with open(path, "rb") as fd:
        head = fd.read(2)
        if head == b"\0B":
            assert fd.read(7) == b"<Plda> "
            mean, transform, psi = _read_kaldi_vector(fd), _read_kaldi_matrix(fd), _read_kaldi_vector(fd)
        else:
            text = (head + fd.read()).decode()
            body = text.split("<Plda>", 1)[1].split("</Plda>", 1)[0]
            vec1, rest = body.split("]", 1)
            mat, vec2 = rest.split("]", 1)
            mean = np.array(vec1.strip(" \n[").split(), dtype=float)
            transform = np.array([r.split() for r in mat.strip(" \n[").splitlines() if r.strip()], dtype=float)
            psi = np.array(vec2.strip(" \n[]").split(), dtype=float)
    return mean, transform, psi


def load_plda(path: Path, lda_dim: int | None = None) -> PLDA:
    """A Kaldi PLDA file (as in VBx ``--plda-file``; ``lda_dim`` defaults to 128)
    or a directory with ``mean.npy``, ``transform.npy``, ``psi.npy`` (as in
    ``--plda-folder``; all dimensions by default).  ``lda_dim`` keeps the
    dimensions with the largest across-class variance."""
    path = Path(path)
    if path.is_dir():
        mean = np.load(path / "mean.npy").reshape(1, -1)
        transform = np.load(path / "transform.npy")
        psi = np.load(path / "psi.npy")
        if lda_dim is not None and lda_dim < len(psi):
            order = np.argsort(-psi, kind="stable")[:lda_dim]
            transform, psi = transform[order], psi[order]
        return PLDA(mean, transform, psi)
    mean, tr, psi = read_kaldi_plda(path)
    W = np.linalg.inv(tr.T.dot(tr))
    B = np.linalg.inv((tr.T / psi).dot(tr))
    acvar, wccn = eigh(B, W)
    lda_dim = 128 if lda_dim is None else lda_dim
    return PLDA(mean, wccn.T[::-1][:lda_dim], acvar[::-1][:lda_dim])


def l2_norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class XvecTransform:
    """``transform.h5`` of the BUT models: centering, LDA, length norms."""

    def __init__(self, path: Path) -> None:
        with h5py.File(path, "r") as f:
            self.mean1, self.mean2, self.lda = (np.array(f[k]) for k in ("mean1", "mean2", "lda"))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return l2_norm(l2_norm(x - self.mean1).dot(self.lda) - self.mean2)


# ---------------------------------------------------------------------------
# AHC
# ---------------------------------------------------------------------------
@numba.njit(cache=True, fastmath=True, error_model="numpy")
def _gmm_stats(scores, c, a):  # pragma: no cover - numba
    """Sums of g, g * x, g * x^2 over all entries of the symmetric ``scores``,
    with g = 1 / (1 + exp(c + a * x)) the posterior of the second component."""
    n = scores.shape[0]
    upper = np.zeros(3)
    diag = np.zeros(3)
    for i in range(n):
        x = scores[i, i]
        g = 1.0 / (1.0 + np.exp(c + a * x))
        diag[0] += g
        diag[1] += g * x
        diag[2] += g * x * x
        s0 = s1 = s2 = 0.0
        for j in range(i + 1, n):
            x = scores[i, j]
            g = 1.0 / (1.0 + np.exp(c + a * x))
            s0 += g
            s1 += g * x
            s2 += g * x * x
        upper[0] += s0
        upper[1] += s1
        upper[2] += s2
    return 2.0 * upper + diag


def two_gmm_calib_lin(scores: np.ndarray, niters: int = 20) -> float:
    """Threshold separating a two-Gaussian (shared variance) GMM of all entries
    of the symmetric score matrix (``twoGMMcalib_lin(scores.ravel())`` without
    the calibrated scores)."""
    scores = np.ascontiguousarray(scores, dtype=np.float64)
    total = np.array([scores.size, scores.sum(), (scores**2).sum()])
    weights = np.array([0.5, 0.5])
    means = np.mean(scores) + np.std(scores) * np.array([-1, 1])
    var = np.var(scores)
    threshold = np.inf
    for _ in range(niters):
        # log p0(x) - log p1(x) = c + a * x for a shared variance
        c = np.log(weights[0] / weights[1]) - 0.5 * (means[0] ** 2 - means[1] ** 2) / var
        a = (means[0] - means[1]) / var
        second = _gmm_stats(scores, c, a)
        cnts, m1, m2 = (np.array([t - s, s]) for t, s in zip(total, second))
        weights = cnts / cnts.sum()
        means = m1 / cnts
        var = (m2 / cnts - means**2).dot(weights)
        threshold = -0.5 * (np.log(weights**2 / var) - means**2 / var).dot([1, -1]) / (means / var).dot([1, -1])
    return float(threshold)


class AHCTree(tp.NamedTuple):
    """Average-linkage dendrogram of the cosine distances (shifted to be
    non-negative by ``adjust``) and the calibrated score threshold; both are
    independent of the AHC threshold, so the tree can be cut at many."""

    linkage: np.ndarray | None
    calibrated: float
    adjust: float

    @classmethod
    def build(cls, x: np.ndarray) -> "AHCTree":
        if len(x) < 2:
            return cls(None, 0.0, 0.0)
        x = l2_norm(x.astype(np.float64))
        scores = x @ x.T
        calibrated = two_gmm_calib_lin(scores)
        dist = squareform(-scores, checks=False)
        del scores
        lin = linkage(dist, method="average")
        adjust = abs(lin[:, 2].min())
        lin[:, 2] += adjust
        return cls(lin, calibrated, adjust)

    def labels(self, threshold: float, n: int) -> np.ndarray:
        """Zero-based labels: clusters merged while their similarity exceeds
        ``calibrated + threshold``."""
        if self.linkage is None:
            return np.zeros(n, dtype=int)
        return fcluster(self.linkage, -(self.calibrated + threshold) + self.adjust, criterion="distance") - 1


def ahc(x: np.ndarray, threshold: float) -> np.ndarray:
    """Average-linkage AHC on cosine similarities, stopped at the calibrated
    threshold + ``threshold``.  Zero-based labels."""
    return AHCTree.build(x).labels(threshold, len(x))


# ---------------------------------------------------------------------------
# VB-HMM
# ---------------------------------------------------------------------------
@numba.njit(cache=True)
def _forward_backward(lls, loop_prob, pi, eps):  # pragma: no cover - numba
    T, S = lls.shape
    jump = (1.0 - loop_prob) * pi + eps  # tr[i, j] + eps = loop_prob * (i == j) + jump[j]
    lfw = np.empty((T, S))
    lbw = np.empty((T, S))

    m = lls[0].max()
    fw = np.exp(lls[0] - m) * (pi + eps)
    total = fw.sum()
    fw /= total
    acc = m + np.log(total)
    lfw[0] = np.log(fw) + acc
    for t in range(1, T):
        m = lls[t].max()
        fw = np.exp(lls[t] - m) * (loop_prob * fw + jump * fw.sum())
        total = fw.sum()
        fw /= total
        acc += m + np.log(total)
        lfw[t] = np.log(fw) + acc

    bw = np.ones(S)
    lbw[T - 1] = 0.0
    acc = 0.0
    for t in range(T - 2, -1, -1):
        m = lls[t + 1].max()
        v = np.exp(lls[t + 1] - m) * bw
        bw = loop_prob * v + (jump * v).sum()
        total = bw.sum()
        bw /= total
        acc += m + np.log(total)
        lbw[t] = np.log(bw) + acc
    return lfw, lbw


def forward_backward(lls: np.ndarray, loop_prob: float, pi: np.ndarray, eps: float = 1e-8):
    """``(gamma, total log-likelihood, log forward, log backward)`` of an HMM
    with transitions ``loop_prob * I + (1 - loop_prob) * pi`` (+ ``eps``, as
    ``log(tr + eps)`` in the reference) and initial probabilities ``pi``."""
    lfw, lbw = _forward_backward(np.ascontiguousarray(lls, dtype=np.float64), float(loop_prob), pi.astype(np.float64), eps)
    tll = logsumexp(lfw[-1])
    return np.exp(lfw + lbw - tll), tll, lfw, lbw


def vbx_plda(
    x: np.ndarray,
    plda: PLDA,
    gamma: np.ndarray,
    loop_prob: float = 0.9,
    Fa: float = 1.0,
    Fb: float = 1.0,
    max_iters: int = 40,
    epsilon: float = 1e-6,
) -> tp.Tuple[np.ndarray, np.ndarray, tp.List[float]]:
    """VB-HMM of ``VBx_plda`` (equation numbers refer to the paper); ``gamma``
    is the initial (T, S) responsibility matrix.  Returns ``(gamma, pi, ELBOs)``."""
    X = (x - plda.mean).dot(plda.transform.T)
    Phi = plda.psi
    D = X.shape[1]
    pi = np.ones(gamma.shape[1]) / gamma.shape[1]
    G = -0.5 * (np.sum(X**2, axis=1, keepdims=True) + D * np.log(2 * np.pi))  # (23)
    rho = X * np.sqrt(Phi)  # (18)
    elbos: tp.List[float] = []
    for ii in range(max_iters):
        invL = 1.0 / (1 + Fa / Fb * gamma.sum(axis=0, keepdims=True).T * Phi)  # (17)
        alpha = Fa / Fb * invL * gamma.T.dot(rho)  # (16)
        log_p_ = Fa * (rho.dot(alpha.T) - 0.5 * (invL + alpha**2).dot(Phi) + G)  # (23)
        gamma, log_pX_, logA, logB = forward_backward(log_p_, loop_prob, pi)  # (19)-(22)
        elbo = log_pX_ + Fb * 0.5 * np.sum(np.log(invL) - invL - alpha**2 + 1)  # (25)
        pi = gamma[0] + (1 - loop_prob) * pi * np.sum(
            np.exp(logsumexp(logA[:-1], axis=1, keepdims=True) + log_p_[1:] + logB[1:] - log_pX_), axis=0
        )  # (24)
        pi = pi / pi.sum()
        elbos.append(float(elbo))
        if ii > 0 and elbo - elbos[-2] < epsilon:
            break
    return gamma, pi, elbos


# ---------------------------------------------------------------------------
# Diarization of one recording
# ---------------------------------------------------------------------------
@dataclass
class VBxParams:
    threshold: float = -0.015
    Fa: float = 0.3
    Fb: float = 17.0
    loopP: float = 0.99
    init_smoothing: float = 5.0
    max_iters: int = 40


def cluster(
    x: np.ndarray,
    plda: PLDA,
    params: VBxParams,
    xvec_transform: XvecTransform | None = None,
    tree: AHCTree | None = None,
) -> np.ndarray:
    """Zero-based speaker label per x-vector (rows of ``x``); ``tree``: the
    ``AHCTree`` of the (transformed) ``x`` if already built."""
    if len(x) == 0:
        return np.zeros(0, dtype=int)
    if xvec_transform is not None:
        x = xvec_transform(x)
    x = x.astype(np.float64)
    labels = (tree or AHCTree.build(x)).labels(params.threshold, len(x))
    qinit = np.zeros((len(labels), labels.max() + 1))
    qinit[np.arange(len(labels)), labels] = 1.0
    qinit = softmax(qinit * params.init_smoothing, axis=1)
    q, _, _ = vbx_plda(x, plda, qinit, params.loopP, params.Fa, params.Fb, params.max_iters)
    return np.argsort(-q, axis=1)[:, 0]


def merge_adjacent_labels(
    starts: np.ndarray, ends: np.ndarray, labels: np.ndarray
) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge adjacent or overlapping segments with the same label; overlapping
    segments with different labels are cut in the middle of the overlap."""
    if len(labels) == 0:
        return starts, ends, labels
    adjacent_or_overlap = np.logical_or(np.isclose(ends[:-1], starts[1:]), ends[:-1] > starts[1:])
    to_split = np.nonzero(np.logical_or(~adjacent_or_overlap, labels[1:] != labels[:-1]))[0]
    starts = starts[np.r_[0, to_split + 1]]
    ends = ends[np.r_[to_split, -1]]
    labels = labels[np.r_[0, to_split + 1]]
    overlaping = np.nonzero(starts[1:] < ends[:-1])[0]
    ends[overlaping] = starts[overlaping + 1] = (ends[overlaping] + starts[overlaping + 1]) / 2.0
    return starts, ends, labels
