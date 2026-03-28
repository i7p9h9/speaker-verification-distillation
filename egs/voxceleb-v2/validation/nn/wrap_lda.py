"""
AntiSpoofWrapper — PyTorch nn.Module that stacks:

    audio  →  backbone  →  LDA projection  →  (x - bias) * scale  →  sigmoid  →  score

Design constraints
------------------
* sigmoid output = 0.5  ↔  x == 0  ↔  input to sigmoid is 0
  → bias is set to the LDA score at the EER threshold, so that the
    decision boundary maps exactly to 0.5.
* The pre-sigmoid range is clipped/designed to land in [-7, +7]:
  scale is chosen so that the most extreme LDA scores (e.g. ±3 σ)
  map to ±7.  The caller may override both bias and scale explicitly.

Usage
-----
    wrapper = AntiSpoofWrapper.from_lda_eval(
        backbone   = student_model,
        lda        = fitted_sklearn_lda,
        eer_threshold = 0.312,           # cosine score at EER
        lda_score_std = 1.4,             # std of LDA scores on eval set
        logit_range   = 7.0,             # pre-sigmoid saturation point  (default)
        n_sigma       = 3.0,             # how many σ map to logit_range (default)
    )

    # or build manually:
    wrapper = AntiSpoofWrapper(backbone, lda_weight, lda_bias, bias, scale)

    audio = torch.randn(4, 1, 48000)     # (B, 1, T)
    score = wrapper(audio)               # (B,)  in (0, 1)
"""

import typing as tp
from pathlib import Path

import numpy as np
import torch
from torch import nn


class AntiSpoofWrapper(nn.Module):
    """
    Differentiable wrapper:  backbone → LDA → affine → sigmoid.

    Parameters
    ----------
    backbone : nn.Module
        Speaker embedding extractor.  Must accept (B, 1, T) and return
        either a tensor (B, D) or a tuple/list whose first element is (B, D).
    lda_weight : Tensor  shape (n_components, D)
        LDA projection matrix  (sklearn LDA's `scalings_` transposed).
    lda_bias : Tensor  shape (n_components,)
        LDA class-mean offset  (sklearn LDA's `xbar_`).
    bias : float | Tensor  scalar
        Subtracted from the LDA output *before* scaling so that the EER
        threshold maps to pre-sigmoid value 0  →  sigmoid output 0.5.
    scale : float | Tensor  scalar
        Multiplies the shifted LDA output.  Chosen so that the expected
        score range fills [-logit_range, +logit_range].
    normalize_embedding : bool
        L2-normalise the backbone embedding before LDA (default True).
    """

    def __init__(
        self,
        backbone: nn.Module,
        lda_weight: torch.Tensor,          # (n_components, D)
        lda_bias: torch.Tensor,            # (n_components,)
        bias: tp.Union[float, torch.Tensor],
        scale: tp.Union[float, torch.Tensor],
        normalize_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.normalize_embedding = normalize_embedding

        # LDA projection — stored as non-trainable buffers
        self.register_buffer("lda_weight", lda_weight.float())   # (K, D)
        self.register_buffer("lda_bias",   lda_bias.float())     # (D,)

        # Affine calibration parameters — also non-trainable by default
        bias_t  = bias  if isinstance(bias,  torch.Tensor) else torch.tensor(float(bias))
        scale_t = scale if isinstance(scale, torch.Tensor) else torch.tensor(float(scale))
        self.register_buffer("bias",  bias_t.float())
        self.register_buffer("scale", scale_t.float())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio : (B, 1, T)  — raw waveform, std-normalised

        Returns:
            score : (B,)  — antispoof probability in (0, 1)
                            > 0.5  →  real,  < 0.5  →  spoof
        """
        # 1. Backbone embedding
        emb = self.backbone(audio)
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        emb = emb.float()                              # (B, D)

        if self.normalize_embedding:
            emb = nn.functional.normalize(emb, dim=-1)

        # 2. LDA projection:  z = (emb - xbar) @ W^T
        #    sklearn convention: transform(x) = (x - xbar_) @ scalings_
        #    lda_weight shape is (K, D) so we do  emb @ W^T  where W = lda_weight
        z = (emb - self.lda_bias) @ self.lda_weight.T  # (B, K)

        # 3. Reduce to scalar: use first (or only) LD component
        z = z[:, 0].squeeze(-1)                        # (B,)

        # 4. Affine shift: centre at EER threshold, scale to [-range, +range]
        logit = (z - self.bias) * self.scale           # (B,)

        # 5. Sigmoid  →  (0, 1)
        return torch.sigmoid(logit)                    # (B,)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_lda_eval(
        cls,
        backbone: nn.Module,
        lda: tp.Any,                    # fitted sklearn LinearDiscriminantAnalysis
        eer_threshold: float,
        lda_score_std: float,
        logit_range: float = 7.0,
        n_sigma: float = 3.0,
        normalize_embedding: bool = True,
    ) -> "AntiSpoofWrapper":
        """
        Build wrapper from a fitted sklearn LDA and EER evaluation results.

        The pre-sigmoid value is designed as:

            logit = (lda_score - bias) * scale

        where
            bias  = lda_score at EER threshold
                    (so EER point → logit 0 → sigmoid 0.5)
            scale = logit_range / (n_sigma * lda_score_std)
                    (so n_sigma standard-deviations → ±logit_range)

        Args:
            backbone        : embedding backbone nn.Module
            lda             : fitted sklearn LDA (LinearDiscriminantAnalysis)
            eer_threshold   : cosine similarity score at EER operating point
            lda_score_std   : standard deviation of LD1 scores on the eval set
                              (used to calibrate scale)
            logit_range     : target magnitude of pre-sigmoid extremes (default 7)
            n_sigma         : how many σ of LD1 corresponds to logit_range  (default 3)
            normalize_embedding : L2-normalise embedding before LDA
        """
        # Extract LDA matrices from sklearn object
        # scalings_ : (D, K)  — projection directions
        # xbar_     : (D,)    — training mean subtracted before projection
        scalings = torch.from_numpy(lda.scalings_.astype(np.float32))  # (D, K)
        xbar     = torch.from_numpy(lda.xbar_.astype(np.float32))      # (D,)

        lda_weight = scalings.T   # (K, D)  — stored row-wise for matmul

        # bias: the LD1 value that corresponds to eer_threshold in cosine space.
        # Because LDA is trained on discrete labels (not cosine scores directly),
        # we calibrate by setting bias = the *mean* LD1 value of the eval set
        # projected at the decision boundary.
        # Simplest approximation: bias = mean(LD1 of real) – shift so that
        # the EER cosine boundary maps to 0.
        # Caller can also pass the bias directly via from_lda_and_bias().
        #
        # Here we use: bias = eer_threshold projected through a 1-D linear
        # approximation.  The exact value is dataset-dependent; the recommended
        # workflow is to call from_lda_and_bias() with the empirically measured
        # mean LD1 at the EER boundary (see compute_lda_bias_at_eer helper).
        bias = torch.tensor(eer_threshold, dtype=torch.float32)

        scale_val = logit_range / max(n_sigma * lda_score_std, 1e-6)
        scale = torch.tensor(scale_val, dtype=torch.float32)

        return cls(
            backbone=backbone,
            lda_weight=lda_weight,
            lda_bias=xbar,
            bias=bias,
            scale=scale,
            normalize_embedding=normalize_embedding,
        )

    @classmethod
    def from_lda_and_bias(
        cls,
        backbone: nn.Module,
        lda: tp.Any,
        bias: float,
        scale: float,
        normalize_embedding: bool = True,
    ) -> "AntiSpoofWrapper":
        """
        Build wrapper with explicit (bias, scale) — most precise option.

        Use compute_lda_bias_at_eer() to obtain the correct bias from eval data.
        """
        scalings   = torch.from_numpy(lda.scalings_.astype(np.float32))
        xbar       = torch.from_numpy(lda.xbar_.astype(np.float32))
        lda_weight = scalings.T

        return cls(
            backbone=backbone,
            lda_weight=lda_weight,
            lda_bias=xbar,
            bias=torch.tensor(bias, dtype=torch.float32),
            scale=torch.tensor(scale, dtype=torch.float32),
            normalize_embedding=normalize_embedding,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: tp.Union[str, Path]) -> None:
        """Save all buffers + backbone state_dict to a single .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "backbone_state":    self.backbone.state_dict(),
            "lda_weight":        self.lda_weight,
            "lda_bias":          self.lda_bias,
            "bias":              self.bias,
            "scale":             self.scale,
            "normalize_embedding": self.normalize_embedding,
        }, path)

    @classmethod
    def load(
        cls,
        path: tp.Union[str, Path],
        backbone: nn.Module,
        map_location: str = "cpu",
    ) -> "AntiSpoofWrapper":
        """Restore wrapper from file saved with .save()."""
        ckpt = torch.load(path, map_location=map_location)
        backbone.load_state_dict(ckpt["backbone_state"])
        return cls(
            backbone=backbone,
            lda_weight=ckpt["lda_weight"],
            lda_bias=ckpt["lda_bias"],
            bias=ckpt["bias"],
            scale=ckpt["scale"],
            normalize_embedding=ckpt.get("normalize_embedding", True),
        )


# ---------------------------------------------------------------------------
# Calibration helper — call this after evaluate_antispoof()
# ---------------------------------------------------------------------------

def compute_lda_bias_at_eer(
    lda: tp.Any,
    embeddings: np.ndarray,
    groups: tp.List[str],
    eer_threshold: float,
    normalize: bool = True,
    logit_range: float = 7.0,
    n_sigma: float = 3.0,
) -> tp.Tuple[float, float, float]:
    """
    Find the LD1 value that corresponds to the EER operating point and
    return (bias, scale) for AntiSpoofWrapper calibration.

    Strategy
    --------
    1. Project all embeddings through LDA → LD1 scores.
    2. Build all (real_i, spoof_j) cosine-similarity scores together with
       their paired LD1 average  ``(ld1[i] + ld1[j]) / 2``.
    3. Fit a linear regression  ``cos_score ~ ld1_avg``  to get the
       monotone mapping from cosine space to LD1 space.
    4. Invert the regression at ``eer_threshold`` →  ``bias``  (the LD1
       value where sigmoid output will be exactly 0.5).
    5. ``scale = logit_range / (n_sigma * std(LD1))``  so that ±n_sigma
       standard deviations from the mean map to ±logit_range before sigmoid.

    Returns
    -------
    bias      : LD1 value at the EER decision boundary
    scale     : affine scale factor
    ld1_std   : standard deviation of all LD1 scores (for reference)
    """
    from sklearn.preprocessing import normalize as sk_normalize

    emb = embeddings.astype(np.float32)
    if normalize:
        emb = sk_normalize(emb, axis=1)

    # 1. Project embeddings → LD1
    proj = (emb - lda.xbar_) @ lda.scalings_   # (N, K)
    ld1  = proj[:, 0]                           # (N,)

    real_idx  = [i for i, g in enumerate(groups) if g == "real"]
    spoof_idx = [i for i, g in enumerate(groups) if g == "spoof"]

    # 2. Build (cos_score, ld1_avg) pairs for all (real_i, spoof_j) pairs
    #    Cap at 20 000 pairs to keep runtime reasonable.
    rng = np.random.default_rng(seed=0)
    max_pairs = 20_000

    ri = np.array(real_idx)
    si = np.array(spoof_idx)

    if len(ri) * len(si) > max_pairs:
        n_each = int(max_pairs ** 0.5) + 1
        ri = rng.choice(ri, size=min(n_each, len(ri)), replace=False)
        si = rng.choice(si, size=min(n_each, len(si)), replace=False)

    # Cosine similarities in embedding space
    emb_r = emb[ri]                                       # (R, D)
    emb_s = emb[si]                                       # (S, D)
    # norms already 1.0 after sk_normalize, but be safe
    norms_r = np.linalg.norm(emb_r, axis=1, keepdims=True).clip(1e-9)
    norms_s = np.linalg.norm(emb_s, axis=1, keepdims=True).clip(1e-9)
    emb_r_n = emb_r / norms_r                             # (R, D)
    emb_s_n = emb_s / norms_s                             # (S, D)
    cos_mat  = emb_r_n @ emb_s_n.T                        # (R, S)

    # LD1 averages for each pair
    ld1_r   = ld1[ri]                                     # (R,)
    ld1_s   = ld1[si]                                     # (S,)
    ld1_avg = (ld1_r[:, None] + ld1_s[None, :]) / 2.0    # (R, S)

    cos_flat = cos_mat.ravel()    # (R*S,)
    ld1_flat = ld1_avg.ravel()    # (R*S,)

    # 3. Linear regression: cos ≈ a * ld1_avg + b  →  ld1_avg = (cos - b) / a
    #    Using np.polyfit (degree 1) for simplicity
    coeffs = np.polyfit(ld1_flat, cos_flat, deg=1)   # [a, b]: cos = a*ld1 + b
    a, b = float(coeffs[0]), float(coeffs[1])

    # 4. Invert at eer_threshold: ld1_at_eer = (eer_threshold - b) / a
    if abs(a) < 1e-9:
        # Degenerate case: LDA gives no cosine discrimination → fallback to midpoint
        bias = float((ld1[real_idx].mean() + ld1[spoof_idx].mean()) / 2.0)
    else:
        bias = (eer_threshold - b) / a

    # 5. Scale
    ld1_std = float(ld1.std())
    scale   = logit_range / max(n_sigma * ld1_std, 1e-6)

    return float(bias), float(scale), ld1_std


# ---------------------------------------------------------------------------
# Quick smoke-test (no real data needed)
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    import torch
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    D = 256   # embedding dim

    # Fake backbone: just a linear layer
    backbone = nn.Sequential(nn.Linear(16_000, D), nn.Tanh())

    # Fake fitted LDA
    n_train = 200
    X_fake  = np.random.randn(n_train, D).astype(np.float32)
    y_fake  = np.array([0] * 100 + [1] * 100)
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_fake, y_fake)

    # Calibrate bias/scale from fake eval embeddings
    emb_eval   = np.random.randn(60, D).astype(np.float32)
    groups_eval = ["real"] * 30 + ["spoof"] * 30
    bias, scale, std = compute_lda_bias_at_eer(lda, emb_eval, groups_eval, eer_threshold=0.0)
    print(f"  bias={bias:.4f}  scale={scale:.4f}  ld1_std={std:.4f}")

    wrapper = AntiSpoofWrapper.from_lda_and_bias(backbone, lda, bias=bias, scale=scale)
    wrapper.eval()

    audio  = torch.randn(4, 1, 16_000)
    with torch.no_grad():
        scores = wrapper(audio)
    assert scores.shape == (4,), f"unexpected shape {scores.shape}"
    assert scores.min() > 0 and scores.max() < 1, "scores not in (0,1)"
    print(f"  scores: {scores.tolist()}")
    print("  smoke-test passed ✓")


if __name__ == "__main__":
    _smoke_test()