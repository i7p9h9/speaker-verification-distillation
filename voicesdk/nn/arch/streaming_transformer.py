"""Streaming (causal) transformer speaker encoders: RoPE and TSCO.

Two encoders over acoustic frames, both causal so they can run chunk by chunk
(``offset`` is the absolute index of the first frame, for RoPE):

* :class:`RoPEEncoder` -- a plain pre-norm transformer with rotary positions;
* :class:`TSCOEncoder` -- a TSCO-inspired block (causal attention + multi-scale
  dilated causal convolutions, correlation and observer gates).  Not a
  validated reproduction of the original.

:class:`StreamingSpeakerEncoder` puts a frontend (as in :class:`ECAPA_TDNN_VAD`)
and a causal convolutional stem in front and pools the encoder frames into one
utterance embedding (mean and std, BatchNorm, linear: the ECAPA head), for
pretraining on verification.  The speaker-slot memory
for diarization is meant to be added on top of a pretrained model, so it is not
here; ``return_frames=True`` gives the frame outputs it needs.

Sizes ``s0``..``s4`` (:data:`PRESETS`) span ~10M..400M parameters on a log
scale (x2.5 per step) with the classic transformer widths, head dim 64:

=====  =======  ======  =========  ==========
size   d_model  heads   RoPE: L    TSCO: L
=====  =======  ======  =========  ==========
s0     256      4       12         7
s1     384      6       14         8
s2     640      10      12         7
s3     1024     16      12         7
s4     1280     20      20         11
=====  =======  ======  =========  ==========

TSCO has fewer layers since its block is ~1.8x larger.  Build with
``RoPESpeaker(size="s2", feat_type="tf", embed_dim=256)`` or
``TSCOSpeaker(...)``; explicit ``d_model``/``n_heads``/``n_layers`` override
the preset.
"""

from __future__ import annotations

import typing as tp

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from voicesdk.nn.feat import features, features_tf

PRESETS: dict[str, dict[str, dict[str, int]]] = {
    "rope": {
        "s0": dict(d_model=256, n_heads=4, n_layers=12),
        "s1": dict(d_model=384, n_heads=6, n_layers=14),
        "s2": dict(d_model=640, n_heads=10, n_layers=12),
        "s3": dict(d_model=1024, n_heads=16, n_layers=12),
        "s4": dict(d_model=1280, n_heads=20, n_layers=20),
    },
    "tsco": {
        "s0": dict(d_model=256, n_heads=4, n_layers=7),
        "s1": dict(d_model=384, n_heads=6, n_layers=8),
        "s2": dict(d_model=640, n_heads=10, n_layers=7),
        "s3": dict(d_model=1024, n_heads=16, n_layers=7),
        "s4": dict(d_model=1280, n_heads=20, n_layers=11),
    },
}


class CausalConvStem(nn.Module):
    """Two causal k=3 convolutions (GELU) from the features to ``d_model``; the
    first one subsamples the frames by ``stride``.  Gives the transformer local
    spectro-temporal patterns instead of single 10 ms frames."""

    def __init__(self, in_dim: int, d_model: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.conv1 = nn.Conv1d(in_dim, d_model, 3, stride=stride)
        self.conv2 = nn.Conv1d(d_model, d_model, 3)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B,T,F] -> [B,T/stride,D]
        y = F.gelu(self.conv1(F.pad(x.transpose(1, 2), (2, 0))))
        y = F.gelu(self.conv2(F.pad(y, (2, 0))))
        return self.norm(y.transpose(1, 2))


class StreamingSpeakerOutput(tp.NamedTuple):
    """Outputs of :class:`StreamingSpeakerEncoder` with ``return_frames=True``.

    Attributes:
        embedding: Utterance embedding ``(B, embed_dim)``.
        frames: Encoder outputs ``(B, T, d_model)`` (after the final norm).
        aux: Auxiliary loss (TSCO consistency), a scalar; zero otherwise.
    """

    embedding: Tensor
    frames: Tensor
    aux: Tensor


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: Tensor, k: Tensor, offset: int) -> tuple[Tensor, Tensor]:
        # q, k: [B,H,T,head_dim]
        positions = torch.arange(offset, offset + q.shape[-2], device=q.device)
        angles = torch.outer(positions.float(), self.inv_freq.float())
        cos = angles.cos().to(q.dtype)[None, None, :, :]
        sin = angles.sin().to(q.dtype)[None, None, :, :]

        def rotate(x: Tensor) -> Tensor:
            even, odd = x[..., ::2], x[..., 1::2]
            return torch.stack((even * cos - odd * sin,
                                even * sin + odd * cos), dim=-1).flatten(-2)
        return rotate(q), rotate(k)


class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = dropout

    def forward(self, x: Tensor, offset: int) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = (y.transpose(1, 2) for y in (q, k, v))
        q, k = self.rope(q, k, offset)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


class RoPEBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(),
                                 nn.Linear(d_model * 4, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, offset: int) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), offset))
        return x + self.dropout(self.mlp(self.norm2(x)))


class RoPEEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [RoPEBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, offset: int) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            x = block(x, offset)
        return self.norm(x), x.new_zeros(())


class CausalWave(nn.Module):
    def __init__(self, d_model: int, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(d_model, d_model, 3, dilation=dilation,
                              groups=d_model)
        self.project = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        y = F.pad(x.transpose(1, 2), (2 * self.dilation, 0))
        return self.project(self.conv(y)).transpose(1, 2)


class TSCOBlock(nn.Module):
    """Cleaned TSCO-inspired block; not a validated reproduction of the original."""
    def __init__(self, d_model: int, n_heads: int, n_scales: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.temporal = RoPEAttention(d_model, n_heads, dropout)
        self.waves = nn.ModuleList(CausalWave(d_model, 2 ** i)
                                   for i in range(n_scales))
        self.fusion = nn.Linear(3 * d_model, d_model)
        self.correlation = nn.Linear(d_model, d_model)
        self.observer = nn.Linear(d_model, d_model)
        self.observer_gate = nn.Linear(2 * d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))
        self.dropout = nn.Dropout(dropout)
        # Bounded learnable scalars (avoid epoch-driven, non-differentiable schedules).
        self.alpha_temporal = nn.Parameter(torch.tensor(0.0))
        self.alpha_wave = nn.Parameter(torch.tensor(-1.0))
        self.alpha_corr = nn.Parameter(torch.tensor(-2.0))
        self.alpha_observer = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: Tensor, offset: int) -> Tensor:
        h = self.norm1(x)
        temporal = self.temporal(h, offset)
        wave = torch.stack([conv(h) for conv in self.waves]).mean(0)
        fused = self.fusion(torch.cat([h, torch.sigmoid(self.alpha_temporal) * temporal,
                                      torch.sigmoid(self.alpha_wave) * wave], dim=-1))
        x = x + self.dropout(fused)
        corr = torch.tanh(x * torch.sigmoid(self.correlation(x)))
        x = x + torch.sigmoid(self.alpha_corr) * corr
        observed = self.observer(x)
        gate = torch.sigmoid(self.observer_gate(torch.cat([x, observed], dim=-1)))
        x = x + torch.sigmoid(self.alpha_observer) * gate * (observed - x)
        return x + self.dropout(self.ffn(self.norm2(x)))


class TSCOEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 n_scales: int, dropout: float, consistency_weight: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([TSCOBlock(d_model, n_heads, n_scales, dropout)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.predictor = nn.Linear(d_model, d_model)
        self.consistency_weight = consistency_weight

    def forward(self, x: Tensor, offset: int) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            x = block(x, offset)
        h = self.norm(x)
        aux = h.new_zeros(())
        if self.training and self.consistency_weight:
            # Experimental auxiliary objective; target is detached.
            aux = self.consistency_weight * F.mse_loss(self.predictor(h), h.detach())
        return h, aux


class StreamingSpeakerEncoder(nn.Module):
    """Frontend -> stem -> causal encoder -> pooling -> BatchNorm -> embedding.

    The frontend follows :class:`ECAPA_TDNN_VAD`: ``feat_type`` selects one
    from :mod:`voicesdk.nn.feat` (``use_feats=False`` or ``feat_type=None``
    takes precomputed ``(B, F, T)`` / ``(B, C, F, T)`` features); multi-channel
    outputs are flattened into the input dimension.

    ``stem="conv"`` is :class:`CausalConvStem` (``subsampling`` divides the frame
    rate), ``"linear"`` a per-frame projection.  ``pooling="stats"`` takes the
    mean and std of the frames, ``"mean"`` the mean only.  The BatchNorm after
    the pooling removes the component shared by all utterances: without it the
    mean of the (LayerNorm-ed) frames is nearly the same vector for every input
    and the embedding collapses (pairwise cosine ~0.6-0.8 at init vs ~0 here).

    ``forward`` returns the embedding ``(B, embed_dim)``, the usual backbone
    interface; ``return_frames=True`` returns :class:`StreamingSpeakerOutput`.
    ``offset`` is the absolute index of the first (encoder) frame (RoPE), for
    running on consecutive chunks; attention and the stem do not reach across
    chunks.
    """

    def __init__(
        self,
        encoder: nn.Module,
        d_model: int,
        *,
        num_frequencies: int = 80,
        spec_dims: int = 1,
        embed_dim: int = 192,
        hop_length: int = 160,
        feat_type: str | None = "pt",
        spec_params: dict | None = None,
        use_feats: bool = True,
        stem: tp.Literal["conv", "linear"] = "conv",
        subsampling: int = 1,
        pooling: tp.Literal["stats", "mean"] = "stats",
    ) -> None:
        super().__init__()
        spec_params = {} if spec_params is None else dict(spec_params)
        if not use_feats or feat_type is None:
            self.spec = None
        elif feat_type in {"pt", "pt_mel"}:
            self.spec = features.MelBanks(n_mels=num_frequencies, hop_length=hop_length, **spec_params)
        elif feat_type in {"tf", "tf_mel"}:
            self.spec = features_tf.TFMelBanks(n_mels=num_frequencies, hop_length=hop_length, **spec_params)
        elif feat_type == "tf_spec":
            self.spec = features_tf.TFSpectrogram(**spec_params)
        elif feat_type == "pt_stft":
            self.spec = features.STFT(**spec_params)
        else:
            raise ValueError(f"unsupported feat_type: {feat_type!r}")

        if self.spec is not None:
            # Frontends do not expose their output size uniformly; probe it.
            with torch.no_grad():
                probe = self.spec(torch.zeros(1, 16000))
            if probe.ndim == 3:
                probe = probe.unsqueeze(1)
            spec_dims, num_frequencies = probe.shape[1], probe.shape[2]

        self.spec_dims = spec_dims
        self.num_frequencies = num_frequencies
        self.d_model = d_model
        self.embedding_dim = embed_dim

        in_dim = spec_dims * num_frequencies
        if stem == "conv":
            self.input_proj: nn.Module = CausalConvStem(in_dim, d_model, subsampling)
        elif stem == "linear":
            if subsampling != 1:
                raise ValueError("subsampling needs stem='conv'")
            self.input_proj = nn.Linear(in_dim, d_model)
        else:
            raise ValueError(f"unsupported stem: {stem!r}")
        self.encoder = encoder
        if pooling not in ("stats", "mean"):
            raise ValueError(f"unsupported pooling: {pooling!r}")
        self.pooling = pooling
        pooled_dim = d_model * (2 if pooling == "stats" else 1)
        self.pool_norm = nn.BatchNorm1d(pooled_dim)
        self.embedding = nn.Linear(pooled_dim, embed_dim)
        self.apply(_init_weights)

    def _prepare_input(self, x: Tensor) -> Tensor:
        if self.spec is not None:
            x = self.spec(x)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError("features must have shape (B, F, T) or (B, C, F, T)")
        if x.shape[1] != self.spec_dims:
            raise ValueError(f"expected {self.spec_dims} feature channels, got {x.shape[1]}")
        if x.shape[2] != self.num_frequencies:
            raise ValueError(f"expected {self.num_frequencies} frequency bins, got {x.shape[2]}")
        return x.flatten(1, 2).transpose(1, 2)  # (B, T, C*F)

    def forward(
        self,
        x: Tensor,
        *,
        offset: int = 0,
        return_frames: bool = False,
    ) -> StreamingSpeakerOutput | Tensor:
        h, aux = self.encoder(self.input_proj(self._prepare_input(x)), offset)
        pooled = h.mean(dim=1)
        if self.pooling == "stats":
            pooled = torch.cat([pooled, h.float().var(dim=1).clamp_min(1e-5).sqrt().to(h.dtype)], dim=-1)
        embedding = self.embedding(self.pool_norm(pooled))
        if not return_frames:
            return embedding
        return StreamingSpeakerOutput(embedding=embedding, frames=h, aux=aux)

    def extract_embedding(self, x: Tensor) -> Tensor:
        """Return only the utterance-level speaker embedding."""
        output = self.forward(x)
        assert isinstance(output, Tensor)
        return output


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _size(arch: str, size: str | None, d_model, n_heads, n_layers) -> tuple[int, int, int]:
    preset = {} if size is None else dict(PRESETS[arch][size])
    for key, value in (("d_model", d_model), ("n_heads", n_heads), ("n_layers", n_layers)):
        if value is not None:
            preset[key] = value
    missing = {"d_model", "n_heads", "n_layers"} - preset.keys()
    if missing:
        raise ValueError(f"give size= or {sorted(missing)}")
    return preset["d_model"], preset["n_heads"], preset["n_layers"]


def RoPESpeaker(
    size: str | None = "s0",
    *,
    d_model: int | None = None,
    n_heads: int | None = None,
    n_layers: int | None = None,
    dropout: float = 0.1,
    **kwargs: tp.Any,
) -> StreamingSpeakerEncoder:
    """RoPE transformer speaker encoder; ``kwargs`` go to :class:`StreamingSpeakerEncoder`."""
    d_model, n_heads, n_layers = _size("rope", size, d_model, n_heads, n_layers)
    return StreamingSpeakerEncoder(RoPEEncoder(d_model, n_heads, n_layers, dropout), d_model, **kwargs)


def TSCOSpeaker(
    size: str | None = "s0",
    *,
    d_model: int | None = None,
    n_heads: int | None = None,
    n_layers: int | None = None,
    n_scales: int = 3,
    dropout: float = 0.1,
    consistency_weight: float = 0.0,
    **kwargs: tp.Any,
) -> StreamingSpeakerEncoder:
    """TSCO speaker encoder; ``kwargs`` go to :class:`StreamingSpeakerEncoder`."""
    d_model, n_heads, n_layers = _size("tsco", size, d_model, n_heads, n_layers)
    encoder = TSCOEncoder(d_model, n_heads, n_layers, n_scales, dropout, consistency_weight)
    return StreamingSpeakerEncoder(encoder, d_model, **kwargs)


__all__ = [
    "PRESETS",
    "RoPEEncoder",
    "RoPESpeaker",
    "StreamingSpeakerEncoder",
    "StreamingSpeakerOutput",
    "TSCOEncoder",
    "TSCOSpeaker",
]
