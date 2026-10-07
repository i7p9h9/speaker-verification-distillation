"""ECAPA2 speaker embedding model with weakly supervised VAD output."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from voicesdk.nn.feat import features, features_tf

from .layers import (
    ChannelDependentAttentiveStatisticsPooling,
    GlobalFeatureExtractor,
    LocalFeatureExtractor,
)


class ECAPA2Output(NamedTuple):
    """ECAPA2 outputs.

    Attributes:
        embedding: Utterance-level speaker embedding of shape ``(B, D)``.
        attention: Frame-level VAD logits of shape ``(B, T)``. These are the
            channel-average pre-softmax attention scores from equation (5) of
            Thienpondt and Demuynck (Odyssey 2024).
    """

    embedding: torch.Tensor
    attention: torch.Tensor


class ECAPA2(nn.Module):
    """Hybrid 2-D CNN/TDNN ECAPA2 architecture.

    Feature extraction follows :class:`ReDimNetWrap`: ``feat_type`` selects a
    frontend from :mod:`voicesdk.nn.feat`, while ``use_feats=False`` accepts
    precomputed features. ``num_frequencies`` is passed directly to the local
    feature extractor and must match the frontend output.

    ``return_attention=False`` preserves the usual embedding-only backbone
    interface when integrating the model into existing training code.
    """

    def __init__(
        self,
        *,
        num_frequencies: int = 80,
        spec_dims: int = 1,
        embed_dim: int = 192,
        hop_length: int = 160,
        feat_type: str | None = "pt",
        spec_params: dict | None = None,
        use_feats: bool = True,
        lfe_channels: tuple[int, ...] = (164, 164, 164, 192, 192),
        lfe_repeats: tuple[int, ...] = (3, 4, 4, 4, 5),
        fwse_hidden_dim: int = 128,
        frequency_encoding: bool = True,
        gfe_hidden_channels: int = 1024,
        gfe_out_channels: int = 1536,
        pooling_hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.num_frequencies = num_frequencies
        self.spec_dims = spec_dims
        spec_params = {} if spec_params is None else dict(spec_params)
        if not use_feats or feat_type is None:
            self.spec = None
        elif feat_type in {"pt", "pt_mel"}:
            self.spec = features.MelBanks(
                n_mels=num_frequencies,
                hop_length=hop_length,
                **spec_params,
            )
        elif feat_type in {"tf", "tf_mel"}:
            self.spec = features_tf.TFMelBanks(
                n_mels=num_frequencies,
                hop_length=hop_length,
                **spec_params,
            )
        elif feat_type == "tf_spec":
            self.spec = features_tf.TFSpectrogram(**spec_params)
        elif feat_type == "pt_stft":
            self.spec = features.STFT(**spec_params)
        else:
            raise ValueError(f"unsupported feat_type: {feat_type!r}")

        self.local = LocalFeatureExtractor(
            num_frequencies=num_frequencies,
            in_channels=spec_dims,
            channels=lfe_channels,
            repeats=lfe_repeats,
            fwse_hidden_dim=fwse_hidden_dim,
            frequency_encoding=frequency_encoding,
        )
        self.global_extractor = GlobalFeatureExtractor(
            self.local.out_channels,
            self.local.out_frequencies,
            hidden_channels=gfe_hidden_channels,
            out_channels=gfe_out_channels,
        )
        self.pooling = ChannelDependentAttentiveStatisticsPooling(
            gfe_out_channels,
            hidden_channels=pooling_hidden_channels,
        )
        self.pool_norm = nn.BatchNorm1d(gfe_out_channels * 2)
        self.embedding = nn.Linear(gfe_out_channels * 2, embed_dim)
        self.embedding_dim = embed_dim

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.spec is not None:
            x = self.spec(x)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError("features must have shape (B, F, T) or (B, C, F, T)")
        if x.shape[1] != self.spec_dims:
            raise ValueError(f"expected {self.spec_dims} feature channels, got {x.shape[1]}")
        if x.shape[2] != self.num_frequencies:
            raise ValueError(
                f"expected {self.num_frequencies} frequency bins, got {x.shape[2]}"
            )
        return x

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = True,
    ) -> ECAPA2Output | torch.Tensor:
        x = self._prepare_input(x)
        x = self.local(x)
        x = self.global_extractor(x)
        pooled, channel_attention_logits = self.pooling(x)
        embedding = self.embedding(self.pool_norm(pooled))

        if not return_attention:
            return embedding
        vad_logits = channel_attention_logits.mean(dim=1)
        return ECAPA2Output(embedding=embedding, attention=vad_logits)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the utterance-level speaker embedding."""
        output = self.forward(x, return_attention=False)
        assert isinstance(output, torch.Tensor)
        return output

    def extract_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Return channel-averaged, pre-softmax frame-level VAD logits."""
        output = self.forward(x, return_attention=True)
        assert isinstance(output, ECAPA2Output)
        return output.attention


__all__ = ["ECAPA2", "ECAPA2Output"]
