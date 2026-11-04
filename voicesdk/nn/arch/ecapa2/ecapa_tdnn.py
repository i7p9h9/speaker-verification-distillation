"""ECAPA-TDNN with frame-local CAS pooling and weakly supervised VAD output."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from voicesdk.nn.feat import features, features_tf

from ..ecapa import Conv1dReluBn, SE_Res2Block
from .layers import ChannelDependentAttentiveStatisticsPooling
from .model import ECAPA2Output


class ECAPA_TDNN_VAD(nn.Module):
    """Original ECAPA-TDNN backbone with the ECAPA2 attention pooling.

    The backbone follows Desplanques et al. (Interspeech 2020): a k=5 TDNN
    layer, three SE-Res2Blocks with dilations 2, 3 and 4, multi-layer feature
    aggregation and a 1x1 convolution to ``mfa_channels``.  Pooling is
    :class:`ChannelDependentAttentiveStatisticsPooling`; its channel-averaged
    pre-softmax scores are the VAD logits (equation (5) of Thienpondt and
    Demuynck, Odyssey 2024).  By default the attention depends on the current
    frame only; ``pooling_global_context=True`` adds the utterance mean and std
    to its input, as in the original ECAPA-TDNN and ECAPA2.
    ``pooling_channel_dependent=False`` uses one attention score per frame,
    shared by all channels, which is then the VAD logit itself.

    Feature extraction mirrors :class:`ECAPA2`.  Multi-channel frontends
    (e.g. ``pt_stft``) are flattened into the TDNN input dimension.
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
        channels: int = 512,
        mfa_channels: int = 1536,
        scale: int = 8,
        pooling_hidden_channels: int = 128,
        is_use_std_pool: bool = True,
        is_use_pooling_global_context: bool = False,
        is_channel_dependent_attention: bool = True,
    ) -> None:
        super().__init__()
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

        if self.spec is not None:
            # Frontends do not expose their output size uniformly; probe it.
            with torch.no_grad():
                probe = self.spec(torch.zeros(1, 16000))
            if probe.ndim == 3:
                probe = probe.unsqueeze(1)
            spec_dims, num_frequencies = probe.shape[1], probe.shape[2]

        self.spec_dims = spec_dims
        self.num_frequencies = num_frequencies

        self.layer1 = Conv1dReluBn(spec_dims * num_frequencies, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=scale)
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=scale)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=scale)
        self.mfa = nn.Conv1d(channels * 3, mfa_channels, kernel_size=1)

        self.pooling = ChannelDependentAttentiveStatisticsPooling(
            mfa_channels,
            hidden_channels=pooling_hidden_channels,
            is_use_std_pool=is_use_std_pool,
            global_context=is_use_pooling_global_context,
            channel_dependent=is_channel_dependent_attention,
        )

        increase = 2 if is_use_std_pool else 1
        self.pool_norm = nn.BatchNorm1d(mfa_channels * increase)
        self.embedding = nn.Linear(mfa_channels * increase, embed_dim)
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
            raise ValueError(f"expected {self.num_frequencies} frequency bins, got {x.shape[2]}")
        return x.flatten(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = True,
    ) -> ECAPA2Output | torch.Tensor:
        x = self._prepare_input(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        x = F.relu(self.mfa(torch.cat((out2, out3, out4), dim=1)))

        pooled, channel_attention_logits = self.pooling(x)
        embedding = self.embedding(self.pool_norm(pooled))

        if not return_attention:
            return embedding

        vad_logits = channel_attention_logits.mean(dim=1)

        return ECAPA2Output(embedding=embedding, attention=vad_logits, embeddings_frames=x)

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


__all__ = ["ECAPA_TDNN_VAD"]
