"""Minimal pure-PyTorch HuBERT-base encoder (mHuBERT-147), no ``transformers`` / ``fairseq``.

Reproduces ``transformers.HubertModel`` (post-LN, ``feat_extract_norm="group"``, ``do_stable_layer_norm=False``)
and uses the same parameter names, so ``model.safetensors`` of ``utter-project/mHuBERT-147`` loads strictly.
Inference-style model: no dropout, no layerdrop, no SpecAugment (``masked_spec_embed`` is kept only for loading).

    model = load_mhubert("/path/to/mhubert")          # dir with model.safetensors + config.json (or the file)
    out = model(wavs)                                  # wavs: (B, L) float, 16 kHz
    out["hidden_states"]                               # 13 x (B, T, 768): after pos-conv+LN, then after each layer
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from voicesdk.nn.arch.usad.usad_model import _read_safetensors


@dataclass
class MHubertConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    conv_dim: Tuple[int, ...] = (512,) * 7
    conv_kernel: Tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: Tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    layer_norm_eps: float = 1e-5

    @classmethod
    def from_json(cls, path: str | Path) -> "MHubertConfig":
        c = json.loads(Path(path).read_text())
        if c.get("do_stable_layer_norm") or c.get("feat_extract_norm", "group") != "group" or c.get("conv_bias"):
            raise ValueError(f"{path}: only the HuBERT-base layout (group norm, post-LN, no conv bias) is supported")
        keys = [f for f in cls.__dataclass_fields__ if f in c]
        return cls(**{k: tuple(c[k]) if isinstance(c[k], list) else c[k] for k in keys})


class ConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel: int, stride: int, group_norm: bool) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel, stride, bias=False)
        if group_norm:  # only the first layer: per-channel normalisation over time
            self.layer_norm = nn.GroupNorm(out_dim, out_dim, affine=True)
        else:
            self.layer_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return F.gelu(x)


class FeatureExtractor(nn.Module):
    def __init__(self, cfg: MHubertConfig) -> None:
        super().__init__()
        dims = (1,) + tuple(cfg.conv_dim)
        self.conv_layers = nn.ModuleList(
            ConvLayer(dims[i], dims[i + 1], cfg.conv_kernel[i], cfg.conv_stride[i], group_norm=(i == 0))
            for i in range(len(cfg.conv_dim))
        )

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:  # (B, L) -> (B, C, T)
        x = wavs.unsqueeze(1)
        for layer in self.conv_layers:
            x = layer(x)
        return x


class FeatureProjection(nn.Module):
    def __init__(self, cfg: MHubertConfig) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(cfg.conv_dim[-1], eps=cfg.layer_norm_eps)
        self.projection = nn.Linear(cfg.conv_dim[-1], cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.layer_norm(x))


class WeightNormConv1d(nn.Module):
    """Conv1d with the legacy weight-norm layout (``weight_g`` (1,1,K), ``weight_v``), as stored in HF/fairseq."""

    def __init__(self, dim: int, kernel: int, groups: int) -> None:
        super().__init__()
        self.kernel, self.groups = kernel, groups
        self.weight_g = nn.Parameter(torch.ones(1, 1, kernel))
        self.weight_v = nn.Parameter(torch.empty(dim, dim // groups, kernel))
        self.bias = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.weight_v, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch._weight_norm(self.weight_v, self.weight_g, 2)
        return F.conv1d(x, weight, self.bias, padding=self.kernel // 2, groups=self.groups)


class PositionalConvEmbedding(nn.Module):
    def __init__(self, cfg: MHubertConfig) -> None:
        super().__init__()
        self.conv = WeightNormConv1d(cfg.hidden_size, cfg.num_conv_pos_embeddings, cfg.num_conv_pos_embedding_groups)
        self.drop_last = cfg.num_conv_pos_embeddings % 2 == 0  # even kernel -> one extra frame

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        y = self.conv(x.transpose(1, 2))
        if self.drop_last:
            y = y[..., :-1]
        return F.gelu(y).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = (p(x).view(b, t, self.heads, d // self.heads).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        y = F.scaled_dot_product_attention(q, k, v)  # scale = head_dim ** -0.5
        return self.out_proj(y.transpose(1, 2).reshape(b, t, d))


class FeedForward(nn.Module):
    def __init__(self, dim: int, inner: int) -> None:
        super().__init__()
        self.intermediate_dense = nn.Linear(dim, inner)
        self.output_dense = nn.Linear(inner, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_dense(F.gelu(self.intermediate_dense(x)))


class EncoderLayer(nn.Module):
    """Post-LN block: x = LN(x + attn(x)); x = LN2(x + ffn(x))."""

    def __init__(self, cfg: MHubertConfig) -> None:
        super().__init__()
        d, eps = cfg.hidden_size, cfg.layer_norm_eps
        self.attention = Attention(d, cfg.num_attention_heads)
        self.layer_norm = nn.LayerNorm(d, eps=eps)
        self.feed_forward = FeedForward(d, cfg.intermediate_size)
        self.final_layer_norm = nn.LayerNorm(d, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x + self.attention(x))
        return self.final_layer_norm(x + self.feed_forward(x))


class Encoder(nn.Module):
    def __init__(self, cfg: MHubertConfig) -> None:
        super().__init__()
        self.pos_conv_embed = PositionalConvEmbedding(cfg)
        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.num_hidden_layers))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.layer_norm(x + self.pos_conv_embed(x))
        states = [x]
        for layer in self.layers:
            x = layer(x)
            states.append(x)
        return states


class MHubert(nn.Module):
    def __init__(self, cfg: Optional[MHubertConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or MHubertConfig()
        self.feature_extractor = FeatureExtractor(self.cfg)
        self.feature_projection = FeatureProjection(self.cfg)
        self.encoder = Encoder(self.cfg)
        self.masked_spec_embed = nn.Parameter(torch.zeros(self.cfg.hidden_size))  # unused, for strict loading

    @property
    def encoder_dim(self) -> int:
        return self.cfg.hidden_size

    @property
    def num_layers(self) -> int:
        """Number of returned hidden states (input of layer 1 + every layer output)."""
        return self.cfg.num_hidden_layers + 1

    def forward(self, wavs: torch.Tensor, normalize: bool = True) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        """wavs: (B, L) float 16 kHz. ``normalize``: per-utterance zero-mean / unit-variance (preprocessor_config)."""
        if normalize:
            wavs = (wavs - wavs.mean(dim=1, keepdim=True)) / torch.sqrt(wavs.var(dim=1, keepdim=True, unbiased=False) + 1e-7)
        x = self.feature_extractor(wavs).transpose(1, 2)
        states = self.encoder(self.feature_projection(x))
        return {"x": states[-1], "hidden_states": states}

    @classmethod
    def load_from_safetensors(cls, weights_path: str | Path, config_path: str | Path | None = None) -> "MHubert":
        weights_path = Path(weights_path)
        config_path = Path(config_path) if config_path else weights_path.with_name("config.json")
        model = cls(MHubertConfig.from_json(config_path) if config_path.is_file() else MHubertConfig())
        model.load_state_dict(_read_safetensors(weights_path), strict=True)
        return model.eval()


def load_mhubert(path: str | Path) -> MHubert:
    """``path``: a dir with model.safetensors + config.json, or the .safetensors file itself."""
    path = Path(path)
    return MHubert.load_from_safetensors(path / "model.safetensors" if path.is_dir() else path)
