"""``ResNetTF`` with ``SelfAttentionTF`` pooling for ``extract.py``.

:class:`ResNetTFVAD` exposes a ``pooling`` module returning
``(pooled, attention_logits)`` like the ECAPA models, so the extractor hooks it
unchanged:

* the input of ``pooling`` is the frame-level features ``H`` (``F x T``, F = 3072
  for the configs in ``egs/voxceleb-v2/data/cfg-models``),
* the attention logits ``E`` (``F x T``) are the pre-softmax output of the
  pooling attention.  With ``add_context`` the pooling appends the mean and std
  of ``H`` as two extra time steps; their logits are dropped.

The backbone downsamples time by 8, so the frames are 80 ms apart.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn

from voicesdk.nn.arch import ResNetTF


class _Pooling(nn.Module):
    def __init__(self, pooling: nn.Module) -> None:
        super().__init__()
        self.pool = pooling

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.pool(x, is_return_attention=True)
        return out.embedding, out.attention[..., : x.shape[-1]]


class ResNetTFVAD(nn.Module):
    """``forward(wave (B, S)) -> embedding (B, D)``; see the module docstring."""

    def __init__(self, model: ResNetTF) -> None:
        super().__init__()
        self.model = model
        self.pooling = _Pooling(model.pooling)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        m = self.model
        h = m.features(wave) if m.features is not None else wave.unsqueeze(1)
        h = m.pre_pool(m.backbone(h))
        pooled, _ = self.pooling(h)
        return m.head(pooled)

    @classmethod
    def from_config(cls, config: str | Path, ckpt: str | Path | None = None, prefix: str = "") -> "ResNetTFVAD":
        """``config``: yaml with ``model_args`` of :class:`ResNetTF`.  ``ckpt``: a
        state dict or a lightning checkpoint; ``prefix`` is stripped from its keys,
        by default ``student_model.model_base.`` if all keys have it."""
        with open(config) as f:
            cfg = dict(yaml.load(f, Loader=yaml.FullLoader))
        model = ResNetTF(**cfg.get("model_args", cfg))
        if ckpt is not None:
            state = torch.load(ckpt, map_location="cpu")
            state = state.get("state_dict", state)
            if not prefix and all(k.startswith("student_model.model_base.") for k in state):
                prefix = "student_model.model_base."
            if prefix:
                state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
            model.load_state_dict(state, strict=True)
        return cls(model)
