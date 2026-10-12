"""Original pre-trained ECAPA2 (``Jenthe/ECAPA2`` on Hugging Face) for ``extract.py``.

The TorchScript model only exposes the embedding and pooled statistics through
its ``labels`` argument, and forward hooks do not fire inside a scripted graph.
:class:`OriginalECAPA2` therefore re-runs the scripted ``forward`` step by step
from Python, calling the scripted submodules, and exposes a Python ``pooling``
module returning ``(pooled, attention_logits)`` like the voicesdk models, so the
extractor hooks it unchanged:

* the input of ``pooling`` is the frame-level features ``H`` (1536 x T),
* the attention logits ``E`` (1536 x T) are the pre-softmax output of
  ``pooling_1[0].attention`` on ``[H, mean(H), std(H)]``; the original model
  uses this global context, so they depend on the whole window.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

HF_REPO = "Jenthe/ECAPA2"
HF_FILE = "ecapa2.pt"

# (block, shortcut) of the local feature extractor in the order of the scripted
# forward: shortcut "id" adds the block input, "none" adds nothing, any other
# name is a downsampling submodule applied to the block input.
_LFE = [
    ("conv_2", "id"), ("conv_17", "id"),
    ("conv_3", "downsample_1"), ("conv_4", "id"), ("conv_5", "id"), ("conv_18", "id"),
    ("conv_6", "downsample_2"), ("conv_7", "id"), ("conv_8", "none"), ("conv_19", "id"),
    ("conv_9", "downsample_3"), ("conv_10", "id"), ("conv_11", "id"), ("conv_20", "id"),
    ("conv_12", "downsample_4"), ("conv_13", "id"), ("conv_14", "id"), ("conv_15", "id"), ("conv_16", "id"),
]  # fmt: skip


class _ScriptedPooling(nn.Module):
    """``pooling_1`` of the scripted model, also returning the attention logits."""

    def __init__(self, pooling: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.pool = getattr(pooling, "0")
        self.norm = getattr(pooling, "1")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=2, keepdim=True).expand_as(x)
        std = x.var(dim=2, keepdim=True).add(1e-5).sqrt().expand_as(x)
        logits = self.pool.attention(torch.cat((x, mean, std), dim=1))
        return self.norm(self.pool(x)), logits


class OriginalECAPA2(nn.Module):
    """``forward(wave (B, S)) -> embedding (B, 192)``; see the module docstring."""

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__()
        if path is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(HF_REPO, HF_FILE)
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self.pooling = _ScriptedPooling(self.model.pooling_1)

    def frames(self, wave: torch.Tensor) -> torch.Tensor:
        """Frame-level features ``H`` of shape ``(B, 1536, T)``."""
        m = self.model
        x = m.pre_processing(wave).squeeze(1)
        x = m.conv_1(x.to(getattr(m.conv_1, "0").weight.dtype).unsqueeze(1))
        for block, shortcut in _LFE:
            y = getattr(m, block)(x)
            if shortcut == "id":
                y = y + x
            elif shortcut != "none":
                y = y + getattr(m, shortcut)(x)
            x = y
        x = x.flatten(1, 2)
        tdnn = [getattr(m.tdnn_1, str(i)) for i in range(3)]
        x = tdnn[2](tdnn[1](tdnn[0](x)))
        x = m.tdnn_2(x) + x
        return getattr(m.dense_1, "1")(getattr(m.dense_1, "0")(x))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.pooling(self.frames(wave))
        return self.model.utterance_level_feature_extraction(pooled)
