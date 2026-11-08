"""Reference variant: mHuBERT-147 on top of ``torchaudio.models.hubert_base`` (no ``transformers`` / ``fairseq``).

Kept as an artifact / cross-check for ``mhubert_model.MHubert``: HF-named safetensors are renamed to the torchaudio
layout. Same interface (``forward(wavs) -> {"x", "hidden_states"}`` with 13 hidden states).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

from voicesdk.nn.arch.usad.usad_model import _read_safetensors


def _rename(key: str) -> str | None:
    if key == "masked_spec_embed":
        return None
    if key.startswith("feature_projection."):
        return "encoder." + key
    if key.startswith("encoder."):
        return "encoder.transformer." + key[len("encoder.") :]
    return key  # feature_extractor.* is identical


def hf_to_torchaudio_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        nk = _rename(k)
        if nk is None:
            continue
        # torchaudio uses torch.nn.utils.parametrizations.weight_norm: original0 = g, original1 = v
        nk = nk.replace("conv.weight_g", "conv.parametrizations.weight.original0")
        nk = nk.replace("conv.weight_v", "conv.parametrizations.weight.original1")
        out[nk] = v
    return out


class TorchaudioMHubert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torchaudio.models import hubert_base

        self.model = hubert_base(
            encoder_projection_dropout=0.0,
            encoder_attention_dropout=0.0,
            encoder_ff_interm_dropout=0.0,
            encoder_dropout=0.0,
            encoder_layer_drop=0.0,
        )

    encoder_dim = 768
    num_layers = 13

    def forward(self, wavs: torch.Tensor, normalize: bool = True) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        if normalize:
            wavs = (wavs - wavs.mean(dim=1, keepdim=True)) / torch.sqrt(wavs.var(dim=1, keepdim=True, unbiased=False) + 1e-7)
        m = self.model
        x, _ = m.feature_extractor(wavs, None)
        x = m.encoder.feature_projection(x)
        transformer = m.encoder.transformer
        states = [transformer._preprocess(x)] + transformer.get_intermediate_outputs(x)
        return {"x": states[-1], "hidden_states": states}


def load_mhubert_torchaudio(path: str | Path) -> TorchaudioMHubert:
    path = Path(path)
    path = path / "model.safetensors" if path.is_dir() else path
    model = TorchaudioMHubert()
    model.model.load_state_dict(hf_to_torchaudio_state(_read_safetensors(path)), strict=True)
    return model.eval()
