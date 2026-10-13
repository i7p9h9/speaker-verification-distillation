"""x-vector extractor presets of the VBx reference (github.com/BUTSpeechFIT/VBx).

Paths are relative to ``vbx_models`` of ``../config.yaml`` (the reference's
``VBx/models`` with the added models).
``frontend``: ``fbank`` - 64 log Mel bands, windows of ``seg_len`` frames
every ``seg_jump`` frames (``extract_xvectors_melbank``); ``pcm`` - raw audio
windows of the model's input size every 0.24 s, normalized per window
(``extract_xvectors_pcm``).  The ``plda`` is a Kaldi file or a directory with
``mean.npy``, ``transform.npy``, ``psi.npy``; the ``threshold`` (AHC) is the
one of the reference runs (``run-test.sh``).
"""

from __future__ import annotations

import sys
import typing as tp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import diar_config  # noqa: E402

PRESETS: tp.Dict[str, tp.Dict[str, tp.Any]] = {
    "resnet101-8k": dict(
        weights="ResNet101_8kHz/nnet/final.onnx",
        frontend="fbank",
        sample_rate=8000,
        plda="ResNet101_8kHz/plda",
        xvec_transform="ResNet101_8kHz/transform.h5",
        threshold=-0.015,
    ),
    "resnet101-16k": dict(
        weights="ResNet101_16kHz/nnet/final.onnx",
        frontend="fbank",
        sample_rate=16000,
        plda="ResNet101_16kHz/plda",
        xvec_transform="ResNet101_16kHz/transform.h5",
        threshold=-0.015,
    ),
    "redimnet": dict(
        weights="redimnet/redimnet-finish-1sec.onnx",
        frontend="pcm",
        sample_rate=16000,
        plda="redimnet/vox2-aug-02-plda",
        xvec_transform=None,
        threshold=-0.010,
    ),
    "redimnet-tel": dict(
        weights="redimnet-tel/epoch=13-step=70000-1sec.onnx",
        frontend="pcm",
        sample_rate=16000,
        plda="redimnet-tel/vox2-aug-02-plda",
        xvec_transform=None,
        threshold=-0.015,
    ),
    "redimnet-tel-small": dict(
        weights="redimnet-tel-small/epoch=23-step=120000-1sec.onnx",
        frontend="pcm",
        sample_rate=16000,
        plda="redimnet-tel-small/plda",
        xvec_transform=None,
        threshold=-0.1,
    ),
}


def preset(name: str) -> tp.Dict[str, tp.Any]:
    """The preset with absolute paths."""
    if name not in PRESETS:
        raise SystemExit(f"unknown model {name}, one of {', '.join(PRESETS)}")
    out = dict(PRESETS[name])
    for key in ("weights", "plda", "xvec_transform"):
        if out[key] is not None:
            out[key] = diar_config.get_path("vbx_models") / out[key]
    return out
