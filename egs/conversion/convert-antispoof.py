#!/usr/bin/env python3
"""Anti-spoofing model -> ONNX (+ onnxsim, OpenVINO load check, onnxruntime check).

Architectures: resnettf, redimnet (full checkpoints of train-antispoof.py), usad-small, usad-base
(train-antispoof-usad.py), mhubert (train-antispoof-mhubert.py). For USAD / mHuBERT the checkpoint holds only the
adapter + head: the base weights come from --pretrained, LoRA / SingLoRA (type and rank are read from the
checkpoint) is merged into the weights, so the ONNX contains plain layers. Accepted: Lightning .ckpt or
``adapter-last.pt``.

Input ``input``: waveform ``(B, L)`` (include-feats, default; for USAD the Kaldi fbank is part of the graph) or
features (exclude-feats; mHuBERT has no separate features: it expects an already mean/var-normalised waveform).
Outputs (batch axis dynamic):
    output  (B, 3)  scaled cosine logits [live, replay, df]
    live    log p_live / (p_replay + p_df)   - one-vs-rest log-odds of p = softmax(output)
    spoof   log (p_replay + p_df) / p_live   (= -live)
    df      log p_df / (p_live + p_replay)
    replay  log p_replay / (p_live + p_df)

Every option can live in a yaml/json file (``--config``, see configs/); options given on the command line win.
Next to each ``x.onnx`` a ``x.onnx.yaml`` with the resolved options is written (usable as ``--config`` again).

Examples:
    python convert-antispoof.py --config configs/antispoof-usad-small.yaml
    python convert-antispoof.py --config configs/antispoof-mhubert.yaml --ckpt .../adapter-last.pt --regular --dynamo
    python convert-antispoof.py --arch resnettf --model-cfg data/cfg-models/resnettf_34.yaml --ckpt data/exps/a/last.ckpt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import add_export_args, parse_with_config, run_exports  # noqa: E402
from models import ARCHS, CNN_ARCHS, AntispoofModel, build_antispoof  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=ARCHS, help="required (command line or config)")
    p.add_argument("--ckpt", type=Path, help="required")
    p.add_argument("--model-cfg", "--cfg", dest="model_cfg", default=None, help="model yaml, required for resnettf / redimnet")
    g = p.add_argument_group("usad / mhubert")
    g.add_argument("--pretrained", default=None, help="base weights (USAD model.safetensors, or the mHuBERT dir); required for usad / mhubert")
    g.add_argument("--lora-alpha", type=float, default=16.0, help="must equal the --lora-alpha used in training (not stored in the ckpt)")
    g.add_argument("--lora-targets", default=None, help="module-name substrings, if they were changed in training")
    add_export_args(p, default_seconds="5")
    args = parse_with_config(p, required=("arch", "ckpt"))
    extra = ("model_cfg",) if args.arch in CNN_ARCHS else ("pretrained",)
    if any(getattr(args, n) in (None, "") for n in extra):
        p.error("--" + extra[0].replace("_", "-") + f" is required for --arch {args.arch} (command line or config)")
    return 1 if run_exports(args, args.ckpt, lambda feats: build_antispoof(args, feats), AntispoofModel.OUTPUT_NAMES) else 0


if __name__ == "__main__":
    raise SystemExit(main())
