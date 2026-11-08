#!/usr/bin/env python3
"""Speaker-embedding model (ResNetTF / ReDimNet) -> ONNX (+ onnxsim, OpenVINO load check, onnxruntime check).

Checkpoint: Lightning ckpt of the distillation (keys ``student_model.model_base.<k>``, detected automatically),
or a plain ``state_dict`` .pt. Input ``input`` is a waveform ``(B, L)`` (include-feats, default) or the features
``(B, n_mels, T)`` (exclude-feats); output ``output`` is the embedding ``(B, D)``. The batch axis is dynamic.

Every option can live in a yaml/json file (``--config``, see configs/); options given on the command line win.
Next to each ``x.onnx`` a ``x.onnx.yaml`` with the resolved options is written (usable as ``--config`` again).

Examples:
    python convert-verification.py --config configs/verification-resnettf50.yaml
    python convert-verification.py --config configs/verification-resnettf50.yaml --ckpt other.ckpt --dynamo
    python convert-verification.py --arch redimnet --model-cfg data/cfg-models/redimnet_M.yaml --ckpt last.ckpt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import add_export_args, parse_with_config, run_exports  # noqa: E402
from models import CNN_ARCHS, build_verification  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=CNN_ARCHS, help="required (command line or config)")
    p.add_argument("--model-cfg", "--cfg", dest="model_cfg", help="model yaml (data/cfg-models/*.yaml); required")
    p.add_argument("--ckpt", type=Path, help="required")
    p.add_argument("--prefix", default=None, help="key prefix in the checkpoint (default: detected)")
    add_export_args(p, default_seconds="1,3")
    args = parse_with_config(p, required=("arch", "model_cfg", "ckpt"))
    return 1 if run_exports(args, args.ckpt, lambda feats: build_verification(args, feats), ["output"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
