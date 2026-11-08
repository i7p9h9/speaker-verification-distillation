"""``--config`` support for the training scripts (shared by train-antispoof-usad.py / train-antispoof-mhubert.py).

* ``--config file.yaml|json`` (alias ``--cfg``) holds any command-line option (``lora_rank`` or ``lora-rank``).
  Priority: command line > config > built-in defaults.
* ``save_config`` writes the resolved options to ``<exp_dir>/config.yaml``; the file can be passed back to
  ``--config`` to start the same experiment again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

PLACEHOLDER = "put/your/path/here"


def parse_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("--config", "--cfg", dest="config", type=Path, default=None,
                        help="yaml/json with the options of this script; explicit command-line options override it")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", "--cfg", dest="config", type=Path)
    known, _ = pre.parse_known_args()
    if known.config is not None:
        with open(known.config) as f:
            raw = json.load(f) if known.config.suffix == ".json" else yaml.safe_load(f)
        if not isinstance(raw, dict):
            parser.error(f"{known.config}: the config must be a mapping of option -> value")
        cfg = {str(k).replace("-", "_"): v for k, v in raw.items() if not str(k).startswith("_")}  # "_meta": metadata
        valid = {a.dest for a in parser._actions} - {"help", "config"}
        unknown = sorted(set(cfg) - valid)
        if unknown:
            parser.error(f"{known.config}: unknown option(s) {unknown}; valid: {sorted(valid)}")
        parser.set_defaults(**cfg)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, (str, Path)) and PLACEHOLDER in str(value):
            parser.error(f"--{name.replace('_', '-')} is still a placeholder ({value}); fill it in the config or on the command line")
    return args


def save_config(args: argparse.Namespace, exp_dir: Path, script: str) -> Path:
    """Write the resolved options to ``exp_dir/config.yaml`` (reusable as ``--config``)."""
    import torch

    exp_dir.mkdir(parents=True, exist_ok=True)
    resolved = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "config"}
    meta = {"script": script, "torch": str(torch.__version__), "config_file": str(args.config) if args.config else None}
    path = exp_dir / "config.yaml"
    path.write_text(yaml.safe_dump({"_meta": meta, **resolved}, sort_keys=False, allow_unicode=True))
    return path


def make_batch_delay_callback(delay_ms: float):
    """Lightning callback: pause ``delay_ms`` after every training batch so the GPU can cool down.
    Synchronises CUDA first, so the pause is real idle time rather than overlapped with queued kernels."""
    import time

    import torch
    from pytorch_lightning.callbacks import Callback

    class _BatchDelay(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time.sleep(delay_ms / 1000.0)

    return _BatchDelay()
