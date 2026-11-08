"""Shared pieces of the ONNX conversion scripts (see convert-verification.py / convert-antispoof.py).

Export variants are the product of two independent switches (both can be given at once -> every combination):
    mode : regular (``torch.onnx.export(dynamo=False)``)  |  dynamo (``dynamo=True``, needs ``onnxscript``)
    feats: include-feats (waveform in, front-end inside the graph)  |  exclude-feats (features in)
Defaults: regular + include-feats.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import types
import typing as tp
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch import nn

SAMPLE_RATE = 16000
EXPORT_BATCH = 2  # example batch; the batch axis is exported as dynamic
EGS_VOXCELEB = Path(__file__).resolve().parents[1] / "voxceleb-v2"  # `validation` classes may be pickled in ckpts


# ---------------------------------------------------------------------------
# CLI + config file
# ---------------------------------------------------------------------------

PLACEHOLDER = "put/your/path/here"  # marker used in configs/ for values that must be filled in


def load_config_file(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: the config must be a mapping of option -> value")
    return cfg


def parse_with_config(parser: argparse.ArgumentParser, required: tp.Sequence[str] = ()) -> argparse.Namespace:
    """``--config file.yaml|json`` holds any command-line option (``--lora-alpha`` as ``lora_alpha`` or ``lora-alpha``).
    Priority: command line > config > built-in defaults. ``required`` are checked after merging."""
    parser.add_argument("--config", type=Path, default=None, help="yaml/json with the options below; explicit command-line options override it")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args()
    if known.config is not None:
        cfg = {str(k).replace("-", "_"): v for k, v in load_config_file(known.config).items() if not str(k).startswith("_")}  # "_export": sidecar metadata
        valid = {a.dest for a in parser._actions} - {"help", "config"}
        unknown = sorted(set(cfg) - valid)
        if unknown:
            parser.error(f"{known.config}: unknown option(s) {unknown}; valid: {sorted(valid)}")
        if isinstance(cfg.get("seconds"), (list, tuple)):
            cfg["seconds"] = ",".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in cfg["seconds"])
        elif isinstance(cfg.get("seconds"), (int, float)):
            cfg["seconds"] = f"{cfg['seconds']:g}"
        for key in ("ckpt", "out_dir"):
            if cfg.get(key) is not None:
                cfg[key] = Path(cfg[key])
        parser.set_defaults(**cfg)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, (str, Path)) and PLACEHOLDER in str(value):
            parser.error(f"--{name.replace('_', '-')} is still a placeholder ({value}); fill it in the config or pass it on the command line")
    missing = [n for n in required if getattr(args, n) in (None, "")]
    if missing:
        parser.error("missing (not on the command line, not in the config): " + ", ".join("--" + n.replace("_", "-") for n in missing))
    return args


def write_sidecar(onnx_path: Path, args: argparse.Namespace, mode: str, include_feats: bool, seconds: float) -> None:
    """``<name>.onnx.yaml`` next to every onnx: the resolved options (config + command line) it was made with,
    so the file can be traced back to its checkpoint. The file can be passed to ``--config`` to repeat the export."""
    resolved = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "config"}
    resolved.update(seconds=f"{seconds:g}", regular=(mode == "regular"), dynamo=(mode == "dynamo"),
                    include_feats=include_feats, exclude_feats=not include_feats)
    meta = {"onnx": onnx_path.name, "torch": str(torch.__version__), "config_file": str(args.config) if args.config else None}
    path = onnx_path.with_name(onnx_path.name + ".yaml")
    path.write_text(yaml.safe_dump({"_export": meta, **resolved}, sort_keys=False))


def add_export_args(p: argparse.ArgumentParser, default_seconds: str) -> None:
    g = p.add_argument_group("export variants (combine freely: every combination is exported)")
    B = argparse.BooleanOptionalAction  # --x / --no-x, so the command line can also switch a config value off
    g.add_argument("--regular", action=B, default=None, help="legacy TorchScript exporter, dynamo=False (default mode)")
    g.add_argument("--dynamo", action=B, default=None, help="torch.export based exporter, dynamo=True (needs onnxscript)")
    g.add_argument("--include-feats", action=B, default=None, help="front-end (fbank/mel) inside the graph, waveform input (default)")
    g.add_argument("--exclude-feats", action=B, default=None, help="features as the input, no front-end in the graph")
    g = p.add_argument_group("output")
    g.add_argument("--seconds", default=default_seconds, help="comma-separated example lengths, one onnx per value")
    g.add_argument("--out-dir", type=Path, default=None, help="default: next to the checkpoint")
    g.add_argument("--opset", type=int, default=None)
    g.add_argument("--simplify", action=B, default=True, help="onnxsim pass (skipped if onnxsim is not installed)")
    g.add_argument("--openvino", action=B, default=True, help="OpenVINO load check (skipped if openvino is not installed)")
    g.add_argument("--verify", action=B, default=True, help="onnxruntime vs torch comparison (skipped if onnxruntime is not installed)")


def variants(args: argparse.Namespace) -> tp.List[tp.Tuple[str, bool]]:
    modes = [m for m, on in (("regular", args.regular), ("dynamo", args.dynamo)) if on] or ["regular"]
    feats = [f for f, on in ((True, args.include_feats), (False, args.exclude_feats)) if on] or [True]
    return [(m, f) for m, f in itertools.product(modes, feats)]


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def read_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return dict(yaml.load(f, Loader=yaml.FullLoader))


def load_checkpoint(path: str | Path) -> tp.Any:
    """torch.load for our own checkpoints. Lightning ckpts may pickle classes of the training scripts
    (validators); if they cannot be imported they are replaced by empty stubs - we only need the tensors."""
    sys.path.insert(0, str(EGS_VOXCELEB))
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except (ImportError, AttributeError):
        class _Stub:
            def __init__(self, *a, **k):
                pass

            def __setstate__(self, state):
                pass

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except Exception:
                    return _Stub

        lenient = types.ModuleType("lenient_pickle")
        lenient.Unpickler, lenient.load = _Unpickler, pickle.load
        return torch.load(path, map_location="cpu", weights_only=False, pickle_module=lenient)
    finally:
        sys.path.pop(0)


def read_state_dict(path: str | Path) -> tp.Dict[str, torch.Tensor]:
    """Flat state dict from a Lightning ckpt (``state_dict``), an adapter file ``{"backbone": .., "am_loss": ..}``
    (keys get ``backbone.`` / ``am_loss.`` prefixes) or a plain ``state_dict`` file."""
    ck = load_checkpoint(path)
    if isinstance(ck, dict) and "state_dict" in ck:
        return dict(ck["state_dict"])
    if isinstance(ck, dict) and "backbone" in ck and "am_loss" in ck:
        return {**{f"backbone.{k}": v for k, v in ck["backbone"].items()}, **{f"am_loss.{k}": v for k, v in ck["am_loss"].items()}}
    if isinstance(ck, dict):
        return dict(ck)
    raise ValueError(f"{path}: unsupported checkpoint format ({type(ck).__name__})")


def find_prefix(state: tp.Dict[str, torch.Tensor], keys: tp.Sequence[str]) -> str:
    """Prefix P such that ``P + key`` is in ``state`` for as many model keys as possible (e.g. ``student_model.model_base.``)."""
    sample = list(keys)[:20]
    candidates = Counter(k[: len(k) - len(m)] for k in state for m in sample if k.endswith(m))
    if not candidates:
        raise KeyError(f"no checkpoint key matches model keys such as {sample[:3]}; checkpoint keys: {list(state)[:5]}")
    return max(candidates, key=lambda p: sum((p + k) in state for k in keys))


FEATURE_PREFIXES = ("features.", "spec.", "fbank.", "ssl.fbank.")  # non-trainable front-end buffers, may be absent in a ckpt


def load_into(model: nn.Module, state: tp.Dict[str, torch.Tensor], prefix: tp.Optional[str] = None, what: str = "model") -> str:
    """Copy ``state[prefix + k]`` into every tensor of ``model``. Missing front-end buffers only warn; any other
    missing key (or a shape mismatch) is an error. Returns the prefix used."""
    own = model.state_dict()
    prefix = find_prefix(state, list(own)) if prefix is None else prefix
    missing = [k for k in own if prefix + k not in state]
    hard = [k for k in missing if not k.startswith(FEATURE_PREFIXES)]
    if hard:
        raise KeyError(f"{what}: {len(hard)} keys missing in the checkpoint (prefix {prefix!r}), e.g. {hard[:5]}")
    if missing:
        print(f"[load] {what}: {len(missing)} front-end tensors not in the checkpoint, keeping the freshly built ones")
    with torch.no_grad():
        for k, v in own.items():
            if prefix + k in state:
                if v.shape != state[prefix + k].shape:
                    raise ValueError(f"{what}: {k}: model {tuple(v.shape)} vs checkpoint {tuple(state[prefix + k].shape)}")
                v.copy_(state[prefix + k])
    print(f"[load] {what}: {len(own) - len(missing)}/{len(own)} tensors from prefix {prefix!r}")
    return prefix


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def example_waveform(seconds: float) -> torch.Tensor:
    return torch.randn(EXPORT_BATCH, int(round(seconds * SAMPLE_RATE)))


def onnx_name(ckpt: Path, seconds: float, include_feats: bool, mode: str, tag: str = "") -> str:
    s = f"{seconds:g}"
    return f"{ckpt.stem}{tag}-{s}sec{'' if include_feats else '-nofeat'}{'-dynamo' if mode == 'dynamo' else ''}.onnx"


def export_onnx(
    model: nn.Module,
    example: torch.Tensor,
    path: Path,
    mode: str,
    output_names: tp.Sequence[str],
    opset: tp.Optional[int] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    kw: dict = dict(input_names=["input"], output_names=list(output_names))
    if opset is not None:
        kw["opset_version"] = opset
    dyn_axes = {"input": {0: "batch"}, **{n: {0: "batch"} for n in output_names}}
    with torch.no_grad():
        if mode == "dynamo":
            torch.onnx.export(model, (example,), f=path, dynamo=True, dynamic_shapes=({0: torch.export.Dim("batch")},), external_data=False, **kw)
        else:
            torch.onnx.export(model, (example,), f=path, dynamo=False, dynamic_axes=dyn_axes, **kw)


def simplify(path: Path) -> tp.Optional[Path]:
    try:
        import onnx
        from onnxsim import simplify as onnxsim_simplify
    except ImportError:
        print("  [sim] onnxsim is not installed - skipped")
        return None
    out = path.with_name(path.stem + "-sim.onnx")
    model_simp, ok = onnxsim_simplify(onnx.load(path))
    if not ok:
        print("  [sim] onnxsim could not validate the simplified model - skipped")
        return None
    onnx.save(model_simp, out)
    print(f"  [sim] {out.name}")
    return out


def check_openvino(path: Path) -> None:
    try:
        import openvino as ov
    except ImportError:
        print("  [ov] openvino is not installed - skipped")
        return
    ov.Core().compile_model(path.as_posix(), "AUTO")
    ov.convert_model(path.as_posix())
    print(f"  [ov] {path.name}: compiled and converted")


def verify_onnx(path: Path, model: nn.Module, example: torch.Tensor, output_names: tp.Sequence[str], atol: float = 1e-3) -> None:
    """Compare onnxruntime with torch on a new random input of another batch size (checks the dynamic batch axis)."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [ort] onnxruntime is not installed - numeric check skipped")
        return
    x = torch.randn(EXPORT_BATCH + 1, *example.shape[1:])
    with torch.no_grad():
        ref = model.eval()(x)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    sess = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    got = sess.run(list(output_names), {"input": x.numpy()})
    diffs = {n: float(abs(g - r.numpy()).max()) for n, g, r in zip(output_names, got, ref)}
    status = "OK" if max(diffs.values()) <= atol else "MISMATCH"
    print(f"  [ort] {status}: max |onnx - torch| = " + ", ".join(f"{n}={d:.2e}" for n, d in diffs.items()))
    if status != "OK":
        raise RuntimeError(f"{path.name}: onnxruntime output differs from torch (atol={atol})")


def run_exports(
    args: argparse.Namespace,
    ckpt: Path,
    build: tp.Callable[[bool], tp.Tuple[nn.Module, tp.Callable[[float], torch.Tensor]]],
    output_names: tp.Sequence[str],
    tag: str = "",
) -> int:
    """``build(include_feats) -> (module, example_fn(seconds))``. Returns the number of failed variants."""
    out_dir = args.out_dir or ckpt.parent
    seconds_list = [float(s) for s in args.seconds.split(",") if s.strip()]
    failed = 0
    for include_feats in sorted({f for _, f in variants(args)}, reverse=True):
        model, example_fn = build(include_feats)
        model.eval()
        for mode in [m for m, f in variants(args) if f == include_feats]:
            for sec in seconds_list:
                path = out_dir / onnx_name(ckpt, sec, include_feats, mode, tag)
                print(f"[export] {path.name}  (mode={mode}, feats={'in' if include_feats else 'out'}, {sec:g}s)")
                try:
                    example = example_fn(sec)
                    export_onnx(model, example, path, mode, output_names, args.opset)
                    import onnx

                    onnx.checker.check_model(path.as_posix())
                    if args.verify:
                        verify_onnx(path, model, example, output_names)
                    final = (simplify(path) or path) if args.simplify else path
                    if args.openvino:
                        check_openvino(final)
                    write_sidecar(path, args, mode, include_feats, sec)
                except Exception as e:  # keep going with the other variants
                    failed += 1
                    print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"Done, {failed} failed variant(s)")
    return failed
