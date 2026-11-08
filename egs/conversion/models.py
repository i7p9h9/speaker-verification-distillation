"""Model builders for the conversion scripts: weights are loaded, LoRA is merged, the result is a plain nn.Module."""

from __future__ import annotations

import typing as tp
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from common import SAMPLE_RATE, EXPORT_BATCH, FEATURE_PREFIXES, example_waveform, find_prefix, load_into, read_state_dict, read_yaml
from minlora.model import LoRAParametrization, SingLoRAParametrization, apply_lora
from torch import nn

from voicesdk.nn.arch import ReDimNetWrap, ResNetTF
from voicesdk.nn.arch.mhubert import DEFAULT_LORA_TARGETS as MHUBERT_LORA_TARGETS
from voicesdk.nn.arch.mhubert import load_mhubert
from voicesdk.nn.arch.usad import UsadConvertible, UsadModel
from voicesdk.nn.loss import AMSoftmaxLoss

ARCHS = ("resnettf", "redimnet", "usad-small", "usad-base", "mhubert")
CNN_ARCHS = ("resnettf", "redimnet")
USAD_LORA_TARGETS = "query_proj,key_proj,value_proj,out_proj,ffn_1,ffn_2"
NUM_CLASSES = 3  # 0 live, 1 replay, 2 df
AM_SCALE = 20.0

ExampleFn = tp.Callable[[float], torch.Tensor]


def make_trace_safe() -> None:
    """``StackFreqDim`` reshapes with ``int(x.size(0))``, which bakes the example batch size into the traced graph
    (a model exported with batch 2 then fails with another batch). ``flatten(1, 2)`` is the same op, shape-agnostic."""
    from voicesdk.nn.arch.resnettf import StackFreqDim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.convert_mode_on:  # original branch, kept as is
            return torch.cat(torch.split(x, 1, dim=1), dim=2)[:, 0, :, :]
        if self.permute_chan_freq:
            x = x.permute((0, 2, 1, 3))
        return x.flatten(1, 2)

    StackFreqDim.forward = forward


make_trace_safe()


# ---------------------------------------------------------------------------
# CNN backbones (ResNetTF / ReDimNet)
# ---------------------------------------------------------------------------


def build_cnn(arch: str, cfg_path: str, include_feats: bool) -> tp.Tuple[nn.Module, ExampleFn]:
    """Full model (front-end included in the weights); for ``include_feats=False`` the front-end is detached
    and the example input is what it would have produced."""
    cfg = read_yaml(cfg_path)
    cls = ResNetTF if arch == "resnettf" else ReDimNetWrap
    model = cls(**cfg["model_args"]).eval()
    attr = "features" if arch == "resnettf" else "spec"
    frontend = getattr(model, attr)
    if frontend is None:
        raise ValueError(f"{cfg_path}: the model has no feature front-end (use_feats=False)")
    if include_feats:
        return model, example_waveform

    setattr(model, attr, None)

    def example(seconds: float) -> torch.Tensor:
        with torch.no_grad():
            feats = frontend.eval()(example_waveform(seconds))
        # the CNN adds the channel axis itself when it has no front-end: (B, 1, F, T) -> (B, F, T)
        return feats[:, 0] if feats.ndim == 4 and feats.shape[1] == 1 else feats

    return model, example


# ---------------------------------------------------------------------------
# SSL backbones (USAD / mHuBERT) + LoRA
# ---------------------------------------------------------------------------


class SslBackbone(nn.Module):
    """Mirror of ``UsadBackbone`` / ``MHubertBackbone`` of the training scripts (attribute ``ssl`` for both):
    encoder -> softmax-weighted sum of hidden states -> mean+std pooling -> LayerNorm + Linear."""

    def __init__(self, ssl: nn.Module, kind: str, dim: int, num_layers: int, embedding_dim: int, include_feats: bool) -> None:
        super().__init__()
        self.ssl, self.kind, self.include_feats = ssl, kind, include_feats
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.proj = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "mhubert":  # include_feats -> per-utterance mean/var normalisation inside
            hidden = self.ssl(x, normalize=self.include_feats)["hidden_states"]
        else:
            hidden = self.ssl(x)
        hidden = torch.stack(list(hidden), dim=0)  # (N, B, T, D)
        w = torch.softmax(self.layer_logits, dim=0).to(hidden.dtype)
        h = torch.einsum("n,nbtd->btd", w, hidden)
        return self.proj(torch.cat([h.mean(dim=1), h.std(dim=1)], dim=-1))


def add_lora_to_linears(model: nn.Module, targets: tp.Sequence[str], rank: int, alpha: float, lora_type: str) -> int:
    ctor = SingLoRAParametrization.from_linear if lora_type == "singlora" else LoRAParametrization.from_linear
    config = {nn.Linear: {"weight": partial(ctor, rank=rank, lora_alpha=alpha, lora_dropout_p=0.0)}}
    matched = [m for n, m in model.named_modules() if type(m) is nn.Linear and any(t in n for t in targets)]
    if not matched:
        raise ValueError(f"no nn.Linear matches the LoRA targets {list(targets)}")
    for layer in matched:
        apply_lora(layer, lora_config=config)
    return len(matched)


def merge_lora(model: nn.Module) -> int:
    """Fold every parametrization into its weight (W + dW) and remove it. Returns the number of merged layers."""
    n = 0
    for module in list(model.modules()):
        if hasattr(module, "parametrizations"):
            for name in list(module.parametrizations.keys()):
                parametrize.remove_parametrizations(module, name, leave_parametrized=True)
            n += 1
    return n


def infer_lora(backbone_state: tp.Dict[str, torch.Tensor]) -> tp.Optional[tp.Tuple[str, int]]:
    """(lora_type, rank) from the adapter tensors, None if the checkpoint has no LoRA."""
    a = next((v for k, v in backbone_state.items() if k.endswith(".lora_A")), None)
    if a is None:
        return None
    has_b = any(k.endswith(".lora_B") for k in backbone_state)
    return ("lora", a.shape[0]) if has_b else ("singlora", a.shape[1])


def build_ssl_backbone(
    arch: str,
    pretrained: str,
    state: tp.Dict[str, torch.Tensor],
    embedding_dim: int,
    include_feats: bool,
    lora_alpha: float,
    lora_targets: tp.Optional[str],
) -> tp.Tuple[nn.Module, ExampleFn]:
    if arch == "mhubert":
        ssl, kind = load_mhubert(pretrained), "mhubert"
        dim, num_layers, targets = ssl.encoder_dim, ssl.num_layers, lora_targets or MHUBERT_LORA_TARGETS
    else:
        usad = UsadModel.load_from_safetensors(pretrained)
        ssl, kind = UsadConvertible(usad, include_feats), "usad"
        dim, num_layers, targets = ssl.encoder_dim, ssl.num_layers, lora_targets or USAD_LORA_TARGETS
    ssl.requires_grad_(False)
    backbone = SslBackbone(ssl, kind, dim, num_layers, embedding_dim, include_feats)

    # checkpoint keys: backbone.usad.* (USAD script) / backbone.ssl.* (mHuBERT script) -> backbone.ssl.*
    bstate = {k[len("backbone.") :].replace("usad.", "ssl.", 1) if k.startswith("backbone.usad.") else k[len("backbone.") :]: v
              for k, v in state.items() if k.startswith("backbone.")}
    lora = infer_lora(bstate)
    if lora is not None:
        lora_type, rank = lora
        n = add_lora_to_linears(backbone.ssl, [t.strip() for t in targets.split(",") if t.strip()], rank, lora_alpha, lora_type)
        print(f"[lora] {lora_type} r={rank} alpha={lora_alpha} on {n} layers (alpha is not stored in the checkpoint - must match training)")
    trainable = {k for k, p in backbone.named_parameters() if p.requires_grad}
    unexpected = [k for k in bstate if k not in backbone.state_dict() and not k.startswith(FEATURE_PREFIXES)]
    if unexpected:
        raise KeyError(f"{len(unexpected)} checkpoint keys do not fit the model (wrong --arch / rank?), e.g. {unexpected[:4]}")
    absent = sorted(trainable - set(bstate))
    if absent:
        raise KeyError(f"checkpoint lacks trainable tensors, e.g. {absent[:4]}")
    with torch.no_grad():
        for k, v in backbone.state_dict().items():
            if k in bstate:
                v.copy_(bstate[k])
    print(f"[load] backbone: {len(bstate)} tensors from the checkpoint, the rest from {pretrained}")
    if lora is not None:
        print(f"[lora] merged {merge_lora(backbone)} layers")

    if include_feats:
        return backbone, example_waveform

    def example(seconds: float) -> torch.Tensor:  # normalised mel / normalised waveform
        wav = example_waveform(seconds)
        if kind == "mhubert":
            return wav
        from voicesdk.nn.arch.usad import UsadFbank

        with torch.no_grad():
            return UsadFbank(mel_dim=usad.mel_dim)(wav)

    return backbone, example


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


class AntispoofModel(nn.Module):
    """backbone -> AM-Softmax cosine head. Outputs (all ``(B, ...)``):

    output : (B, 3)  scaled cosine logits, classes [live, replay, df]
    live   : log p_live / (p_replay + p_df)      one-vs-rest log-odds, p = softmax(logits)
    spoof  : log (p_replay + p_df) / p_live      = -live
    df     : log p_df / (p_live + p_replay)
    replay : log p_replay / (p_live + p_df)

    (the literal ``ln(p_a - (p_b + p_c))`` is undefined whenever p_a < p_b + p_c, the log-odds is its smooth form).
    """

    OUTPUT_NAMES = ("output", "live", "spoof", "df", "replay")

    def __init__(self, backbone: nn.Module, am_loss: AMSoftmaxLoss) -> None:
        super().__init__()
        self.backbone = backbone
        self.am_loss = am_loss

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, ...]:
        emb = F.normalize(self.backbone(x), p=2, dim=1)
        logits = F.linear(emb, F.normalize(self.am_loss.weight, p=2, dim=1)) * self.am_loss.scale
        live, replay, df = logits[:, 0], logits[:, 1], logits[:, 2]

        def lse(*t: torch.Tensor) -> torch.Tensor:
            return torch.logsumexp(torch.stack(t, dim=1), dim=1)

        spoof_lse = lse(replay, df)
        return logits, live - spoof_lse, spoof_lse - live, df - lse(live, replay), replay - lse(live, df)


def build_antispoof(args, include_feats: bool) -> tp.Tuple[nn.Module, ExampleFn]:
    state = read_state_dict(args.ckpt)
    head = {k[len("am_loss.") :]: v for k, v in state.items() if k.startswith("am_loss.")}
    if "weight" not in head:
        raise KeyError(f"{args.ckpt}: no am_loss.weight (is this an anti-spoofing checkpoint?)")
    num_classes, embedding_dim = head["weight"].shape
    am_loss = AMSoftmaxLoss(embedding_dim=embedding_dim, num_classes=num_classes, scale=AM_SCALE)
    load_into(am_loss, head, prefix="", what="am_loss")

    if args.arch in CNN_ARCHS:
        backbone, example = build_cnn(args.arch, args.model_cfg, include_feats)
        prefix = find_prefix({k: None for k in state if k.startswith("backbone.")}, list(backbone.state_dict()))
        load_into(backbone, state, prefix=prefix, what="backbone")
    else:
        backbone, example = build_ssl_backbone(
            args.arch, args.pretrained, state, embedding_dim, include_feats, args.lora_alpha, args.lora_targets
        )
    return AntispoofModel(backbone, am_loss).eval(), example


def build_verification(args, include_feats: bool) -> tp.Tuple[nn.Module, ExampleFn]:
    model, example = build_cnn(args.arch, args.model_cfg, include_feats)
    load_into(model, read_state_dict(args.ckpt), prefix=args.prefix, what="model")
    return model.eval(), example
