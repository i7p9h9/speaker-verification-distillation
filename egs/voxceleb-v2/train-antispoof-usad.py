"""Anti-spoofing fine-tuning of USAD 2.0 (Base / Small) with LoRA (minLoRA).

Based on ``train-antispoof.py``: same data pipeline, loss weighting strategy and
AM-Softmax head. What differs:

  * backbone  - pretrained USAD 2.0 encoder (local fairseq ``usad2_{small,base}.pt``, vendored code in ``voicesdk.nn.arch.usad``),
                frozen, adapted only through LoRA on its Linear layers;
  * trainable - LoRA matrices + layer-weights + pooling projection + AM head;
  * validation - 20% hold-out split of ASVspoof 2017 and (optional) ReMASC, EER for each.

Usage::

    python train-antispoof-usad.py --model base  --asv17-dir /path/to/ASV17_eval --remasc-dir /path/to/ReMASC
    python train-antispoof-usad.py --model small --lora-rank 16 --lora-alpha 32
"""

import argparse
import os
import shutil
import typing as tp
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
from audimentation import AddNoise, FileListAudioProvider, OneOf, Reverb, SequentialCompose
from minlora.model import LoRAParametrization, apply_lora
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from validation import AggregatedAntiSpoofingValidator, AntiSpoofingValidator, VoxDataset
from validation.utils.get_spoof_datasets import get_asv_spoof_17_holdout, get_remasc

from voicesdk.dataset import (
    LabeledAggregatedDataset,
    LabeledSource,
    StrategyMomentumLossWeighting,
    WeightedDataset,
    sources_from_subfolders,
)
from voicesdk.distillation.data import (
    AudioReaderBegin,
    AudioReaderTelSimulated,
    collate_batch_labeled_segments_fn,
    collate_batch_segments_fn,
)
from voicesdk.distillation.training.trainer_antispoof import AntispoofLightningModule, WeightLogger
from voicesdk.nn.arch.usad import UsadModel
from voicesdk.nn.loss import AMSoftmaxLoss
from voicesdk.utils.find_files import find_files_recursive

# ---------------------------------------------------------------------------
# Config (defaults; most of them can be overridden from the CLI, see parse_args)
# ---------------------------------------------------------------------------

# size -> (encoder_dim, default local checkpoint). Checkpoints are the fairseq ``usad2_*.pt`` files from
# https://github.com/vectominist/usad/tree/main/usad_inference#download-checkpoints
USAD_MODELS: tp.Dict[str, tp.Tuple[int, str]] = {
    "small": (384, "data/pretrained/usad2_small.pt"),
    "base": (768, "data/pretrained/usad2_base.pt"),
}

# Linear layers inside the Conformer blocks that receive LoRA (substring match on module name).
# Attention projections + both feed-forward modules; framewise_in_proj stays frozen.
DEFAULT_LORA_TARGETS: str = "query_proj,key_proj,value_proj,out_proj,ffn_1,ffn_2"

# AM-Softmax head
NUM_CLASSES: int = 3  # 0 = bonafide, 1 = replay (LA), 2 = synthetic (DF)
AM_SCALE: float = 20.0
AM_MARGIN: float = 0.15

# Dataset roots
PATH_TRAIN_VOX1: str = "/media/ssd/voice/datasets/vox1/"
PATH_TRAIN_VOX2: str = "/media/ssd/voice/datasets/vox2/dev-16k/aac/"
PATH_TRAIN_SPGI: str = "/media/ssd/voice/datasets/spgispeech/"
PATH_TRAIN_TIDY_1: str = "/media/ssd/voice/datasets/TidyVoiceX/"
PATH_TRAIN_TIDY_2: str = "/media/ssd/voice/datasets/TidyVoiceX2/"
PATH_TRAIN_VCTK: str = "/media/ssd/voice/datasets/vctk/"
PATH_TRAIN_LIBRI: str = "/media/ssd/voice/datasets/librispeech/"
PATH_TRAIN_MCV13: str = "/media/ssd/voice/datasets/mcv13"
PATH_TRAIN_VOXTUBE: str = "/media/ssd/voice/datasets/voxtube"
PATH_TRAIN_ASV21: str = "/media/ssd/voice/datasets/ASV21Eval"

PATH_TRAIN_CN_CELEB: str = "/media/ssd/voice/datasets/cn-celeb_v2/CN-Celeb_wav-16k"
PATH_TRAIN_LIVE_CML: str = "/media/ssd/voice/datasets/antispoof/live/cml"

# Synthesis
PATH_TRAIN_VC_TTS: str = "/media/ssd/voice/datasets/antispoof/vc_tts_engines-16k/data"
PATH_SYN_TTS_COMMON: str = "/media/ssd/voice/datasets/antispoof/synthesis-train/tts"
PATH_SYN_VOCODERS_V1: str = "/media/ssd/voice/datasets/antispoof/synthesis-train/vocoders-v1/train/synthes"
PATH_SYN_VC_VOL1: str = "/media/ssd/voice/datasets/antispoof/synthesis-train/voice_clones/train/synthes"
PATH_SYN_VC_VOL2: str = "/media/ssd/voice/datasets/antispoof/synthesis-train/voice-clone-vol2/train"

# Dataset PAD
PATH_PAD: str = "/media/ssd/voice/datasets/antispoof/replay/"

# Augmentation
DIR_RIR = Path("/media/ssd/voice/datasets/RIRs/RIRS_NOISES/")
DIR_NOISE = Path("/media/ssd/voice/datasets/musan/")

# Validation: ASVspoof 2017 root with meta.csv (path,type,spk,phrase,env,playback_device,recording_device)
# and the eval/ folder. Can be overridden with --asv17-dir.
DIR_ASV_17: str = "/path/to/ASV17_eval"


def disable_native_triton_ops_without_compiler() -> None:
    """torch>=2.13 overrides some eager ops (``bmm``) with Triton kernels that are JIT-compiled.

    Without a C compiler (e.g. the slim dev container) the first such call crashes in backward
    with ``Failed to find C compiler``; in that case fall back to the regular ATen kernels.
    """
    if os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc"):
        return
    try:
        from torch._native import registry

        registry.deregister_op_overrides(disable_dsl_names="triton")
        print("[env] no C compiler found: torch native Triton op overrides disabled")
    except Exception as e:  # private torch API - never block training because of it
        print(f"[env] could not disable torch native Triton overrides: {e!r}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_usad(ckpt_path: str, expected_dim: int) -> UsadModel:
    """Load a USAD 2.0 encoder from a local fairseq checkpoint (vendored ``voicesdk.nn.arch.usad``)."""
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"USAD checkpoint not found: {ckpt_path} (set --usad-ckpt)")
    model = UsadModel.load_from_fairseq_ckpt(ckpt_path)
    if model.encoder_dim != expected_dim:
        raise ValueError(f"{ckpt_path}: encoder_dim={model.encoder_dim}, expected {expected_dim} for this --model")
    return model


class UsadBackbone(nn.Module):
    """USAD 2.0 encoder -> utterance embedding ``(B, embedding_dim)``.

    Pipeline: waveform -> (frozen + LoRA) USAD encoder -> learnable softmax-weighted
    sum of all layer outputs -> mean+std pooling over time -> Linear.

    Input is a float waveform batch ``(B, L)`` at 16 kHz in ``[-1, 1]`` (peak-normalised
    by the audio reader); fbank extraction is done inside USAD.
    """

    def __init__(
        self,
        ckpt_path: str,
        expected_dim: int,
        embedding_dim: int,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_targets: tp.Sequence[str],
    ) -> None:
        super().__init__()
        self.usad = load_usad(ckpt_path, expected_dim)
        self.num_layers: int = self.usad.num_layers
        dim: int = self.usad.encoder_dim

        # 1) freeze the whole pretrained encoder, 2) inject LoRA (new params are trainable).
        self.usad.requires_grad_(False)
        self.n_lora_layers = add_lora_to_linears(
            self.usad, lora_targets, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout
        )

        self.layer_logits = nn.Parameter(torch.zeros(self.num_layers))
        self.proj = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, embedding_dim))

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        out = self.usad(wavs=wavs)
        hidden = torch.stack(out["hidden_states"], dim=0)  # (N, B, T, D)
        w = torch.softmax(self.layer_logits, dim=0).to(hidden.dtype)
        x = torch.einsum("n,nbtd->btd", w, hidden)
        # Segments have fixed length (short files are tiled by the reader) -> no padding mask needed.
        stats = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)
        return self.proj(stats)


def add_lora_to_linears(
    model: nn.Module,
    targets: tp.Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> int:
    """Attach minLoRA parametrizations to ``nn.Linear`` layers whose name contains a target substring.

    Unlike ``minlora.add_lora_by_name`` each matching ``nn.Linear`` is patched exactly once
    (USAD wraps ``nn.Linear`` into a custom ``Linear`` module, so name-based ``model.apply``
    would register LoRA twice on the inner layer).
    """
    lora_config = {
        nn.Linear: {
            "weight": partial(
                LoRAParametrization.from_linear,
                rank=rank,
                lora_alpha=alpha,
                lora_dropout_p=dropout,
            ),
        },
    }
    matched = [
        layer
        for name, layer in model.named_modules()
        if type(layer) is nn.Linear and any(t in name for t in targets)
    ]
    if not matched:
        raise ValueError(f"No nn.Linear layer matches LoRA targets {list(targets)}")
    for layer in matched:
        apply_lora(layer, lora_config=lora_config)
    return len(matched)


def trainable_state_dict(module: nn.Module) -> tp.Dict[str, torch.Tensor]:
    """State dict restricted to trainable parameters (LoRA + head)."""
    names = {n for n, p in module.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in module.state_dict().items() if k in names}


def get_model(args: argparse.Namespace) -> tp.Tuple[UsadBackbone, AMSoftmaxLoss]:
    backbone = UsadBackbone(
        ckpt_path=args.usad_ckpt or USAD_MODELS[args.model][1],
        expected_dim=USAD_MODELS[args.model][0],
        embedding_dim=args.embedding_dim,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
    )
    am_loss = AMSoftmaxLoss(
        embedding_dim=args.embedding_dim,
        num_classes=NUM_CLASSES,
        scale=AM_SCALE,
        margin=AM_MARGIN,
    )

    n_total = sum(p.numel() for p in backbone.parameters())
    n_train = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in am_loss.parameters())
    print(
        f"[model] USAD2-{args.model} ({args.usad_ckpt or USAD_MODELS[args.model][1]}): LoRA on {backbone.n_lora_layers} Linear layers "
        f"(r={args.lora_rank}, alpha={args.lora_alpha}); trainable backbone {n_train / 1e6:.2f}M "
        f"/ {n_total / 1e6:.2f}M (+ AM head {n_head / 1e3:.1f}K)"
    )
    return backbone, am_loss


class SaveAdapterCallback(Callback):
    """Saves only trainable weights (LoRA + head) - a few MB instead of a full checkpoint."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: tp.Dict[str, tp.Any]
    ) -> None:
        """Drop frozen USAD weights from every Lightning checkpoint (ModelCheckpoint, last.ckpt, ...)."""
        frozen = {n for n, p in pl_module.named_parameters(remove_duplicate=False) if not p.requires_grad}
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if k not in frozen}

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backbone": trainable_state_dict(pl_module.backbone),
                "am_loss": pl_module.am_loss.state_dict(),
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
            },
            self.path,
        )


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def backup_script(log_dir: Path) -> None:
    """Copy the current script to log_dir/scripts/."""
    src = Path(__file__).resolve()
    dst_dir = log_dir / "scripts"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    print(f"Script backed up: {src} -> {dst}")


# ---------------------------------------------------------------------------
# Validation: 20% of ASVspoof 2017
# ---------------------------------------------------------------------------


def get_validation(
    reader,
    asv17_dir: str,
    fraction: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    remasc_dir: tp.Optional[str] = None,
):
    """Validators: 20% hold-out of ASVspoof 2017 and (optionally) the full ReMASC test set.

    Every validator reports its own EER (``val/spoof-joined/per_validator/<name>/eer``).
    """
    data_loader = partial(
        DataLoader,
        batch_size=batch_size,
        collate_fn=collate_batch_segments_fn,
        num_workers=num_workers,
        persistent_workers=False,
        shuffle=False,
        drop_last=False,
    )

    def make_validator(name: str, datasets) -> AntiSpoofingValidator:
        n_live, n_spoof = len(datasets.dataset_live), len(datasets.dataset_spoof)
        print(f"[{name}] live: {n_live} | spoof: {n_spoof} | total: {n_live + n_spoof}")
        return AntiSpoofingValidator(
            name=name,
            bonafide_loader=data_loader(datasets.dataset_live),
            spoof_loader=data_loader(datasets.dataset_spoof),
        )

    validators = [
        make_validator(
            "asv_spoof_17",
            get_asv_spoof_17_holdout(
                dir_main=asv17_dir, reader=reader, fraction=fraction, subset="val", seed=seed
            ),
        )
    ]
    if remasc_dir:
        validators.append(make_validator("remasc", get_remasc(dir_main=remasc_dir, reader=reader, seed=seed)))

    return AggregatedAntiSpoofingValidator(name="spoof-joined", validators=validators, return_scores=False)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def build_augmentation_pipeline() -> SequentialCompose:
    rir_provider_point = FileListAudioProvider(paths=find_files_recursive(DIR_RIR / "pointsource_noises", extension=".wav"))
    rir_provider_real = FileListAudioProvider(paths=find_files_recursive(DIR_RIR / "real_rirs_isotropic_noises", extension=".wav"))
    rir_provider_sim = FileListAudioProvider(paths=find_files_recursive(DIR_RIR / "simulated_rirs", extension=".wav"))

    noise_provider_music = FileListAudioProvider(paths=find_files_recursive(DIR_NOISE / "music", extension=".wav"))
    noise_provider_noise = FileListAudioProvider(paths=find_files_recursive(DIR_NOISE / "noise", extension=".wav"))
    noise_provider_speech = FileListAudioProvider(paths=find_files_recursive(DIR_NOISE / "speech", extension=".wav"))

    return SequentialCompose(
        stages=[
            OneOf(
                stages=[
                    Reverb(rir_provider=rir_provider_point, name="reverb_point", wet_dry_range=(0.2, 0.5)),
                    Reverb(rir_provider=rir_provider_real, name="reverb_real", wet_dry_range=(0.2, 0.5)),
                    Reverb(rir_provider=rir_provider_sim, name="reverb_sim", wet_dry_range=(0.2, 0.5)),
                ],
                weights=[1, 3, 1],
                name="reverb",
                p=0.01,
            ),
            OneOf(
                stages=[
                    AddNoise(noise_provider=noise_provider_music, name="noise_music", snr_range=(5.0, 10.0)),
                    AddNoise(noise_provider=noise_provider_noise, name="noise_noise", snr_range=(3.0, 10.0)),
                    AddNoise(noise_provider=noise_provider_speech, name="noise_speech", snr_range=(10.0, 30.0)),
                ],
                weights=[3, 6, 1],
                name="noise",
                p=1.0,
            ),
        ],
        p=0.05,
    )


# ---------------------------------------------------------------------------
# Train dataset (same sources as train-antispoof.py)
# ---------------------------------------------------------------------------


def build_train_dataset(reader: tp.Any) -> LabeledAggregatedDataset:
    """Assemble labeled training sources. Bonafide -> 0, replay -> 1, synthetic -> 2."""
    dataset_factory = partial(VoxDataset, reader=reader)

    label_live = 0
    label_la = 1
    label_df = 2 if NUM_CLASSES == 3 else 1

    create_spoof_source = partial(
        sources_from_subfolders,
        factory=dataset_factory,
        ignore_empty=False,
        prefix="legacy-",
    )
    create_live_source = partial(
        sources_from_subfolders,
        factory=dataset_factory,
        ignore_empty=False,
    )

    def single(root: str, label: int, weight: float, name: str) -> LabeledSource:
        return LabeledSource(
            dataset=[WeightedDataset(VoxDataset(reader=reader, root=root))],
            label=label,
            weight=weight,
            name=name,
        )

    sources: tp.List[LabeledSource] = [
        # bonafide
        single(PATH_TRAIN_VOX1, label_live, 6.0, "vox1"),
        single(PATH_TRAIN_VOX2, label_live, 6.0, "vox2"),
        single(PATH_TRAIN_SPGI, label_live, 2.0, "spgispeech"),
        single(PATH_TRAIN_TIDY_1, label_live, 1.0, "tidy_voice_1"),
        single(PATH_TRAIN_TIDY_2, label_live, 1.0, "tidy_voice_2"),
        single(PATH_TRAIN_VCTK, label_live, 1.0, "VCTK"),
        single(PATH_TRAIN_MCV13, label_live, 1.0, "MCV-13"),
        single(PATH_TRAIN_VOXTUBE, label_live, 2.0, "voxtube"),
        single(PATH_TRAIN_LIBRI, label_live, 3.0, "libri_speech"),
        single(PATH_TRAIN_ASV21, label_live, 1.0, "asv-21"),
        single(PATH_TRAIN_CN_CELEB, label_live, 1.0, "cn-celeb-v2"),
    ]
    sources += create_live_source(root=PATH_TRAIN_LIVE_CML, label=label_live, source_weight=0.8, prefix="cml-")

    # spoof - LA (replay)
    sources.append(single(f"{PATH_PAD}/voxceleb_toloka_replays_2026_02_15-16k", label_la, 4.0, "Replay-02-15"))
    sources.append(single(f"{PATH_PAD}/voxceleb_toloka_replays_2026_02_21-16k", label_la, 4.0, "Replay-02-21"))

    # spoof - DF (synthetic)
    sources += create_spoof_source(
        root=PATH_SYN_TTS_COMMON, label=label_df, source_weight=2.0, ignore_pattern="*tts_test_set*"
    )
    sources += create_spoof_source(root=PATH_TRAIN_VC_TTS, label=label_df, source_weight=0.8)
    sources += create_spoof_source(root=PATH_SYN_VOCODERS_V1, label=label_df, source_weight=1.2)
    sources += create_spoof_source(root=PATH_SYN_VC_VOL1, label=label_df, source_weight=0.5)
    sources += create_spoof_source(
        root=PATH_SYN_VC_VOL2, label=label_df, ignore_pattern="*resamble_denoiser*", source_weight=1.0
    )

    return LabeledAggregatedDataset(sources=sources, name="train_dataset")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(USAD_MODELS), default="base", help="USAD 2.0 size")
    p.add_argument("--usad-ckpt", default=None, help="local usad2_*.pt checkpoint (default: data/pretrained/usad2_<model>.pt)")
    p.add_argument("--embedding-dim", type=int, default=256)

    g = p.add_argument_group("LoRA (minLoRA)")
    g.add_argument("--lora-rank", type=int, default=8)
    g.add_argument("--lora-alpha", type=float, default=16.0)
    g.add_argument("--lora-dropout", type=float, default=0.0)
    g.add_argument("--lora-targets", default=DEFAULT_LORA_TARGETS, help="comma-separated module-name substrings")

    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--weight-decay", type=float, default=1e-3)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--num-workers", type=int, default=8)
    g.add_argument("--steps-per-epoch", type=int, default=5000)
    g.add_argument("--max-epochs", type=int, default=30)
    g.add_argument("--warmup-ratio", type=float, default=0.5, help="warmup length in epochs")
    g.add_argument("--min-lr-ratio", type=float, default=0.0)
    g.add_argument("--segment-ms", type=int, default=5000)
    g.add_argument("--precision", default="bf16-mixed")
    g.add_argument("--resume", default=None, help="Lightning checkpoint to resume from")

    g = p.add_argument_group("validation")
    g.add_argument("--asv17-dir", default=DIR_ASV_17, help="ASVspoof 2017 root (meta.csv + eval/)")
    g.add_argument("--remasc-dir", default=None, help="ReMASC root (meta.csv + eval/); validator skipped if unset")
    g.add_argument("--val-fraction", type=float, default=0.2)
    g.add_argument("--val-batch-size", type=int, default=64)
    g.add_argument("--val-num-workers", type=int, default=2)
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("logging")
    g.add_argument("--log-dir", default="data/exps/")
    g.add_argument("--exp-name", default=None, help="default: antispoof-usad2-<model>-lora")
    g.add_argument("--log-weights-every", type=int, default=100)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    disable_native_triton_ops_without_compiler()
    exp_name = args.exp_name or f"antispoof-usad2-{args.model}-lora"
    total_steps = args.steps_per_epoch * args.max_epochs
    pl.seed_everything(args.seed)

    # ---- data ------------------------------------------------------------
    # Peak normalisation keeps the waveform in [-1, 1], which is what USAD's fbank expects.
    reader_train = AudioReaderTelSimulated(norm_type="peak", length_segment_ms=args.segment_ms, p_tel=0.0)
    reader_val = AudioReaderBegin(norm_type="peak", length_segment_ms=args.segment_ms)

    dataset_train = build_train_dataset(reader_train)
    strategy = StrategyMomentumLossWeighting(dataset_train, momentum=0.985, scale=7.0)
    loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        collate_fn=collate_batch_labeled_segments_fn,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
    )

    backbone, am_loss = get_model(args)

    # ---- loggers / callbacks ---------------------------------------------
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name=exp_name)
    exp_dir = Path(args.log_dir) / exp_name
    weight_logger = WeightLogger(
        tb_logger=tb_logger,
        csv_path=exp_dir / "dataset_weights.csv",
        log_every_n_steps=args.log_weights_every,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(exp_dir / "checkpoints"),
        filename="{epoch}-{step}",
        save_top_k=3,
        monitor="train/loss",
        mode="min",
        save_last=True,
    )
    adapter_cb = SaveAdapterCallback(exp_dir / "adapter-last.pt")

    # ---- validation ------------------------------------------------------
    validator = get_validation(
        reader=reader_val,
        asv17_dir=args.asv17_dir,
        fraction=args.val_fraction,
        batch_size=args.val_batch_size,
        num_workers=args.val_num_workers,
        seed=args.seed,
        remasc_dir=args.remasc_dir,
    )

    # ---- lightning module ------------------------------------------------
    module = AntispoofLightningModule(
        backbone=backbone,
        am_loss=am_loss,
        dataset=dataset_train,
        strategy=strategy,
        validators=[validator],
        weight_logger=weight_logger,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=int(args.warmup_ratio * args.steps_per_epoch),
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
        log_every_n_steps=50,
        aug_pipeline=build_augmentation_pipeline(),
    )

    # Checkpoints contain only trainable weights (see SaveAdapterCallback) -> load them non-strictly
    # on resume; the frozen USAD weights come from the pretrained USAD checkpoint.
    module.strict_loading = False

    # ---- trainer ---------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        logger=tb_logger,
        callbacks=[checkpoint_cb, adapter_cb],
        enable_progress_bar=True,
        log_every_n_steps=50,
        limit_train_batches=args.steps_per_epoch,
        limit_val_batches=0,  # validators run in AntispoofLightningModule.on_train_epoch_end
    )

    backup_script(exp_dir)
    trainer.fit(module, train_dataloaders=loader_train, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
