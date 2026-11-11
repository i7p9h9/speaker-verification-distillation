"""Anti-spoofing fine-tuning of mHuBERT-147 with LoRA / SingLoRA (minLoRA).

Copy of ``train-antispoof-usad.py`` (same data pipeline, validation, loss weighting, AM-Softmax head) with another
backbone: the multilingual HuBERT-base (``utter-project/mHuBERT-147``), frozen and adapted only through LoRA on the
Linear layers of its transformer blocks.

Backends (``--backend``), both without ``transformers`` / ``fairseq``, both load the HF ``model.safetensors``:
  * own         - minimal pure-PyTorch HuBERT, ``voicesdk.nn.arch.mhubert.MHubert`` (default);
  * torchaudio  - ``torchaudio.models.hubert_base`` with renamed weights (reference / cross-check).
They give identical hidden states.

Usage::

    python train-antispoof-mhubert.py --mhubert-ckpt /media/hdd/shared/ckpt/mhubert
    python train-antispoof-mhubert.py --backend torchaudio --lora-type singlora
"""

import argparse
import os
import shutil
import subprocess
import typing as tp
from dataclasses import dataclass, field, fields
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
from audimentation import AddNoise, FileListAudioProvider, OneOf, Reverb, SequentialCompose
from minlora.model import LoRAParametrization, SingLoRAParametrization, apply_lora, set_singlora_ramp
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from train_config import make_batch_delay_callback, parse_with_config, save_config
from validation import AggregatedAntiSpoofingValidator, AntiSpoofingValidator, GroupedAntiSpoofingValidator, VoxDataset
from validation.utils.get_spoof_datasets import get_asv_spoof_17_holdout, get_playbackspoof, get_remasc

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
from voicesdk.nn.arch.mhubert import DEFAULT_LORA_TARGETS, load_mhubert, load_mhubert_torchaudio
from voicesdk.nn.loss import AMSoftmaxLoss, SoftmaxLoss
from voicesdk.utils.find_files import find_files_recursive

# ---------------------------------------------------------------------------
# Config (defaults; most of them can be overridden from the CLI, see parse_args)
# ---------------------------------------------------------------------------

# HuggingFace snapshot utter-project/mHuBERT-147: <dir>/model.safetensors + <dir>/config.json
MHUBERT_CKPT = "/media/hdd/shared/ckpt/mhubert"
MHUBERT_DIM = 768
MHUBERT_BACKENDS = {"own": load_mhubert, "torchaudio": load_mhubert_torchaudio}

# AM-Softmax head
NUM_CLASSES: int = 3  # 0 = bonafide, 1 = replay (LA), 2 = synthetic (DF)
AM_SCALE: float = 20.0
AM_MARGIN: float = 0.15

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DIR_VERIF = "/media/shared/data/verification"
DIR_SPOOF = "/media/shared/data/spoof"

# Share of every PlaybackSpoof group used for validation (the full set is large and slow to evaluate).
PLAYBACKSPOOF_FRACTION: float = 0.25


@dataclass(frozen=True)
class LivePaths:
    """Bonafide (label 0)."""

    vox1: str = f"{DIR_VERIF}/vox1/"
    vox2: str = f"{DIR_VERIF}/vox2/dev-wav/aac/"
    # spgi: str = f"{DIR_VERIF}/spgispeech/"  # disabled on purpose
    tidy_1: str = f"{DIR_VERIF}/TidyVoice/TidyVoiceX_Train/"
    tidy_2: str = f"{DIR_VERIF}/TidyVoice/TidyVoiceX2_ASV/"
    vctk: str = f"{DIR_VERIF}/vctk/"
    libri: str = f"{DIR_VERIF}/librispeech/"
    mcv13: str = f"{DIR_VERIF}/mcv13"
    voxtube: str = f"{DIR_VERIF}/voxtube"
    asv21: str = f"{DIR_VERIF}/ASVspoof2021_DF_eval"
    cn_celeb: str = f"{DIR_VERIF}/CN-Celeb_wav/data"
    cml: str = f"{DIR_SPOOF}/live/cml"  # subfolder per source
    hifi_tts: str = f"{DIR_SPOOF}/live/hi_fi_tts_v0"
    playbackspoof: str = f"{DIR_SPOOF}/playbackspoof/audio/train/live/clean"  # .flac
    # ReMASC genuine, regrouped by egs/dataset-processing/regroup-remasc.py: <env>/*.wav, one source per environment.
    # Must NOT contain the validation files (regroup with --exclude-meta <validation original_meta.csv>).
    remasc: str = f"{DIR_SPOOF}/remasc/core/live"


@dataclass(frozen=True)
class ReplayPaths:
    """Presentation attacks: recorded playback (label 1)."""

    formanta: str = f"{DIR_SPOOF}/replay/data"
    playbackspoof: str = f"{DIR_SPOOF}/playbackspoof/audio/train/spoof/replay"  # <HH|HL|LH|LL>/*.flac
    # replayed TTS/VC: playback and re-recording, so it is a replay attack as well
    playbackspoof_deep_fake: str = f"{DIR_SPOOF}/playbackspoof/audio/train/spoof/replay-deep-fake"  # .flac
    # ReMASC replayed, regrouped by regroup-remasc.py: <env>/*.wav, one source per environment (see LivePaths.remasc)
    remasc: str = f"{DIR_SPOOF}/remasc/core/spoof/replay"


@dataclass(frozen=True)
class DeepFakePaths:
    """Synthetic speech without playback (label 2 for 3 classes)."""

    tts_common: str = f"{DIR_SPOOF}/synthesis/tts"  # subfolder per engine
    vc_tts: str = f"{DIR_SPOOF}/vc_tts_engines/"  # subfolder per engine, .ogg
    vocoders_v1: str = f"{DIR_SPOOF}/synthesis/vocoders-v1/train/synthes"
    vc_vol1: str = f"{DIR_SPOOF}/synthesis/voice_clones/train/synthes"
    vc_vol2: str = f"{DIR_SPOOF}/synthesis/voice-clone-vol2/train"
    playbackspoof: str = f"{DIR_SPOOF}/playbackspoof/audio/train/spoof/deep-fake"  # clean/*.flac


@dataclass(frozen=True)
class AugmentationPaths:
    rir: str = f"{DIR_VERIF}/RIRs/RIRS_NOISES/"
    noise: str = f"{DIR_VERIF}/musan/"


@dataclass(frozen=True)
class ValidationPaths:
    # ASVspoof 2017 root: meta.csv (path,type,spk,phrase,env,playback_device,recording_device) + eval/
    asv17: str = f"{DIR_SPOOF}/antispoof-testing/ASV17_eval"
    # ReMASC root: meta.csv + eval/
    remasc: str = f"{DIR_SPOOF}/antispoof-testing/ReMASC"
    # PlaybackSpoof regrouped by egs/dataset-processing/regroup-playbackspoof.py (live/, spoof/...)
    playbackspoof: str = f"{DIR_SPOOF}/playbackspoof/audio/test"


@dataclass(frozen=True)
class PathsConfig:
    live: LivePaths = field(default_factory=LivePaths)
    replay: ReplayPaths = field(default_factory=ReplayPaths)
    deep_fake: DeepFakePaths = field(default_factory=DeepFakePaths)
    augmentation: AugmentationPaths = field(default_factory=AugmentationPaths)
    validation: ValidationPaths = field(default_factory=ValidationPaths)

    def items(self) -> tp.List[tp.Tuple[str, str]]:
        """Flat ``("section.name", path)`` list."""
        return [
            (f"{section.name}.{leaf.name}", getattr(getattr(self, section.name), leaf.name))
            for section in fields(self)
            for leaf in fields(getattr(self, section.name))
        ]

    def missing(self) -> tp.List[tp.Tuple[str, str]]:
        return [(key, path) for key, path in self.items() if not Path(path).is_dir()]


PATHS = PathsConfig()


def _c_compiler_works() -> bool:
    """True if a C compiler can build what Triton builds: a Python extension using libc + Python headers.

    ``which gcc`` is not enough: a bare ``gcc`` without ``libc6-dev`` fails with ``stdlib.h: No such file``.
    """
    import subprocess
    import sysconfig
    import tempfile

    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if not cc:
        return False
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.c"
        src.write_text("#include <stdlib.h>\n#include <Python.h>\nint main(void) { return 0; }\n")
        cmd = [cc, "-c", str(src), "-o", str(Path(tmp) / "probe.o"), f"-I{sysconfig.get_paths()['include']}"]
        try:
            return subprocess.run(cmd, capture_output=True, timeout=30).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False


def disable_native_triton_ops_without_compiler() -> None:
    """torch>=2.13 overrides some eager ops (``bmm``) with Triton kernels that are JIT-compiled.

    Without a working C compiler (e.g. the slim dev container) the first such call crashes in backward
    with ``Failed to find C compiler`` / ``stdlib.h: No such file``; in that case fall back to the
    regular ATen kernels.
    """
    if _c_compiler_works():
        return
    try:
        from torch._native import registry

        registry.deregister_op_overrides(disable_dsl_names="triton")
        print("[env] no working C compiler (gcc + libc6-dev + Python.h): torch native Triton op overrides disabled")
    except Exception as e:  # private torch API - never block training because of it
        print(f"[env] could not disable torch native Triton overrides: {e!r}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_ssl(ckpt_path: str, backend: str) -> nn.Module:
    """Load mHuBERT-147 from a HF snapshot dir / model.safetensors with the chosen backend."""
    p = Path(ckpt_path)
    if not (p / "model.safetensors" if p.is_dir() else p).is_file():
        raise FileNotFoundError(f"mHuBERT weights not found: {ckpt_path} (set --mhubert-ckpt)")
    model = MHUBERT_BACKENDS[backend](ckpt_path)
    if model.encoder_dim != MHUBERT_DIM:
        raise ValueError(f"{ckpt_path}: encoder_dim={model.encoder_dim}, expected {MHUBERT_DIM}")
    return model


class MHubertBackbone(nn.Module):
    """mHuBERT-147 encoder -> utterance embedding ``(B, embedding_dim)``.

    Pipeline: waveform -> (frozen + LoRA) mHuBERT encoder -> learnable softmax-weighted
    sum of all layer outputs -> mean+std pooling over time -> Linear.

    Input is a float waveform batch ``(B, L)`` at 16 kHz in ``[-1, 1]`` (peak-normalised
    by the audio reader); per-utterance mean/var normalisation is done inside the encoder.
    """

    def __init__(
        self,
        ckpt_path: str,
        backend: str,
        embedding_dim: int,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_targets: tp.Sequence[str],
        lora_type: str = "lora",
    ) -> None:
        super().__init__()
        self.ssl = load_ssl(ckpt_path, backend)
        self.num_layers: int = self.ssl.num_layers
        dim: int = self.ssl.encoder_dim

        # 1) freeze the whole pretrained encoder, 2) inject LoRA (new params are trainable).
        self.ssl.requires_grad_(False)
        self.n_lora_layers = add_lora_to_linears(
            self.ssl, lora_targets, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
            lora_type=lora_type,
        )

        self.layer_logits = nn.Parameter(torch.zeros(self.num_layers))
        self.proj = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, embedding_dim))

    def forward(self, wavs: torch.Tensor) -> torch.Tensor:
        out = self.ssl(wavs=wavs)
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
    lora_type: str = "lora",
) -> int:
    """Attach minLoRA parametrizations to ``nn.Linear`` layers whose name contains a target substring.

    Unlike ``minlora.add_lora_by_name`` each matching ``nn.Linear`` is patched exactly once
    (a wrapper may wrap ``nn.Linear`` into a custom ``Linear`` module, so name-based ``model.apply``
    would register LoRA twice on the inner layer).
    """
    lora_config = {
        nn.Linear: {
            "weight": partial(
                SingLoRAParametrization.from_linear if lora_type == "singlora" else LoRAParametrization.from_linear,
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


def get_model(args: argparse.Namespace) -> tp.Tuple[MHubertBackbone, nn.Module]:
    backbone = MHubertBackbone(
        ckpt_path=args.mhubert_ckpt,
        backend=args.backend,
        embedding_dim=args.embedding_dim,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_type=args.lora_type,
        lora_targets=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
    )
    if args.loss == "softmax":
        am_loss = SoftmaxLoss(embedding_dim=args.embedding_dim, num_classes=NUM_CLASSES)
    else:
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
        f"[model] mHuBERT-147/{args.backend} ({args.mhubert_ckpt}): LoRA on {backbone.n_lora_layers} Linear layers "
        f"({args.lora_type}, r={args.lora_rank}, alpha={args.lora_alpha}); trainable backbone {n_train / 1e6:.2f}M "
        f"/ {n_total / 1e6:.2f}M (+ {args.loss} head {n_head / 1e3:.1f}K)"
    )
    return backbone, am_loss


class SingLoRARampCallback(Callback):
    """SingLoRA ramp-up u(t) = min(step / ramp_steps, 1); full strength (1.0) outside of fit()."""

    def __init__(self, ramp_steps: int) -> None:
        self.ramp_steps = ramp_steps

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx) -> None:
        set_singlora_ramp(pl_module, min(trainer.global_step / self.ramp_steps, 1.0) if self.ramp_steps > 0 else 1.0)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        set_singlora_ramp(pl_module, 1.0)


class SaveAdapterCallback(Callback):
    """Saves only trainable weights (LoRA + head) - a few MB instead of a full checkpoint."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: tp.Dict[str, tp.Any]
    ) -> None:
        """Drop frozen mHuBERT weights from every Lightning checkpoint (ModelCheckpoint, last.ckpt, ...)."""
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


def backup_code(log_dir: Path) -> None:
    """Copy every git-tracked file (the working-tree version, incl. uncommitted edits) to ``log_dir/code/``.

    Also writes ``code/git-info.txt`` (commit hash + ``git status``). Tracked files that are deleted in the
    working tree are skipped. Never fails the training: without git it falls back to copying this script only.
    """
    script = Path(__file__).resolve()
    dst_root = log_dir / "code"
    dst_root.mkdir(parents=True, exist_ok=True)

    def git(root: Path, *args: str) -> str:
        # safe.directory=* : the repo is often owned by another uid inside the container
        cmd = ["git", "-c", "safe.directory=*", "-C", str(root), *args]
        return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60).stdout

    try:
        root = Path(git(script.parent, "rev-parse", "--show-toplevel").strip())
        names = [n for n in git(root, "ls-files", "-z").split("\0") if n]
        copied = skipped = 0
        for name in names:
            src = root / name
            if not src.is_file():
                skipped += 1
                continue
            dst = dst_root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        info = f"commit: {git(root, 'rev-parse', 'HEAD').strip()}\n\n{git(root, 'status', '--short', '--untracked-files=no')}"
        (dst_root / "git-info.txt").write_text(info)
        print(f"Code backed up: {copied} git-tracked files ({skipped} deleted in the working tree skipped) -> {dst_root}")
    except (OSError, subprocess.SubprocessError) as e:
        dst = dst_root / script.name
        shutil.copy2(script, dst)
        print(f"[backup_code] git unavailable ({e!r}); copied only the script: {script} -> {dst}")


# ---------------------------------------------------------------------------
# Validation: 20% of ASVspoof 2017
# ---------------------------------------------------------------------------


def live_log_odds_score(model_output: tp.Any) -> tp.List[float]:
    """Per-file score for EER: log-odds of "live" against all spoof classes.

    The head returns the cosines to the class prototypes ``(B, C)``; as in AM-Softmax training
    ``p = softmax(AM_SCALE * cos)``, and the score is ``log(p_live / (1 - p_live))``
    = ``K*cos_live - logsumexp(K*cos_spoof)`` (class 0 = live, the others = every spoof class).
    Always defined (unlike a log of a difference of cosines), higher = more likely bonafide.
    """
    if isinstance(model_output, (tuple, list)):
        model_output = model_output[0]
    logits = AM_SCALE * model_output.detach().float()
    return (logits[:, 0] - torch.logsumexp(logits[:, 1:], dim=1)).cpu().tolist()


def get_validation(
    reader,
    asv17_dir: str,
    fraction: float,
    batch_size: int,
    num_workers: int,
    seed: int,
    remasc_dir: tp.Optional[str] = None,
    playbackspoof_dir: tp.Optional[str] = None,
    playbackspoof_fraction: float = 1.0,
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
            score_fn=live_log_odds_score,
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

    joined = AggregatedAntiSpoofingValidator(name="spoof-joined", validators=validators, return_scores=False)
    if not playbackspoof_dir:
        return [joined]

    ps = get_playbackspoof(playbackspoof_dir, reader, fraction=playbackspoof_fraction, seed=seed)
    print(
        f"[playbackspoof] fraction={playbackspoof_fraction} | live: {len(ps.dataset_live)} | "
        f"deep-fake: {len(ps.dataset_deep_fake)} | replay: {len(ps.dataset_replay)} | "
        f"replay-deep-fake: {len(ps.dataset_replay_deep_fake)}"
    )
    # EER: live vs df | live vs replay | live vs replay-df | live vs pooled (per_validator/<group>/eer)
    playbackspoof = GroupedAntiSpoofingValidator(
        name="playbackspoof",
        bonafide_loader=data_loader(ps.dataset_live),
        spoof_loaders={
            "deep-fake": data_loader(ps.dataset_deep_fake),
            "replay": data_loader(ps.dataset_replay),
            "replay-deep-fake": data_loader(ps.dataset_replay_deep_fake),
        },
        score_fn=live_log_odds_score,
    )
    return [joined, playbackspoof]


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def build_augmentation_pipeline() -> SequentialCompose:
    rir_provider_point = FileListAudioProvider(paths=find_files_recursive(Path(PATHS.augmentation.rir) / "pointsource_noises", extension=".wav"))
    rir_provider_real = FileListAudioProvider(paths=find_files_recursive(Path(PATHS.augmentation.rir) / "real_rirs_isotropic_noises", extension=".wav"))
    rir_provider_sim = FileListAudioProvider(paths=find_files_recursive(Path(PATHS.augmentation.rir) / "simulated_rirs", extension=".wav"))

    noise_provider_music = FileListAudioProvider(paths=find_files_recursive(Path(PATHS.augmentation.noise) / "music", extension=".wav"))
    noise_provider_noise = FileListAudioProvider(paths=find_files_recursive(Path(PATHS.augmentation.noise) / "noise", extension=".wav"))
    noise_provider_speech = FileListAudioProvider(paths=find_files_recursive(Path(PATHS.augmentation.noise) / "speech", extension=".wav"))

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
    # vc_tts_engines was resampled to .ogg (resample-dir.py --format ogg)
    create_spoof_source_ogg = partial(
        sources_from_subfolders,
        factory=partial(VoxDataset, reader=reader, file_extension="ogg"),
        ignore_empty=False,
        prefix="legacy-",
    )
    create_live_source = partial(
        sources_from_subfolders,
        factory=dataset_factory,
        ignore_empty=False,
    )

    def single(root: str, label: int, weight: float, name: str, file_extension: str = "wav") -> LabeledSource:
        return LabeledSource(
            dataset=[WeightedDataset(VoxDataset(reader=reader, root=root, file_extension=file_extension))],
            label=label,
            weight=weight,
            name=name,
        )

    live, replay, df = PATHS.live, PATHS.replay, PATHS.deep_fake

    sources: tp.List[LabeledSource] = [
        # bonafide
        single(live.vox1, label_live, 6.0, "vox1"),
        single(live.vox2, label_live, 6.0, "vox2"),
        # single(live.spgi, label_live, 2.0, "spgispeech"),
        single(live.tidy_1, label_live, 1.0, "tidy_voice_1"),
        single(live.tidy_2, label_live, 1.0, "tidy_voice_2"),
        single(live.vctk, label_live, 1.0, "VCTK"),
        single(live.mcv13, label_live, 1.0, "MCV-13"),
        single(live.voxtube, label_live, 2.0, "voxtube"),
        single(live.libri, label_live, 3.0, "libri_speech"),
        single(live.asv21, label_live, 1.0, "asv-21"),
        single(live.cn_celeb, label_live, 1.0, "cn-celeb-v2"),
        single(live.hifi_tts, label_live, 1.0, "hifi-tts"),
        single(live.playbackspoof, label_live, 1.0, "playbackspoof-live", file_extension="flac"),
    ]
    sources += create_live_source(root=live.cml, label=label_live, source_weight=0.8, prefix="cml-")

    # ReMASC: every environment (outdoor / indoor-1 / indoor-2 / vehicle) is a separate source for live and for replay
    create_env_source = partial(sources_from_subfolders, factory=dataset_factory, ignore_empty=True)
    remasc_live = create_env_source(root=live.remasc, label=label_live, source_weight=0.5, prefix="remasc-live-")
    remasc_replay = create_env_source(root=replay.remasc, label=label_la, source_weight=1.0, prefix="remasc-replay-")
    print(f"[remasc train] {len(remasc_live)} live + {len(remasc_replay)} replay sources: "
          f"{[src.name for src in remasc_live + remasc_replay]}")
    sources += remasc_live

    # spoof - LA (replay)
    sources += [
        single(replay.formanta, label_la, 4.0, "Replay-Formanta"),
        single(replay.playbackspoof, label_la, 2.0, "playbackspoof-replay", file_extension="flac"),
        single(replay.playbackspoof_deep_fake, label_la, 1.0, "playbackspoof-replay-deep-fake", file_extension="flac"),
    ]
    sources += remasc_replay

    # spoof - DF (synthetic)
    sources += create_spoof_source(
        root=df.tts_common, label=label_df, source_weight=2.0, ignore_pattern="*tts_test_set*"
    )
    sources += create_spoof_source_ogg(root=df.vc_tts, label=label_df, source_weight=0.8)
    sources += create_spoof_source(root=df.vocoders_v1, label=label_df, source_weight=1.2)
    sources += create_spoof_source(root=df.vc_vol1, label=label_df, source_weight=0.5)
    sources += create_spoof_source(
        root=df.vc_vol2, label=label_df, ignore_pattern="*resamble_denoiser*", source_weight=1.0
    )
    sources.append(single(df.playbackspoof, label_df, 1.0, "playbackspoof-deep-fake", file_extension="flac"))

    return LabeledAggregatedDataset(sources=sources, name="train_dataset")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mhubert-ckpt", default=MHUBERT_CKPT, help="mHuBERT-147: dir with model.safetensors (+ config.json) or the .safetensors file")
    p.add_argument("--backend", choices=sorted(MHUBERT_BACKENDS), default="own", help="own: voicesdk.nn.arch.mhubert; torchaudio: torchaudio.models.hubert_base")
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--loss", choices=["amsoftmax", "softmax"], default="amsoftmax", help="classification head: AM-Softmax (cosine, scale, margin) or plain linear softmax")

    g = p.add_argument_group("LoRA (minLoRA)")
    g.add_argument("--lora-type", choices=["lora", "singlora"], default="lora", help="singlora: single matrix A (dW = A A^T), see thirdparty/minLoRA")
    g.add_argument("--singlora-ramp-steps", type=int, default=1000, help="SingLoRA ramp-up length in optimizer steps")
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
    g.add_argument("--asv17-dir", default=PATHS.validation.asv17, help="ASVspoof 2017 root (meta.csv + eval/)")
    g.add_argument(
        "--remasc-dir",
        default=PATHS.validation.remasc,
        help="ReMASC root (meta.csv + eval/); default from PATHS, pass an empty string to skip the validator",
    )
    g.add_argument(
        "--playbackspoof-dir",
        default=PATHS.validation.playbackspoof,
        help="regrouped PlaybackSpoof root (live/, spoof/...); default from PATHS, pass an empty string to skip the validator",
    )
    g.add_argument(
        "--playbackspoof-fraction",
        type=float,
        default=PLAYBACKSPOOF_FRACTION,
        help="share of every PlaybackSpoof group used for validation, 1.0 = all",
    )
    g.add_argument("--val-fraction", type=float, default=0.2)
    g.add_argument("--val-batch-size", type=int, default=64)
    g.add_argument("--val-num-workers", type=int, default=2)
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("logging")
    g.add_argument("--log-dir", default="data/exps/")
    g.add_argument("--exp-name", default=None, help="default: antispoof-mhubert147-<lora-type>")
    g.add_argument("--log-weights-every", type=int, default=100)
    g.add_argument("--batch-delay-ms", type=float, default=0.0, help="pause after every training batch, ms (lets the GPU cool down); 0 = off")
    return parse_with_config(p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    disable_native_triton_ops_without_compiler()
    exp_name = args.exp_name or f"antispoof-mhubert147-{args.lora_type}"
    args.exp_name = exp_name
    total_steps = args.steps_per_epoch * args.max_epochs
    pl.seed_everything(args.seed)

    # ---- data ------------------------------------------------------------
    # Peak normalisation keeps the waveform in [-1, 1], which is the scale the waveform reader gives to the encoder.
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
        save_top_k=20,
        monitor="train/loss",
        mode="min",
        save_last=True,
    )
    adapter_cb = SaveAdapterCallback(exp_dir / "adapter-last.pt")

    # ---- validation ------------------------------------------------------
    validators = get_validation(
        reader=reader_val,
        asv17_dir=args.asv17_dir,
        fraction=args.val_fraction,
        batch_size=args.val_batch_size,
        num_workers=args.val_num_workers,
        seed=args.seed,
        remasc_dir=args.remasc_dir,
        playbackspoof_dir=args.playbackspoof_dir,
        playbackspoof_fraction=args.playbackspoof_fraction,
    )

    # ---- lightning module ------------------------------------------------
    module = AntispoofLightningModule(
        backbone=backbone,
        am_loss=am_loss,
        dataset=dataset_train,
        strategy=strategy,
        validators=validators,
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
    # on resume; the frozen mHuBERT weights come from the pretrained mHuBERT checkpoint.
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
        callbacks=[checkpoint_cb, adapter_cb]
        + ([make_batch_delay_callback(args.batch_delay_ms)] if args.batch_delay_ms > 0 else [])
        + ([SingLoRARampCallback(args.singlora_ramp_steps)] if args.lora_type == "singlora" else []),
        enable_progress_bar=True,
        log_every_n_steps=50,
        limit_train_batches=args.steps_per_epoch,
        limit_val_batches=0,  # validators run in AntispoofLightningModule.on_train_epoch_end
    )

    backup_code(exp_dir)
    save_config(args, exp_dir, Path(__file__).name)
    trainer.fit(module, train_dataloaders=loader_train, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
