import shutil
import typing as tp
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from audimentation import AddNoise, FileListAudioProvider, OneOf, Reverb, SequentialCompose
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from validation import AggregatedAntiSpoofingValidator, AntiSpoofingValidator, VoxDataset
from validation.utils.get_spoof_datasets import get_asv_spoof_17, get_asv_spoof_19, get_splitted

from voicesdk.dataset import (
    LabeledAggregatedDataset,
    LabeledSource,
    StrategyLossWeighting,
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
from voicesdk.nn.arch import ReDimNetWrap, ResNetTF
from voicesdk.nn.loss import AMSoftmaxLoss
from voicesdk.utils.find_files import find_files_recursive

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STEPS_PER_EPOCH: int = 5000
MAX_EPOCHS: int = 30

# BACKBONE_CFG: str = "data/cfg-models/redimnet_M.yaml"
# BACKBONE_CKPT: str = 'data/exps/tel-subnet-emb-cosine-016/checkpoints/last.ckpt'
BACKBONE_CFG = "data/cfg-models/resnettf_34.yaml"
BACKBONE_CKPT = "data/exps/mic-emb-cosine-014/student-last.ckpt"
RESUME_CKPT: tp.Optional[str] = None  # path to Lightning checkpoint to resume

# AM-Softmax head
NUM_CLASSES: int = 3 # 0 = bonafide, 1 = spoof
EMBEDDING_DIM: int = 256 # must match backbone output dim
AM_SCALE: float = 20.0
AM_MARGIN: float = 0.15

# Optimizer / scheduler
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-3
WARMUP_STEPS: int = int(0.5 * STEPS_PER_EPOCH)
MIN_LR_RATIO: float = 0.0

# Data
BATCH_SIZE: int = 64
NUM_WORKERS: int = 8

# Logging
LOG_DIR: str = "data/exps/"
EXPERIMENT_NAME: str = "antispoof-join-train-015"
LOG_WEIGHTS_EVERY: int = 100  # steps between dataset-weight log entries

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
PATH_TRAIN_CODECS: str = "/media/ssd/voice/datasets/antispoof/codecs-16k/audio_codecs_results"
PATH_TRAIN_VC_TTS: str = "/media/ssd/voice/datasets/antispoof/vc_tts_engines-16k/data"
PATH_SYN_TTS_COMMON: str = "/media/ssd/voice/datasets/antispoof/synthesis-train/tts"
PATH_SYN_ASV21_EVAL_DF: str = "/media/ssd/voice/datasets/antispoof/synthesis-train/ASV21Eval/ASVspoof2021_DF_eval"
PATH_SYN_ASV21_EVAL_LA: str = "/media/ssd/voice/datasets/antispoof/synthesis-train/ASV21Eval/ASVspoof2021_LA_eval"
PATH_SYN_ASV19_EVAL_LA: str = "/media/ssd/voice/datasets/antispoof/synthesis-train/ASVSpoof2019_LA_eval"
PATH_SYN_VOCODERS_V1: str = "/media/ssd/voice/datasets/antispoof/synthesis-train/vocoders-v1/train/synthes"
PATH_SYN_VC_VOL1: str = "/media/ssd/voice/datasets/antispoof/synthesis-train/voice_clones/train/synthes"
PATH_SYN_VC_VOL2: str = "/media/ssd/voice/datasets/antispoof/synthesis-train/voice-clone-vol2/train"

# Dataset PAD
PATH_PAD = "/media/ssd/voice/datasets/antispoof/replay/"

# augmentation
DIR_RIR = Path("/media/ssd/voice/datasets/RIRs/RIRS_NOISES/")
DIR_NOISE = Path("/media/ssd/voice/datasets/musan/")

# Validation
DIR_ASV_17 = Path("/media/ssd/voice/datasets/antispoof/synthesis-test/ASV17_eval/")
DIR_ASV_19 = Path("/media/ssd/voice/datasets/antispoof/synthesis-test/ASVspoof2019_LA_cellular/")
DIR_DEEP_VOICE = Path("/media/ssd/voice/datasets/antispoof/synthesis-test/DEEP-VOICE/AUDIO_16k_cut/")
DIR_FAKE_OR_REAL = Path("/media/ssd/voice/datasets/antispoof/synthesis-test/FakeOrReal/for-norm/validation/")


def backup_script(log_dir: Path) -> None:
    """Copy the current script to log_dir/scripts/."""
    src = Path(__file__).resolve()
    dst_dir = log_dir / "scripts"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    print(f"Script backed up: {src} -> {dst}")


def get_validation_datasets(reader, batch_size=100, num_workers: int = 2):
    _data_loader = partial(
        DataLoader,
        batch_size=batch_size,
        collate_fn=collate_batch_segments_fn,
        num_workers=num_workers,
        persistent_workers=False,
        shuffle=False,
        drop_last=False,
    )

    def print_dataset_stats(name: str, datasets) -> None:
        n_live = len(datasets.dataset_live)
        n_spoof = len(datasets.dataset_spoof)
        total = n_live + n_spoof
        print(
            f"[{name}] live: {n_live:>6} | spoof: {n_spoof:>6} | total: {total:>6}"
        )

    datasets_asv_17 = get_asv_spoof_17(
        dir_main=DIR_ASV_17,
        reader=reader,
        limit_live=1600,
        limit_spoof=1600,
    )
    print_dataset_stats("ASVspoof17", datasets_asv_17)

    datasets_asv_19 = get_asv_spoof_19(
        dir_wav=DIR_ASV_19 / "dev" / "ASVspoof2019_LA_dev" / "wav",
        file_meta=DIR_ASV_19 / "ASVspoof2019_LA_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt",
        reader=reader,
        limit_live=1600,
        limit_spoof=1600,
    )
    print_dataset_stats("ASVspoof19", datasets_asv_19)

    datasets_fake_or_real = get_splitted(
        dir_live=DIR_FAKE_OR_REAL / "real",
        dir_spoof=DIR_FAKE_OR_REAL / "fake",
        reader=reader,
        limit_live=1600,
        limit_spoof=1600,
    )
    print_dataset_stats("FakeOrReal", datasets_fake_or_real)

    # NOTE: original code reassigned datasets_fake_or_real — using separate variable here
    datasets_deep_voice = get_splitted(
        dir_live=DIR_DEEP_VOICE / "REAL",
        dir_spoof=DIR_DEEP_VOICE / "FAKE",
        reader=reader,
        limit_live=1600,
        limit_spoof=1600,
    )
    print_dataset_stats("DeepVoice", datasets_deep_voice)

    validators: tp.List[AntiSpoofingValidator] = []

    validators.append(
        AntiSpoofingValidator(
            name="asv_spoof_17",
            bonafide_loader=_data_loader(datasets_asv_17.dataset_live),
            spoof_loader=_data_loader(datasets_asv_17.dataset_spoof),
        )
    )
    # validators.append(
    #     AntiSpoofingValidator(
    #         name="asv_spoof_19",
    #         bonafide_loader=_data_loader(datasets_asv_19.dataset_live),
    #         spoof_loader=_data_loader(datasets_asv_19.dataset_spoof),
    #     )
    # )
    validators.append(
        AntiSpoofingValidator(
            name="fake_or_real",
            bonafide_loader=_data_loader(datasets_fake_or_real.dataset_live),
            spoof_loader=_data_loader(datasets_fake_or_real.dataset_spoof),
        )
    )
    validators.append(
        AntiSpoofingValidator(
            name="deep_voice",
            bonafide_loader=_data_loader(datasets_deep_voice.dataset_live),
            spoof_loader=_data_loader(datasets_deep_voice.dataset_spoof),
        )
    )

    total_live = sum(len(v.bonafide_loader.dataset) for v in validators)
    total_spoof = sum(len(v.spoof_loader.dataset) for v in validators)
    print(f"{'─' * 55}")
    print(
        f"[TOTAL]     live: {total_live:>6} | spoof: {total_spoof:>6} | total: {total_live + total_spoof:>6}"
    )

    return AggregatedAntiSpoofingValidator(
        name="spoof-joined",
        validators=validators,
        return_scores=False,
    )


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


def read_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    return dict(hparams)


# ---- model -----------------------------------------------------------
# def get_model():
#     with open(BACKBONE_CFG, "r") as f:
#         cfg = yaml.safe_load(f)
#     backbone = ReDimNetWrap(**cfg["model_args"])

#     if BACKBONE_CKPT is not None:
#         weights_path = Path(BACKBONE_CKPT)
#         state_dict_loaded = torch.load(weights_path)
#         state_dict_student = backbone.state_dict()
#         for k in state_dict_student:
#             state_dict_student[k].copy_(state_dict_loaded["state_dict"][f"student_model.model_base.{k}"])

#     am_loss = AMSoftmaxLoss(
#         embedding_dim=EMBEDDING_DIM,
#         num_classes=NUM_CLASSES,
#         scale=AM_SCALE,
#         margin=AM_MARGIN,
#     )

#     return backbone, am_loss

def get_model():
    cfg = read_yaml(BACKBONE_CFG)
    backbone = ResNetTF(**cfg["model_args"])
    # backbone = ReDimNetWrap(**cfg["model_args"])

    weights_path = Path(BACKBONE_CKPT)
    state_dict_loaded = torch.load(weights_path)
    backbone.load_state_dict(state_dict_loaded)

    am_loss = AMSoftmaxLoss(
        embedding_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        scale=AM_SCALE,
        margin=AM_MARGIN,
    )

    return backbone, am_loss



def build_train_dataset(reader: tp.Any) -> LabeledAggregatedDataset:
    """Assemble labeled training sources.

    Bonafide → label=0, spoof → label=1.
    """
    dataset_factory = partial(VoxDataset, reader=reader)

    label_live = 0
    label_la = 1
    if NUM_CLASSES == 2:
        label_df = 1
    elif NUM_CLASSES == 3:
        label_df = 2
    else:
        raise Exception("NUM_CLASSES must be less 3")

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

    sources: tp.List[LabeledSource] = []

    # bonafide
    sources.append(LabeledSource(
        dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_TRAIN_VOX1))],
        label=label_live, weight=6.0, name="vox1",
    ))
    sources.append(LabeledSource(
        dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_TRAIN_VOX2))],
        label=label_live, weight=6.0, name="vox2",
    ))
    sources.append(LabeledSource(
        dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_TRAIN_SPGI))],
        label=label_live, weight=2.0, name="spgispeech",
    ))
    sources.append(LabeledSource(
        dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_TRAIN_TIDY_1))],
        label=label_live, weight=1.0, name="tidy_voice_1",
    ))
    sources.append(LabeledSource(
        dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_TRAIN_TIDY_2))],
        label=0, weight=1.0, name="tidy_voice_2",
    ))
    sources.append(LabeledSource(
        dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_TRAIN_VCTK))],
        label=label_live, weight=1.0, name="VCTK",
    ))
    sources.append(LabeledSource(
        dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_TRAIN_MCV13))],
        label=label_live, weight=1.0, name="MCV-13",
    ))
    sources.append(LabeledSource(
        dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_TRAIN_VOXTUBE))],
        label=label_live, weight=2.0, name="voxtube",
    ))
    sources.append(LabeledSource(
        dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_TRAIN_LIBRI))],
        label=label_live, weight=3.0, name="libri_speech",
    ))
    sources.append(LabeledSource(
        dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_TRAIN_ASV21))],
        label=label_live, weight=1.0, name="asv-21",
    ))
    sources.append(LabeledSource(
        dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_TRAIN_CN_CELEB))],
        label=label_live, weight=1.0, name="cn-celeb-v2",
    ))
    sources += create_live_source(
        root=PATH_TRAIN_LIVE_CML,
        label=label_live,
        source_weight=0.8,
        prefix="cml-"
    )

    # spoof — LA
    # sources.append(LabeledSource(
    #     dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_SYN_ASV21_EVAL_LA))],
    #     label=label_la, weight=0.2, name="ASV21Eval_LA",
    # ))
    # sources.append(LabeledSource(
    #     dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_SYN_ASV19_EVAL_LA))],
    #     label=label_la, weight=0.2, name="ASV19Eval_LA",
    # ))
    sources.append(LabeledSource(
        dataset=[WeightedDataset(VoxDataset(reader=reader, root=f"{PATH_PAD}/voxceleb_toloka_replays_2026_02_15-16k"))],
        label=label_la, weight=4.0, name="Replay-02-15",
    ))
    sources.append(LabeledSource(
        dataset=[WeightedDataset(VoxDataset(reader=reader, root=f"{PATH_PAD}/voxceleb_toloka_replays_2026_02_21-16k"))],
        label=label_la, weight=4.0, name="Replay-02-21",
    ))

    # spoof — DF
    # sources.append(LabeledSource(
    #     dataset=[WeightedDataset(VoxDataset(reader=reader, root=PATH_SYN_ASV21_EVAL_DF))],
    #     label=label_df, weight=0.2, name="ASV21Eval_DF",
    # ))

    sources += create_spoof_source(
        root=PATH_SYN_TTS_COMMON, label=label_df, source_weight=2.0,
        ignore_pattern="*tts_test_set*",
    )
    # sources += create_spoof_source(root=PATH_TRAIN_CODECS, label=label_df, source_weight=1.0)
    sources += create_spoof_source(root=PATH_TRAIN_VC_TTS, label=label_df, source_weight=0.8)
    # sources += create_spoof_source(root=PATH_SYN_VOCODERS_V1, label=label_df, ignore_pattern="*BigVGAN*", source_weight=1.2)
    sources += create_spoof_source(root=PATH_SYN_VOCODERS_V1, label=label_df, source_weight=1.2)
    sources += create_spoof_source(root=PATH_SYN_VC_VOL1, label=label_df, source_weight=0.5)
    sources += create_spoof_source(root=PATH_SYN_VC_VOL2, label=label_df, ignore_pattern="*resamble_denoiser*", source_weight=1.0)

    return LabeledAggregatedDataset(sources=sources, name="train_dataset")

# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------

def create_trainer(
    max_epochs: int = 10,
    max_steps: int = -1,
    accelerator: str = "auto",
    devices: tp.Union[int, tp.List[int], str] = "auto",
    precision: tp.Union[int, str] = 32,
    gradient_clip_val: tp.Optional[float] = 1.0,
    accumulate_grad_batches: int = 1,
    log_dir: str = "logs",
    experiment_name: str = "distillation",
    checkpoint_dir: tp.Optional[str] = None,
    save_top_k: int = 3,
    monitor_metric: str = "val/loss",
    monitor_mode: str = "min",
    callbacks: tp.Optional[tp.List[Callback]] = None,
    limit_train_batches: tp.Optional[tp.Union[int, float]] = None,
) -> pl.Trainer:
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)

    all_callbacks: tp.List[Callback] = callbacks or []
    ckpt_dir = checkpoint_dir or f"{log_dir}/{experiment_name}/checkpoints"
    all_callbacks.append(ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch}-{step}",
        save_top_k=save_top_k,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_last=True,
    ))

    return pl.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=tb_logger,
        callbacks=all_callbacks,
        enable_progress_bar=True,
        limit_val_batches=1,
        log_every_n_steps=50,
        limit_train_batches=limit_train_batches,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_steps = STEPS_PER_EPOCH * MAX_EPOCHS

    # ---- data ------------------------------------------------------------
    reader_train = AudioReaderTelSimulated(
        norm_type="std",
        length_segment_ms=5_000,
        p_tel=0.0,
    )
    reader_val = AudioReaderBegin(
        norm_type="std",
        length_segment_ms=5_000,
    )
    dataset_train = build_train_dataset(reader_train)
    # strategy = StrategyLossWeighting(dataset_train, ema_alpha=0.2)
    strategy = StrategyMomentumLossWeighting(dataset_train, momentum=0.985, scale=7.0)

    loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=BATCH_SIZE,
        collate_fn=collate_batch_labeled_segments_fn,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        shuffle=True,
    )

    backbone, am_loss = get_model()

    # ---- loggers / callbacks ---------------------------------------------
    tb_logger = TensorBoardLogger(save_dir=LOG_DIR, name=EXPERIMENT_NAME)

    exp_dir = Path(LOG_DIR) / EXPERIMENT_NAME
    weight_logger = WeightLogger(
        tb_logger=tb_logger,
        csv_path=exp_dir / "dataset_weights.csv",
        log_every_n_steps=LOG_WEIGHTS_EVERY,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(exp_dir / "checkpoints"),
        filename="{epoch}-{step}",
        save_top_k=3,
        monitor="train/loss",
        # monitor_metric="val/vox1-base/eer",
        mode="min",
        save_last=True,
    )

    # ---- validation defenition -------------------------------------------
    validator = get_validation_datasets(reader=reader_val)

    # ---- lightning module ------------------------------------------------
    module = AntispoofLightningModule(
        backbone=backbone,
        am_loss=am_loss,
        dataset=dataset_train,
        strategy=strategy,
        validators=[validator],
        weight_logger=weight_logger,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        total_steps=total_steps,
        min_lr_ratio=MIN_LR_RATIO,
        log_every_n_steps=50,
        aug_pipeline=build_augmentation_pipeline(),
    )

    # ---- trainer ---------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        logger=tb_logger,
        callbacks=[checkpoint_cb],
        enable_progress_bar=True,
        log_every_n_steps=50,
        limit_train_batches=STEPS_PER_EPOCH,
        limit_val_batches=0,  # no val loader — attach validator Callbacks if needed
    )

    backup_script(Path(LOG_DIR) / EXPERIMENT_NAME)

    trainer.fit(
        module,
        train_dataloaders=loader_train,
        ckpt_path=RESUME_CKPT,
    )


if __name__ == "__main__":
    main()
