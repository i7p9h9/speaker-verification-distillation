import typing as tp
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from audimentation import AddNoise, FileListAudioProvider, OneOf, Reverb, SequentialCompose
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from validation import AggregatedDataset, TrialBasedValidator, ValidationTrial, VoxDataset, WeightedDataset

from voicesdk.distillation.data import AudioReaderFull, AudioReaderRandom, collate_batch_segments_fn
from voicesdk.distillation.loss import LossDistillationEmbeddings
from voicesdk.distillation.nn import HeadClassificationCentroids, HeadModelWrapper
from voicesdk.distillation.training import DistillationLightningModule
from voicesdk.nn.arch import ReDimNetWrap, ResNetTF
from voicesdk.utils.find_files import find_files_recursive

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STEPS_PER_EPOCH = 5000
MAX_EPOCH = 25

TEACHER_CFG = "data/cfg-models/rn100_v016_flr_vox4_v2.yaml"
TEACHER_CKPT = "data/ckpt/rn100_v016_flr_vox4_v2/model.pt"

STUDENT_CFG = "data/cfg-models/redimnet_L.yaml"
STUDENT_CKPT = "data/exps/vox2-emb-cosine-001/student-last.ckpt"
# STUDENT_CKPT = None

HEAD_CKPT = "data/centroids/vox2/rn100_v016_flr_vox4_v2/head_centroid.pt"
HEAD_SPEAKERS = "data/centroids/vox2/rn100_v016_flr_vox4_v2/head_centroid_speakers.json"

TRAIN_VOX = "/media/ssd/voice/datasets/vox2/dev-16k/aac/"
TRAIN_SIGI = "/media/ssd/voice/datasets/spgispeech/"
TRAIN_TIDY_1 = "/media/ssd/voice/datasets/TidyVoiceX/"
TRAIN_TIDY_2 = "/media/ssd/voice/datasets/TidyVoiceX2/"

VAL_ROOT = "/media/ssd/voice/datasets/vox1/test/wav"
TRIALS_PATH = "data/test_vox/trials"

DIR_RIR = Path("/media/ssd/voice/datasets/RIRs/RIRS_NOISES/")
DIR_NOISE = Path("/media/ssd/voice/datasets/musan/")

LOG_DIR = "data/exps/"
EXPERIMENT_NAME = "mic-emb-cosine-009"
SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return dict(yaml.load(f, Loader=yaml.FullLoader))


def get_teacher() -> nn.Module:
    cfg = read_yaml(TEACHER_CFG)
    state_dict = torch.load(TEACHER_CKPT, map_location="cpu")
    model = ResNetTF(**cfg["model_args"])
    model.eval()
    model.load_state_dict(state_dict, strict=True)
    return model


def get_student() -> nn.Module:
    cfg = read_yaml(STUDENT_CFG)
    model = ReDimNetWrap(**cfg["model_args"])
    if STUDENT_CKPT is not None:
        model.load_state_dict(torch.load(STUDENT_CKPT))
    return model


def get_head() -> HeadClassificationCentroids:
    state_dict = torch.load(HEAD_CKPT, map_location="cpu")
    centroids = state_dict["centroids.weight"]
    num_classes, embedding_dim = centroids.shape
    head = HeadClassificationCentroids(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        normalize_input=False,
    )
    head.load_state_dict(state_dict)
    head.load_speaker_mapping(HEAD_SPEAKERS)
    return head


def get_trials_vox1(trial_path: str) -> tp.List[ValidationTrial]:
    trials = []
    with open(trial_path, "r") as f:
        for line in f:
            target, left, right = line.strip().split(" ")
            trials.append(ValidationTrial(
                trial_left=left,
                trial_right=right,
                is_target=target == "1",
            ))
    return trials


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------
# Self-contained: move this function + DIR_* constants to a separate module
# when needed.
# ---------------------------------------------------------------------------

def build_augmentation_pipeline() -> SequentialCompose:
    rir_provider_point = FileListAudioProvider(paths=find_files_recursive(DIR_RIR / "pointsource_noises",         extension=".wav"))
    rir_provider_real  = FileListAudioProvider(paths=find_files_recursive(DIR_RIR / "real_rirs_isotropic_noises", extension=".wav"))
    rir_provider_sim   = FileListAudioProvider(paths=find_files_recursive(DIR_RIR / "simulated_rirs",             extension=".wav"))

    noise_provider_music  = FileListAudioProvider(paths=find_files_recursive(DIR_NOISE / "music",  extension=".wav"))
    noise_provider_noise  = FileListAudioProvider(paths=find_files_recursive(DIR_NOISE / "noise",  extension=".wav"))
    noise_provider_speech = FileListAudioProvider(paths=find_files_recursive(DIR_NOISE / "speech", extension=".wav"))

    return SequentialCompose(
        stages=[
            OneOf(
                stages=[
                    Reverb(rir_provider=rir_provider_point, name="reverb_point", wet_dry_range=(0.2, 0.5)),
                    Reverb(rir_provider=rir_provider_real,  name="reverb_real",  wet_dry_range=(0.2, 0.5)),
                    Reverb(rir_provider=rir_provider_sim,   name="reverb_sim",   wet_dry_range=(0.2, 0.5)),
                ],
                weights=[10, 5, 5],
                name="reverb",
                p=0.3,
            ),
            OneOf(
                stages=[
                    AddNoise(noise_provider=noise_provider_music,  name="noise_music",  snr_range=(-1.0, 10.0)),
                    AddNoise(noise_provider=noise_provider_noise,  name="noise_noise",  snr_range=(-1.0, 10.0)),
                    AddNoise(noise_provider=noise_provider_speech, name="noise_speech", snr_range=(5.0,  12.0)),
                ],
                weights=[10, 1, 1],
                name="noise",
                p=1.0,
            ),
        ],
        p=0.75,
    )


# ---------------------------------------------------------------------------
# Validation callback
# ---------------------------------------------------------------------------

class ValidationCallback(Callback):
    """Runs validation at fixed step intervals."""

    def __init__(
        self,
        validate_every_n_epochs: int = 1,
        validate_every_n_steps: tp.Optional[int] = None,
    ) -> None:
        super().__init__()
        self.validate_every_n_epochs = validate_every_n_epochs
        self.validate_every_n_steps = validate_every_n_steps
        self._last_val_step = 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: DistillationLightningModule,
        outputs: tp.Any,
        batch: tp.Any,
        batch_idx: int,
    ) -> None:
        if self.validate_every_n_steps is not None:
            steps_since_last = trainer.global_step - self._last_val_step
            if steps_since_last >= self.validate_every_n_steps:
                self._last_val_step = trainer.global_step
                trainer.validate(pl_module)


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
    validate_every_n_epochs: int = 1,
    validate_every_n_steps: tp.Optional[int] = None,
    limit_train_batches: tp.Optional[tp.Union[int, float]] = None,
) -> pl.Trainer:
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)

    all_callbacks: tp.List[Callback] = callbacks or []
    all_callbacks.append(ValidationCallback(
        validate_every_n_epochs=validate_every_n_epochs,
        validate_every_n_steps=validate_every_n_steps,
    ))

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
    # --- Models ---
    head = get_head()
    model_teacher = HeadModelWrapper(
        model_base=get_teacher(),
        head=head,
        enable_grad=False,
        enable_train=False,
    )
    model_student = HeadModelWrapper(
        model_base=get_student(),
        head=head,
        enable_grad=True,
        enable_train=True,
    )

    # --- Data ---
    reader_val = AudioReaderFull(
        norm_type="std",
        length_segment_ms=6000,
        segments_step_ms=4000,
        sample_rate=SAMPLE_RATE,
    )
    reader_train = AudioReaderRandom(norm_type="std", length_segment_ms=3000)

    dataset_val = VoxDataset(reader=reader_val, root=VAL_ROOT)
    dataset_train = AggregatedDataset(sources=[
        WeightedDataset(dataset=VoxDataset(reader=reader_train, root=TRAIN_VOX),    weight=0.8),
        WeightedDataset(dataset=VoxDataset(reader=reader_train, root=TRAIN_SIGI),   weight=0.3),
        WeightedDataset(dataset=VoxDataset(reader=reader_train, root=TRAIN_TIDY_1), weight=0.2),
        WeightedDataset(dataset=VoxDataset(reader=reader_train, root=TRAIN_TIDY_2), weight=0.1),
    ])

    loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=64,
        collate_fn=collate_batch_segments_fn,
        num_workers=8,
        persistent_workers=True,
        shuffle=True,
    )

    validator = TrialBasedValidator(
        name="vox1-base",
        trials=get_trials_vox1(TRIALS_PATH),
        trial_to_idx=dataset_val.trial_to_index,
        dataset=dataset_val,
        use_weighted_aggregation=True,
    )

    # --- Loss & module ---
    loss_fn = LossDistillationEmbeddings(loss_type="cosine", normalize=True)

    module_distill = DistillationLightningModule(
        teacher_model=model_teacher,
        student_model=model_student,
        loss_fn=loss_fn,
        validators=[validator],
        learning_rate=0.003,
        final_lr=0.0001,
        weight_decay=1e-3,
        warmup_steps=int(0.5 * STEPS_PER_EPOCH),
        warm_from_zero=True,
        scale_ratio=1.0,
        total_steps=STEPS_PER_EPOCH * MAX_EPOCH,
        # --- augmentation ---
        aug_pipeline=build_augmentation_pipeline(),
        aug_teacher_original=False,   # False -> teacher also receives augmented audio
        aug_sample_rate=SAMPLE_RATE,
    )

    # --- Trainer ---
    trainer = create_trainer(
        max_epochs=MAX_EPOCH,
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_dir=LOG_DIR,
        experiment_name=EXPERIMENT_NAME,
        save_top_k=3,
        monitor_metric="val/vox1-base/eer",
        monitor_mode="min",
        validate_every_n_epochs=1,
        limit_train_batches=STEPS_PER_EPOCH,
    )

    trainer.fit(module_distill, train_dataloaders=loader_train, val_dataloaders=loader_train)


if __name__ == "__main__":
    main()