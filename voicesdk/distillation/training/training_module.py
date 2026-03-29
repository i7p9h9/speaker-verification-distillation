import math
import typing as tp
from dataclasses import fields

import pytorch_lightning as pl
import torch
from torch import nn

from voicesdk.distillation.data import BatchSegments
from voicesdk.distillation.loss import DFLossBase, LossDistillationBase, ModelOutput
from voicesdk.distillation.validation import ValidatorBase

from .flow_context import TrainingFlowContext

if tp.TYPE_CHECKING:
    from audimentation import SequentialCompose


def _augment_batch(
    segments: torch.Tensor,
    pipeline: "SequentialCompose",
    sample_rate: int,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Apply augmentation pipeline to a batch.

    Args:
        segments:    Float tensor ``[N, L]``.
        pipeline:    Augmentation pipeline (supports batched input natively).
        sample_rate: Audio sample rate in Hz.

    Returns:
        ``(augmented, original)`` — both ``[N, L]``.
    """
    from audimentation import DataSample

    sample = DataSample(signal=segments, original=segments.clone(), sample_rate=sample_rate)
    result = pipeline(sample)
    return result.output, result.original


class DistillationLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for knowledge distillation training.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        loss_fn: LossDistillationBase,
        validators: tp.Optional[tp.List[ValidatorBase]] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        warm_from_zero: bool = False,
        final_lr: float = 1e-4,
        scale_ratio: float = 1.0,
        total_steps: tp.Optional[int] = None,
        log_every_n_steps: int = 100,
        # --- augmentation ---
        aug_pipeline: tp.Optional["SequentialCompose"] = None,
        aug_teacher_original: bool = True,
        aug_sample_rate: int = 16_000,
    ):
        """
        Args:
            teacher_model:        Pre-trained teacher model (frozen).
            student_model:        Student model to train.
            loss_fn:              Distillation loss function.
            validators:           List of validators to run after each epoch.
            learning_rate:        Learning rate.
            weight_decay:         Weight decay for optimizer.
            warmup_steps:         Number of warmup steps for scheduler.
            total_steps:          Total training steps (for scheduler).
            log_every_n_steps:    Log training metrics every N steps.
            aug_pipeline:         Optional augmentation pipeline.  When ``None``
                                  augmentation is skipped entirely (default behaviour).
            aug_teacher_original: If ``True`` the teacher receives the clean signal;
                                  if ``False`` it receives the augmented signal.
                                  Has no effect when ``aug_pipeline`` is ``None``.
            aug_sample_rate:      Sample rate passed to ``DataSample``.
        """
        super().__init__()

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.loss_fn = loss_fn
        self.validators = validators or []

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.warm_from_zero = warm_from_zero
        self.scale_ratio = scale_ratio
        self.total_steps = total_steps
        self.final_lr = final_lr
        self.log_every_n_steps = log_every_n_steps

        self.aug_pipeline = aug_pipeline
        self.aug_teacher_original = aug_teacher_original
        self.aug_sample_rate = aug_sample_rate

        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Save hyperparameters
        self.save_hyperparameters(ignore=["teacher_model", "student_model", "loss_fn", "validators", "aug_pipeline"])

        # Training context
        self._current_context: tp.Optional[TrainingFlowContext] = None

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass through student model."""
        return self.student_model(x)

    def training_step(
        self,
        batch: BatchSegments,
        batch_idx: int,
    ) -> torch.Tensor:
        # Create context
        self._current_context = TrainingFlowContext(
            epoch=self.current_epoch,
            global_step=self.global_step,
            batch_idx=batch_idx,
        )

        segments_teacher = batch.segments
        segments_student = batch.segments

        if self.aug_pipeline is not None:
            augmented, original = _augment_batch(batch.segments, self.aug_pipeline, self.aug_sample_rate)
            segments_student  = augmented
            segments_teacher  = original if self.aug_teacher_original else augmented

        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_output = self.teacher_model(segments_teacher)

        # Get student outputs
        student_output = self.student_model(segments_student)

        # Compute loss
        loss_result = self.loss_fn(
            student_output=student_output,
            teacher_output=teacher_output,
            targets=batch.target,
            step=self.global_step,
        )

        self._current_context.add_loss("distillation", loss_result)
        self._log_training_losses(loss_result)

        return loss_result.value

    def _log_training_losses(self, loss_result: DFLossBase) -> None:
        """Log training losses to TensorBoard."""
        for field in fields(loss_result):
            value = getattr(loss_result, field.name)
            if isinstance(value, torch.Tensor):
                self.log(
                    f"train/{field.name}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=(field.name == "value"),
                    logger=True,
                )

    def on_validation_epoch_start(self) -> None:
        self.student_model.eval()
        print("on_validation_epoch_start")

    def validation_step(
        self,
        batch: tp.Tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> tp.Optional[torch.Tensor]:
        return None

    def on_train_epoch_end(self) -> None:
        """Run validators at the end of each training epoch."""
        self.student_model.eval()

        if not self.validators:
            return

        device = next(self.student_model.parameters()).device

        for validator in self.validators:
            metrics = validator.run(
                model=self.student_model,
                device=device,
                epoch=self.current_epoch,
                global_step=self.global_step,
            )

            for key, value in metrics.to_tensorboard_dict().items():
                self.log(
                    f"val/{validator.name}/{key}",
                    value,
                    on_epoch=True,
                    logger=True,
                )

            print(f"\n[Validator: {validator.name}] {metrics.to_line()}")

        self.student_model.train()

    def on_train_end(self) -> None:
        """Save validator histories at end of training."""
        for validator in self.validators:
            validator.save_history()

    def configure_optimizers(self) -> tp.Dict[str, tp.Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # optimizer = AdamP(
        #     self.student_model.parameters(),
        #     lr=self.learning_rate,
        #     weight_decay=self.weight_decay,
        # )

        def lr_lambda_exp_decay(step: int) -> float:
            """Exponential decay with linear warmup."""
            if step < self.warmup_steps:
                if self.warm_from_zero:
                    coeff = step / max(1, self.warmup_steps)
                elif self.scale_ratio > 1.0:
                    coeff = (self.scale_ratio - 1.0) * step / max(1, self.warmup_steps) + 1.0
                else:
                    coeff = self.scale_ratio
            else:
                coeff = self.scale_ratio

            decay = math.exp(
                (step / max(1, self.total_steps)) *
                math.log(self.final_lr / self.learning_rate)
            )
            return coeff * decay

        def lr_lambda_cosine(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            if self.total_steps is None:
                return 1.0
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_cosine)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
