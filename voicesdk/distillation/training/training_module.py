import math
import typing as tp
from dataclasses import asdict

import lightning as pl
import torch
from torch import nn

from voicesdk.distillation.loss import DFLossBase, LossDistillationBase, ModelOutput
from voicesdk.distillation.validation import ValidatorBase

from .flow_context import TrainingFlowContext


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
        total_steps: tp.Optional[int] = None,
        log_every_n_steps: int = 100,
    ):
        """
        Args:
            teacher_model: Pre-trained teacher model (frozen)
            student_model: Student model to train
            loss_fn: Distillation loss function
            validators: List of validators to run after each epoch
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for scheduler
            total_steps: Total training steps (for scheduler)
            log_every_n_steps: Log training metrics every N steps
        """
        super().__init__()

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.loss_fn = loss_fn
        self.validators = validators or []

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.log_every_n_steps = log_every_n_steps

        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Save hyperparameters
        self.save_hyperparameters(ignore=["teacher_model", "student_model", "loss_fn", "validators"])

        # Training context
        self._current_context: tp.Optional[TrainingFlowContext] = None

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass through student model."""
        return self.student_model(x)

    def training_step(
        self,
        batch: tp.Tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Tuple of (inputs, targets) or just (inputs,)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # Create context
        self._current_context = TrainingFlowContext(
            epoch=self.current_epoch,
            global_step=self.global_step,
            batch_idx=batch_idx,
        )

        # Unpack batch
        if len(batch) == 2:
            inputs, targets = batch
        else:
            inputs = batch[0]
            targets = None

        # Get teacher outputs (no gradients)
        with torch.no_grad():
            teacher_output = self.teacher_model(inputs)

        # Get student outputs
        student_output = self.student_model(inputs)

        # Compute loss
        loss_result = self.loss_fn(
            student_output=student_output,
            teacher_output=teacher_output,
            targets=targets,
            step=self.global_step,
        )

        # Add to context
        self._current_context.add_loss("distillation", loss_result)

        # Log losses
        self._log_training_losses(loss_result)

        return loss_result.value

    def _log_training_losses(self, loss_result: DFLossBase) -> None:
        """Log training losses to TensorBoard."""
        loss_dict = asdict(loss_result)

        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.log(
                    f"train/{key}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=(key == "value"),
                    logger=True,
                )

    def on_validation_epoch_start(self) -> None:
        """Called at the start of validation epoch."""
        self.student_model.eval()

    def validation_step(
        self,
        batch: tp.Tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> tp.Optional[torch.Tensor]:
        """
        Validation step (optional, for dataloader-based validation).
        Override if you need batch-wise validation.
        """
        return None

    def on_validation_epoch_end(self) -> None:
        """Run validators at the end of each validation epoch."""
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

            # Log to TensorBoard
            for key, value in metrics.to_tensorboard_dict().items():
                self.log(
                    f"val/{validator.name}/{key}",
                    value,
                    on_epoch=True,
                    logger=True,
                )

            # Print metrics
            print(f"\n[Validator: {validator.name}] {metrics.to_line()}")

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

        # Warmup + cosine decay scheduler
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            if self.total_steps is None:
                return 1.0
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
