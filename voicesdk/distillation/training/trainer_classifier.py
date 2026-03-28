import typing as tp
from dataclasses import dataclass, fields

import pytorch_lightning as pl
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Re-exported types (assumed to live in your voicesdk package)                #
# --------------------------------------------------------------------------- #
from voicesdk.distillation.data import BatchSegments
from voicesdk.distillation.loss import CCELossOutput, ModelOutput
from voicesdk.distillation.loss.classifier import LossCCE
from voicesdk.distillation.validation import ValidatorBase

if tp.TYPE_CHECKING:
    from audimentation import SequentialCompose

# =========================================================================== #
# Schedulers                                                                   #
# =========================================================================== #

class SchedulerBase:
    """Abstract LR lambda scheduler.

    Subclasses must implement :meth:`__call__` and are meant to be passed
    directly to ``torch.optim.lr_scheduler.LambdaLR``.

    All schedulers are *callables* so that the factory pattern is simple::

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduler_fn)
    """

    def __call__(self, step: int) -> float:
        raise NotImplementedError


class CosineScheduler(SchedulerBase):
    """Cosine annealing with linear warmup.

    Args:
        warmup_steps: Number of linear warmup steps.
        total_steps:  Total number of training steps.
        min_ratio:    Minimum LR ratio at the end of decay (default ``0.1``).
    """

    def __init__(
        self,
        warmup_steps: int,
        total_steps: int,
        min_ratio: float = 0.1,
    ) -> None:
        import math
        self._math = math
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_ratio = min_ratio

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        return max(
            self.min_ratio,
            0.5 * (1.0 + self._math.cos(self._math.pi * progress)),
        )


class ExpDecayScheduler(SchedulerBase):
    """Exponential decay with linear warmup.

    Args:
        warmup_steps:  Number of warmup steps.
        total_steps:   Total number of training steps.
        initial_lr:    Base (peak) learning rate — used to compute the
                       target decay ratio relative to ``final_lr``.
        final_lr:      Desired learning rate at ``total_steps``.
        scale_ratio:   Peak multiplier applied after warmup (default ``1.0``).
        warm_from_zero: If ``True``, warmup starts from 0; otherwise starts
                        from ``scale_ratio * initial_lr``.
    """

    def __init__(
        self,
        warmup_steps: int,
        total_steps: int,
        initial_lr: float,
        final_lr: float,
        scale_ratio: float = 1.0,
        warm_from_zero: bool = False,
    ) -> None:
        import math
        self._math = math
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.scale_ratio = scale_ratio
        self.warm_from_zero = warm_from_zero

    def __call__(self, step: int) -> float:
        math = self._math
        if step < self.warmup_steps:
            if self.warm_from_zero:
                coeff = step / max(1, self.warmup_steps)
            elif self.scale_ratio > 1.0:
                coeff = (
                    (self.scale_ratio - 1.0) * step / max(1, self.warmup_steps)
                    + 1.0
                )
            else:
                coeff = self.scale_ratio
        else:
            coeff = self.scale_ratio

        decay = math.exp(
            (step / max(1, self.total_steps))
            * math.log(self.final_lr / max(1e-12, self.initial_lr))
        )
        return coeff * decay


# =========================================================================== #
# Factory type aliases                                                         #
# =========================================================================== #

#: A callable that receives a parameter iterable and returns an optimizer.
OptimizerFactory = tp.Callable[
    [tp.Iterable[nn.Parameter]],
    torch.optim.Optimizer,
]

#: A callable that receives an optimizer and returns a LR scheduler.
SchedulerFactory = tp.Callable[
    [torch.optim.Optimizer],
    torch.optim.lr_scheduler.LRScheduler,
]


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

    sample = DataSample(
        signal=segments,
        original=segments.clone(),
        sample_rate=sample_rate,
    )
    result = pipeline(sample)
    return result.output, result.original


# =========================================================================== #
# Lightning module                                                              #
# =========================================================================== #

class TrainingLightningModule(pl.LightningModule):
    """PyTorch Lightning module for supervised training with CCE loss.

    Optimizer and LR scheduler are supplied externally via factory callables,
    keeping the module free of hardcoded training policy.

    Example::

        import torch
        from functools import partial

        optimizer_factory = partial(
            torch.optim.AdamW, lr=1e-4, weight_decay=0.01
        )

        scheduler_fn = CosineScheduler(warmup_steps=500, total_steps=10_000)
        scheduler_factory = lambda opt: torch.optim.lr_scheduler.LambdaLR(
            opt, scheduler_fn
        )

        module = TrainingLightningModule(
            model=my_model,
            loss_fn=CCELoss(),
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
        )
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: LossCCE,
        optimizer_factory: OptimizerFactory,
        scheduler_factory: tp.Optional[SchedulerFactory] = None,
        validators: tp.Optional[tp.List[ValidatorBase]] = None,
        log_every_n_steps: int = 100,
        # --- augmentation ---
        aug_pipeline: tp.Optional["SequentialCompose"] = None,
        aug_sample_rate: int = 16_000,
    ) -> None:
        """
        Args:
            model:              Model to train.
            loss_fn:            CCE loss instance.
            optimizer_factory:  Callable ``(params) -> Optimizer``.
            scheduler_factory:  Optional callable ``(optimizer) -> Scheduler``.
                                When ``None``, no LR scheduling is applied.
            validators:         Validators executed at the end of each epoch.
            log_every_n_steps:  Log training metrics every N steps.
            aug_pipeline:       Optional augmentation pipeline.  When ``None``
                                augmentation is skipped entirely.
            aug_sample_rate:    Sample rate passed to the augmentation pipeline.
        """
        super().__init__()

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.validators = validators or []
        self.log_every_n_steps = log_every_n_steps

        self.aug_pipeline = aug_pipeline
        self.aug_sample_rate = aug_sample_rate

        self.save_hyperparameters(
            ignore=["model", "loss_fn", "optimizer_factory", "scheduler_factory",
                    "validators", "aug_pipeline"]
        )

    # ----------------------------------------------------------------------- #
    # Forward                                                                   #
    # ----------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass through the model."""
        return self.model(x)

    # ----------------------------------------------------------------------- #
    # Training                                                                  #
    # ----------------------------------------------------------------------- #

    def training_step(
        self,
        batch: BatchSegments,
        batch_idx: int,
    ) -> torch.Tensor:
        segments = batch.segments

        if self.aug_pipeline is not None:
            segments, _ = _augment_batch(
                segments, self.aug_pipeline, self.aug_sample_rate
            )

        output: ModelOutput = self.model(segments)

        loss_output: CCELossOutput = self.loss_fn(output, batch.target)

        self._log_loss(loss_output, prefix="train")

        return loss_output.value

    # ----------------------------------------------------------------------- #
    # Validation                                                                #
    # ----------------------------------------------------------------------- #

    def on_validation_epoch_start(self) -> None:
        self.model.eval()

    def validation_step(
        self,
        batch: tp.Tuple[torch.Tensor, ...],
        batch_idx: int,
    ) -> tp.Optional[torch.Tensor]:
        return None

    # ----------------------------------------------------------------------- #
    # End of epoch hooks                                                        #
    # ----------------------------------------------------------------------- #

    def on_train_epoch_end(self) -> None:
        """Run validators at the end of each training epoch."""
        self.model.eval()

        if not self.validators:
            self.model.train()
            return

        device = next(self.model.parameters()).device

        for validator in self.validators:
            metrics = validator.run(
                model=self.model,
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

        self.model.train()

    def on_train_end(self) -> None:
        """Persist validator histories at the end of training."""
        for validator in self.validators:
            validator.save_history()

    # ----------------------------------------------------------------------- #
    # Optimizer & scheduler                                                     #
    # ----------------------------------------------------------------------- #

    def configure_optimizers(
        self,
    ) -> tp.Union[torch.optim.Optimizer, tp.Dict[str, tp.Any]]:
        """Build optimizer (and optionally scheduler) from external factories."""
        optimizer = self.optimizer_factory(self.model.parameters())

        if self.scheduler_factory is None:
            return optimizer

        scheduler = self.scheduler_factory(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ----------------------------------------------------------------------- #
    # Helpers                                                                   #
    # ----------------------------------------------------------------------- #

    def _log_loss(self, loss_output: CCELossOutput, prefix: str) -> None:
        """Log all tensor fields of the loss output."""
        for field in fields(loss_output):
            value = getattr(loss_output, field.name)
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                self.log(
                    f"{prefix}/{field.name}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=(field.name == "value"),
                    logger=True,
                )
