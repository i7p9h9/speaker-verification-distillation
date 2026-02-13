import typing as tp
import lightning as pl
from pytorch_lightning.callbacks import Callback

from voicesdk.distillation.training.training_module import DistillationLightningModule


class ValidationCallback(Callback):
    """
    Lightning callback to run validation at specific intervals.
    """

    def __init__(
        self,
        validate_every_n_epochs: int = 1,
        validate_every_n_steps: tp.Optional[int] = None,
    ):
        """
        Args:
            validate_every_n_epochs: Run validation every N epochs
            validate_every_n_steps: Run validation every N steps (overrides epochs)
        """
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