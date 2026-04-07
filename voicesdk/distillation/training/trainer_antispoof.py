"""
Antispoof classifier training (PyTorch Lightning).

Key differences from distillation script:
  - AMSoftmaxLoss head instead of distillation loss
  - LabeledAggregatedDataset + LossWeightingStrategy
  - Per-dataset loss aggregation → strategy.update() every step
  - Dataset weights logged to TensorBoard (absolute + cumsum) and CSV
"""

import csv
import logging
import math
import typing as tp
from collections import defaultdict
from pathlib import Path

import matplotlib as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from voicesdk.dataset import (
    LabeledAggregatedDataset,
    LossWeightingStrategy,
)
from voicesdk.distillation.validation import ValidatorBase
from voicesdk.nn.loss import AMSoftmaxLoss

if tp.TYPE_CHECKING:
    from audimentation import SequentialCompose

log = logging.getLogger(__name__)

# =========================================================================== #
# LR scheduler                                                                 #
# =========================================================================== #

class CosineScheduler:
    """LambdaLR-compatible cosine annealing with linear warmup.

    Args:
        warmup_steps: Steps for linear ramp 0 → 1.
        total_steps: Total training steps (warmup + cosine).
        min_ratio: Floor multiplier at the end of cosine decay.
    """

    def __init__(
        self,
        warmup_steps: int,
        total_steps: int,
        min_ratio: float = 0.0,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_ratio = min_ratio

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(self.min_ratio, cosine)

# =========================================================================== #
# Audimentation                                                               #
# =========================================================================== #

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

# =========================================================================== #
# Per-dataset loss aggregation                                                 #
# =========================================================================== #

def aggregate_loss_per_dataset(
    dataset_names: tp.List[str],
    loss_values: torch.Tensor,
) -> tp.Dict[str, float]:
    """Average detached per-sample losses grouped by dataset name.

    Gradients are NOT retained — only plain Python floats are returned so that
    ``strategy.update()`` never holds a reference to the live computation graph.

    Args:
        dataset_names: One name string per sample in the batch.
        loss_values: Per-sample loss tensor ``(B,)``; detached inside here.

    Returns:
        ``{dataset_name: mean_loss_float}`` for every dataset in the batch.
    """
    sums: tp.Dict[str, float] = defaultdict(float)
    counts: tp.Dict[str, int] = defaultdict(int)
    for name, val in zip(dataset_names, loss_values.detach().cpu().tolist()):
        sums[name]   += val
        counts[name] += 1
    return {name: sums[name] / counts[name] for name in sums}


# =========================================================================== #
# Dataset-weight logger (TensorBoard + CSV)                                    #
# =========================================================================== #

class WeightLogger:
    """Logs dataset sampling weights to TensorBoard via matplotlib figures.

    TensorBoard outputs per call:
      - ``weights/heatmap``      — imshow figure, axes X=step, Y=dataset,
                                   color intensity proportional to weight.
                                   Accumulates history across calls so the
                                   heatmap grows rightward over training.
      - ``weights/snapshot``     — horizontal bar chart of current weights,
                                   useful as a quick readable snapshot.
      - ``weights/absolute/<n>`` — scalar per dataset (raw weight value).
      - ``weights/cumsum/<n>``   — scalar per dataset (cumulative sum).

    Both figures are created with ``add_figure`` so matplotlib handles
    colorbars, axis labels, and tick names — no manual pixel math.

    Args:
        tb_logger:          TensorBoardLogger attached to the trainer.
        csv_path:           Destination path for the CSV weight log.
        log_every_n_steps:  Write interval in global training steps.
        max_history:        Maximum number of steps kept in the heatmap
                            buffer (older entries are dropped, default 500).
    """

    def __init__(
        self,
        tb_logger: TensorBoardLogger,
        csv_path: Path,
        log_every_n_steps: int = 10,
        max_history: int = 500,
    ) -> None:
        self._tb = tb_logger
        self._path = csv_path
        self._every = log_every_n_steps
        self._max_history = max_history

        # Heatmap history: list of (step, {name: weight}) snapshots
        self._history: tp.List[tp.Tuple[int, tp.Dict[str, float]]] = []

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not csv_path.exists():
            with csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(["step", "dataset", "weight"])

    # ---------------------------------------------------------------------- #
    # Figure builders                                                          #
    # ---------------------------------------------------------------------- #

    def _make_heatmap(self) -> "plt.Figure":
        """Build an imshow heatmap from accumulated history.

        Returns:
            matplotlib Figure — X axis = training steps, Y axis = datasets,
            color = sampling weight (viridis colormap).
        """
        import matplotlib.pyplot as plt

        steps = [s for s, _ in self._history]
        names = sorted(self._history[0][1])          # stable Y-axis order
        n_ds = len(names)
        n_t = len(steps)

        # Build matrix [n_datasets, n_steps]
        matrix = np.zeros((n_ds, n_t), dtype=np.float32)
        for t_idx, (_, w_dict) in enumerate(self._history):
            for d_idx, name in enumerate(names):
                matrix[d_idx, t_idx] = np.log(w_dict.get(name, 0.0))

        fig, ax = plt.subplots(figsize=(max(12, n_t * 0.06 + 2), max(3, n_ds * 0.4 + 1)))
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            interpolation="nearest",
        )
        # fig.colorbar(im, ax=ax, label="weight")

        ax.set_yticks(range(n_ds))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("training step")
        ax.set_title("dataset sampling weights")

        # Show only a reasonable number of X tick labels
        max_xticks = 10
        tick_indices = np.linspace(0, n_t - 1, min(max_xticks, n_t), dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([str(steps[i]) for i in tick_indices], rotation=45, ha="right", fontsize=7)

        # Compute left margin from the longest dataset name so ytick labels
        # never overflow — tight_layout raises warnings when label strings are
        # too long for it to resolve margins automatically.
        max_name_len = max(len(n) for n in names)
        left_margin = min(0.02 + max_name_len * 0.012, 0.55)
        fig.subplots_adjust(left=left_margin, right=0.88, top=0.92, bottom=0.18)
        return fig

    def _make_snapshot(
        self,
        names: tp.List[str],
        values: tp.List[float],
    ) -> "plt.Figure":
        """Build a horizontal bar chart of current weights.

        Args:
            names:  Dataset names (Y axis, sorted alphabetically).
            values: Corresponding weight values.

        Returns:
            matplotlib Figure.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, max(2, len(names) * 0.35 + 0.5)))
        y_pos = range(len(names))
        ax.barh(list(y_pos), np.log(1 + np.array(values) / np.max(values)), color="steelblue", height=0.6)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("weight")
        ax.set_title("dataset weights (current step)")
        ax.invert_yaxis()
        max_name_len = max((len(n) for n in names), default=10)
        left_margin = min(0.02 + max_name_len * 0.015, 0.6)
        fig.subplots_adjust(left=left_margin, right=0.95, top=0.90, bottom=0.18)
        return fig

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    def maybe_log(
        self,
        weights: tp.Dict[str, float],
        global_step: int,
    ) -> None:
        """Write a weight snapshot if the logging interval has been reached.

        Args:
            weights:     ``{dataset_name: weight}`` from
                         ``LabeledAggregatedDataset.get_weights()``.
            global_step: Current global training step.
        """
        if global_step % self._every != 0:
            return

        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend, safe in training loops

        writer = self._tb.experiment   # underlying SummaryWriter
        names = sorted(weights)
        values = [weights[n] for n in names]

        # ---- accumulate history for heatmap ------------------------------
        self._history.append((global_step, dict(weights)))
        if len(self._history) > self._max_history:
            self._history.pop(0)

        # ---- heatmap (full history) --------------------------------------
        fig_heatmap = self._make_heatmap()
        writer.add_figure("weights/heatmap", fig_heatmap, global_step=global_step)

        # ---- snapshot (current step) -------------------------------------
        fig_snap = self._make_snapshot(names, values)
        writer.add_figure("weights/snapshot", fig_snap, global_step=global_step)

        # ---- scalars + CSV -----------------------------------------------
        cumsum = 0.0
        with self._path.open("a", newline="") as f:
            csv_w = csv.writer(f)
            for name, val in zip(names, values):
                cumsum += val
                writer.add_scalar(f"weights/absolute/{name}", val,    global_step)
                writer.add_scalar(f"weights/cumsum/{name}",   cumsum, global_step)
                csv_w.writerow([global_step, name, val])

class JoinedModel(nn.Module):
    def __init__(self, backbone, loss):
        super().__init__()
        self.backbone = backbone
        self.loss = loss

    def predict(self, x):
        emb = self.backbone(x)
        return self.loss.predict(emb)

    def forward(self, x, labels=None):
        if labels is None:
            return self.predict(x)
        emb = self.backbone(x)
        return self.loss(emb, labels)

# =========================================================================== #
# Lightning module                                                             #
# =========================================================================== #

class AntispoofLightningModule(pl.LightningModule):
    """Lightning module for binary antispoof training with AM-Softmax loss.

    Accepts a ``LabeledAggregatedDataset`` + ``LossWeightingStrategy`` pair
    and re-weights sampling each step based on per-dataset mean loss.

    Args:
        backbone: Feature extractor; output ``(B, embedding_dim)``.
        am_loss: AMSoftmaxLoss head (holds the class prototype matrix).
        dataset: Labeled training dataset (for weight queries).
        strategy: LossWeightingStrategy wrapping the same dataset.
        weight_logger: WeightLogger for TB + CSV.
        learning_rate: Peak learning rate for AdamW.
        weight_decay: L2 regularisation coefficient.
        warmup_steps: Linear warmup duration in steps.
        total_steps: Total training steps for cosine schedule.
        min_lr_ratio: Floor LR multiplier at end of cosine decay.
        log_every_n_steps: Interval for scalar loss logging.
        aug_pipeline: Optional augmentation pipeline.  When ``None``
            augmentation is skipped entirely (default behaviour).
    """

    def __init__(
        self,
        backbone: nn.Module,
        am_loss: AMSoftmaxLoss,
        dataset: LabeledAggregatedDataset,
        strategy: LossWeightingStrategy,
        weight_logger: WeightLogger,
        validators: tp.Optional[tp.List[ValidatorBase]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        warmup_steps: int = 500,
        total_steps: int = 150_000,
        min_lr_ratio: float = 0.0,
        log_every_n_steps: int = 50,
        aug_pipeline: tp.Optional["SequentialCompose"] = None,
        aug_sample_rate: int = 16_000,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.am_loss = am_loss
        self.joined_model = JoinedModel(backbone=backbone, loss=am_loss)
        self.dataset = dataset
        self.strategy = strategy
        self.weight_logger = weight_logger
        self.validators = validators

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.log_every_n_steps = log_every_n_steps
        self.aug_pipeline = aug_pipeline
        self.aug_sample_rate = aug_sample_rate

        self.save_hyperparameters(
            ignore=["backbone", "am_loss", "dataset", "strategy", "weight_logger", "aug_pipeline"]
        )

    # ----------------------------------------------------------------------- #
    # Forward                                                                   #
    # ----------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        return self.am_loss.predict(emb)

    # ----------------------------------------------------------------------- #
    # Training                                                                  #
    # ----------------------------------------------------------------------- #

    def training_step(
        self,
        batch: tp.Any,  # BatchLabeledSegments
        batch_idx: int,
    ) -> torch.Tensor:
        segments: torch.Tensor = batch.segments       # (B, L)
        labels: torch.Tensor = batch.labels         # (B,)
        dataset_names: tp.List[str] = batch.dataset_names  # len == B

        if self.aug_pipeline is not None:
            segments, _ = _augment_batch(batch.segments, self.aug_pipeline, self.aug_sample_rate)

        embeddings = self.backbone(segments)              # (B, D)
        am_out = self.am_loss(embeddings, labels)     # AMSoftmaxOutput

        # ---- per-dataset loss aggregation --------------------------------
        # aggregate_loss_per_dataset detaches internally → no graph retained
        per_dataset_loss = aggregate_loss_per_dataset(
            dataset_names, am_out.loss_values
        )
        self.strategy.update(per_dataset_loss)

        # ---- dataset weight logging (TB + CSV) ---------------------------
        self.weight_logger.maybe_log(
            self.dataset.get_weights(),
            global_step=self.global_step,
        )

        # ---- scalar loss logging -----------------------------------------
        do_log = (self.global_step % self.log_every_n_steps == 0)
        self.log(
            "train/loss", am_out.loss,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True,
        )
        if do_log:
            writer = self.logger.experiment  # type: ignore[union-attr]
            for name, val in per_dataset_loss.items():
                writer.add_scalar(
                    f"train/loss_per_dataset/{name}", val, self.global_step
                )

        return am_out.loss  # backward is called on this scalar

    # ----------------------------------------------------------------------- #
    # Validation (stub — add external Callback validators as needed)          #
    # ----------------------------------------------------------------------- #

    def validation_step(self, batch: tp.Any, batch_idx: int) -> None:
        return None

    def on_train_epoch_end(self) -> None:
        """Run validators at the end of each training epoch."""
        self.joined_model.eval()

        if not self.validators:
            return

        device = next(self.joined_model.parameters()).device

        for validator in self.validators:
            metrics = validator.run(
                model=self.joined_model,
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

        self.joined_model.train()

    # ----------------------------------------------------------------------- #
    # Optimizer + LR scheduler                                                #
    # ----------------------------------------------------------------------- #

    def configure_optimizers(self) -> tp.Dict[str, tp.Any]:
        params = (
            list(self.backbone.parameters())
            + list(self.am_loss.parameters())
        )

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        sched_fn = CosineScheduler(
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_ratio=self.min_lr_ratio,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=sched_fn
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
