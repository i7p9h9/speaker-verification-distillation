import typing as tp
from dataclasses import asdict, dataclass, field

import torch

from voicesdk.distillation.loss import DFLossBase
from voicesdk.distillation.metrics.dataflow import DFMetricsBase


@dataclass
class TrainingFlowContext:
    """
    Context object passed through training pipeline stages.
    Allows each stage to add its results for logging and tracking.
    """
    epoch: int = 0
    global_step: int = 0
    batch_idx: int = 0

    # Stage results
    losses: tp.Dict[str, DFLossBase] = field(default_factory=dict)
    metrics: tp.Dict[str, DFMetricsBase] = field(default_factory=dict)
    custom_data: tp.Dict[str, tp.Any] = field(default_factory=dict)

    def add_loss(self, name: str, loss: DFLossBase) -> None:
        """Add loss result from a training stage."""
        self.losses[name] = loss

    def add_metrics(self, name: str, metrics: DFMetricsBase) -> None:
        """Add metrics from a validation stage."""
        self.metrics[name] = metrics

    def add_custom(self, name: str, value: tp.Any) -> None:
        """Add custom data to context."""
        self.custom_data[name] = value

    def get_tensorboard_scalars(self) -> tp.Dict[str, float]:
        """Get all scalars for TensorBoard logging."""
        scalars = {}

        # Add losses
        for name, loss in self.losses.items():
            loss_dict = asdict(loss)
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    scalars[f"loss/{name}/{k}"] = v.item()
                elif isinstance(v, (int, float)):
                    scalars[f"loss/{name}/{k}"] = float(v)

        # Add metrics
        for name, metric in self.metrics.items():
            for k, v in metric.to_tensorboard_dict().items():
                scalars[f"val/{name}/{k}"] = v

        return scalars
