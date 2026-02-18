import json
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn

from voicesdk.distillation.metrics import DFMetricsBase


class ValidatorBase(ABC):
    """
    Abstract base class for validation.

    Implement this class to create custom validators for your specific task.
    """

    def __init__(
        self,
        name: str,
        save_checkpoint: bool = False,
        checkpoint_dir: tp.Optional[Path] = None,
        log_dir: tp.Optional[Path] = None,
        metric_for_best: tp.Optional[str] = None,
        metric_mode: str = 'min',
    ):
        """
        Args:
            name: Name of the validator (used for logging)
            save_checkpoint: Whether to save model checkpoint
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save metric logs
            metric_for_best: Metric name to track for best model
            metric_mode: 'min' or 'max' for best model tracking
        """
        self.name = name
        self.save_checkpoint = save_checkpoint
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.log_dir = Path(log_dir) if log_dir else None
        self.metric_for_best = metric_for_best
        self.metric_mode = metric_mode

        self.best_metric_value: tp.Optional[float] = None
        self.metrics_history: tp.List[DFMetricsBase] = []

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def validate(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> DFMetricsBase:
        """
        Run validation and return metrics.

        Args:
            model: Model to validate
            device: Device to run validation on

        Returns:
            Metrics dataclass with validation results
        """
        pass

    def _is_better(self, new_value: float) -> bool:
        """Check if new metric value is better than current best."""
        if self.best_metric_value is None:
            return True
        if self.metric_mode == 'min':
            return new_value < self.best_metric_value
        return new_value > self.best_metric_value

    def run(
        self,
        model: nn.Module,
        device: torch.device,
        epoch: int,
        global_step: int,
    ) -> DFMetricsBase:
        """
        Run validation, log results, and optionally save checkpoint.

        Args:
            model: Model to validate
            device: Device to run validation on
            epoch: Current epoch
            global_step: Current global training step

        Returns:
            Metrics dataclass with validation results
        """
        metrics = self.validate(model, device)
        self.metrics_history.append(metrics)

        # Log metrics
        if self.log_dir:
            log_file = self.log_dir / f"{self.name}_metrics.log"
            with open(log_file, 'a') as f:
                f.write(f"epoch={epoch} step={global_step} {metrics.to_line()}\n")

        # Check for best model and save checkpoint
        if self.save_checkpoint and self.metric_for_best and self.checkpoint_dir:
            metric_dict = metrics.to_dict()
            if self.metric_for_best in metric_dict:
                current_value = metric_dict[self.metric_for_best]
                if self._is_better(current_value):
                    self.best_metric_value = current_value
                    checkpoint_path = self.checkpoint_dir / f"{self.name}_best.pt"
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'metrics': metrics.to_dict(),
                    }, checkpoint_path)

        return metrics

    def save_history(self) -> None:
        """Save full metrics history to JSON file."""
        if self.log_dir:
            history_file = self.log_dir / f"{self.name}_history.json"
            history_data = [m.to_dict() for m in self.metrics_history]
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2)
