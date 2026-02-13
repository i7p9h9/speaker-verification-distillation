import torch
from torch import nn
from torch.utils.data import DataLoader

from voicesdk.distillation.validation._types import ValidatorBase
from ..dataflow.metrics import DFMetricsClassification


class ClassificationValidator(ValidatorBase):
    """Validator for classification tasks."""

    def __init__(
        self,
        name: str,
        dataloader: DataLoader,
        num_classes: int,
        **kwargs,
    ):
        """
        Args:
            name: Validator name
            dataloader: DataLoader with (input, label) pairs
            num_classes: Number of classes
            **kwargs: Additional arguments for ValidatorBase
        """
        super().__init__(name=name, **kwargs)
        self.dataloader = dataloader
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def validate(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> DFMetricsClassification:
        """Run classification evaluation."""
        model.eval()

        total_loss = 0.0
        correct = 0
        correct_top3 = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            for batch in self.dataloader:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                logits = output.logits

                # Loss
                loss = self.ce_loss(logits, labels)
                total_loss += loss.item() * inputs.size(0)

                # Top-1 accuracy
                _, predicted = logits.max(dim=-1)
                correct += (predicted == labels).sum().item()

                # Top-k accuracy
                _, top3_pred = logits.topk(3, dim=-1)
                _, top5_pred = logits.topk(min(5, self.num_classes), dim=-1)

                for i, label in enumerate(labels):
                    if label in top3_pred[i]:
                        correct_top3 += 1
                    if label in top5_pred[i]:
                        correct_top5 += 1

                total += inputs.size(0)

        return DFMetricsClassification(
            accuracy=correct / total if total > 0 else 0.0,
            top3_accuracy=correct_top3 / total if total > 0 else 0.0,
            top5_accuracy=correct_top5 / total if total > 0 else 0.0,
            loss=total_loss / total if total > 0 else 0.0,
            num_samples=total,
        )
