import typing as tp
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DFLossBase:
    """Base class for all loss outputs."""
    value: torch.Tensor
    loss_batch: torch.Tensor | None = None

    @property
    def loss(self) -> torch.Tensor:
        return self.value

    def backward(self) -> None:
        self.value.backward()

    def item(self) -> float:
        return self.value.item()

    def detach(self) -> torch.Tensor:
        return self.value.detach()


@dataclass
class ModelOutput:
    """
    Standard output from teacher/student models.

    Attributes:
        logits: Model logits, shape can vary (batch, classes) or (batch, seq, classes)
        embeddings: Optional embeddings - can be single tensor or list of tensors
                    Shape can be (batch, dim) or (batch, seq, dim) or list of such
    """
    logits: torch.Tensor
    embeddings: tp.Optional[tp.Union[torch.Tensor, tp.List[torch.Tensor]]] = None


class LossDistillationBase(nn.Module, ABC):
    """Abstract base class for distillation losses."""

    @abstractmethod
    def forward(
        self,
        student_output: ModelOutput,
        teacher_output: ModelOutput,
        targets: tp.Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> DFLossBase:
        """
        Compute distillation loss.

        Args:
            student_output: Output from student model
            teacher_output: Output from teacher model
            targets: Optional ground truth labels
            step: Current training step (for scheduled losses)

        Returns:
            Loss dataclass with computed loss value(s)
        """
        pass


@dataclass
class CCELossOutput(DFLossBase):
    """Output of the CCE loss function.

    Attributes:
        value:     Scalar cross-entropy loss tensor.
        embedding: Optional mean pooled embedding tensor ``[N, D]``,
                   kept for downstream use (e.g. metric logging, probing).
    """
    value: torch.Tensor
    embedding: tp.Optional[torch.Tensor] = None