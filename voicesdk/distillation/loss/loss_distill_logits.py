import typing as tp

import torch
import torch.nn.functional as F
from torch import nn

from ._types import LossDistillationBase, ModelOutput
from .dataflow import DFLossDistillationLogits


class LossDistillationLogits(LossDistillationBase):
    """
    Knowledge distillation loss using soft targets from logits.
    Uses KL divergence between teacher and student softmax outputs.
    """

    def __init__(self, temperature: float = 4.0):
        """
        Args:
            temperature: Softmax temperature for softening distributions
        """
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(
        self,
        student_output: ModelOutput,
        teacher_output: ModelOutput,
        targets: tp.Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> DFLossDistillationLogits:
        student_logits = student_output.logits
        teacher_logits = teacher_output.logits

        # Soft targets
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence scaled by T^2
        loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        return DFLossDistillationLogits(value=loss)
