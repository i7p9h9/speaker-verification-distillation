import math
import typing as tp
from enum import Enum

import torch

from .loss_distill_embeddings import LossDistillationEmbeddings
from .loss_distill_logits import LossDistillationLogits
from ._types import ModelOutput
from .dataflow import DFLossDistillationMixed
from ._types import LossDistillationBase


class ScheduledMixedDistillationLoss(LossDistillationBase):
    """
    Mixed distillation loss with scheduled weight changes over training.
    """

    class ScheduleType(Enum):
        LINEAR = 'linear'
        COSINE = 'cosine'
        STEP = 'step'

    def __init__(
        self,
        initial_logits_weight: float = 0.5,
        final_logits_weight: float = 0.8,
        initial_embeddings_weight: float = 0.5,
        final_embeddings_weight: float = 0.2,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        schedule_type: str = 'linear',
        temperature: float = 4.0,
        embedding_loss_type: str = 'mse',
        normalize_embeddings: bool = True,
    ):
        """
        Args:
            initial_logits_weight: Starting weight for logits loss
            final_logits_weight: Final weight for logits loss
            initial_embeddings_weight: Starting weight for embeddings loss
            final_embeddings_weight: Final weight for embeddings loss
            warmup_steps: Steps before scheduling starts
            total_steps: Total training steps
            schedule_type: Type of schedule ('linear', 'cosine', 'step')
            temperature: Softmax temperature for logits distillation
            embedding_loss_type: Loss type for embeddings
            normalize_embeddings: Whether to normalize embeddings
        """
        super().__init__()
        self.initial_logits_weight = initial_logits_weight
        self.final_logits_weight = final_logits_weight
        self.initial_embeddings_weight = initial_embeddings_weight
        self.final_embeddings_weight = final_embeddings_weight
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule_type = self.ScheduleType(schedule_type)

        self.logits_loss = LossDistillationLogits(temperature=temperature)
        self.embeddings_loss = LossDistillationEmbeddings(
            loss_type=embedding_loss_type,
            normalize=normalize_embeddings,
        )

    def _get_weights(self, step: int) -> tp.Tuple[float, float]:
        """Calculate current weights based on step and schedule."""
        if step < self.warmup_steps:
            return self.initial_logits_weight, self.initial_embeddings_weight

        progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))

        if self.schedule_type == self.ScheduleType.LINEAR:
            factor = progress
        elif self.schedule_type == self.ScheduleType.COSINE:
            factor = 0.5 * (1 - math.cos(math.pi * progress))
        elif self.schedule_type == self.ScheduleType.STEP:
            factor = 1.0 if progress > 0.5 else 0.0
        else:
            factor = progress

        logits_weight = self.initial_logits_weight + factor * (
            self.final_logits_weight - self.initial_logits_weight
        )
        embeddings_weight = self.initial_embeddings_weight + factor * (
            self.final_embeddings_weight - self.initial_embeddings_weight
        )

        return logits_weight, embeddings_weight

    def forward(
        self,
        student_output: ModelOutput,
        teacher_output: ModelOutput,
        targets: tp.Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> DFLossDistillationMixed:
        logits_weight, embeddings_weight = self._get_weights(step)

        logits_loss = self.logits_loss(student_output, teacher_output)
        embeddings_loss = self.embeddings_loss(student_output, teacher_output)

        total_loss = (
            logits_weight * logits_loss.value +
            embeddings_weight * embeddings_loss.value
        )

        return DFLossDistillationMixed(
            value=total_loss,
            value_logits=logits_loss.value,
            value_embeddings=embeddings_loss.value,
        )
