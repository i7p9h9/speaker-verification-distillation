import typing as tp
import torch

from .loss_distill_embeddings import LossDistillationEmbeddings
from .loss_distill_logits import LossDistillationLogits
from ._types import ModelOutput
from .dataflow import DFLossDistillationMixed
from ._types import LossDistillationBase


class LossMixedDistillation(LossDistillationBase):
    """
    Combined logits and embeddings distillation loss.
    """

    def __init__(
        self,
        logits_weight: float = 0.5,
        embeddings_weight: float = 0.5,
        temperature: float = 4.0,
        embedding_loss_type: str = 'mse',
        normalize_embeddings: bool = True,
    ):
        """
        Args:
            logits_weight: Weight for logits loss
            embeddings_weight: Weight for embeddings loss
            temperature: Softmax temperature for logits distillation
            embedding_loss_type: Loss type for embeddings
            normalize_embeddings: Whether to normalize embeddings
        """
        super().__init__()
        self.logits_weight = logits_weight
        self.embeddings_weight = embeddings_weight

        self.logits_loss = LossDistillationLogits(temperature=temperature)
        self.embeddings_loss = LossDistillationEmbeddings(
            loss_type=embedding_loss_type,
            normalize=normalize_embeddings,
        )

    def forward(
        self,
        student_output: ModelOutput,
        teacher_output: ModelOutput,
        targets: tp.Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> DFLossDistillationMixed:
        logits_loss = self.logits_loss(student_output, teacher_output)
        embeddings_loss = self.embeddings_loss(student_output, teacher_output)

        total_loss = (
            self.logits_weight * logits_loss.value +
            self.embeddings_weight * embeddings_loss.value
        )

        return DFLossDistillationMixed(
            value=total_loss,
            value_logits=logits_loss.value,
            value_embeddings=embeddings_loss.value,
        )
