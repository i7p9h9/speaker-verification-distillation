import typing as tp

import torch
from torch import nn

from .loss_distill_embeddings import LossDistillationEmbeddings
from .loss_distill_logits import LossDistillationLogits
from ._types import LossDistillationBase, ModelOutput
from .dataflow import DFLossDistillationWithTarget


class LossDistillationWithTarget(LossDistillationBase):
    """
    Combined distillation loss with ground truth target loss.
    Supports logits distillation, embeddings distillation, and target cross-entropy.
    """

    def __init__(
        self,
        logits_weight: float = 0.4,
        embeddings_weight: float = 0.3,
        target_weight: float = 0.3,
        temperature: float = 4.0,
        embedding_loss_type: str = 'mse',
        normalize_embeddings: bool = True,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            logits_weight: Weight for logits distillation loss
            embeddings_weight: Weight for embeddings distillation loss
            target_weight: Weight for target cross-entropy loss
            temperature: Softmax temperature for logits distillation
            embedding_loss_type: Loss type for embeddings
            normalize_embeddings: Whether to normalize embeddings
            label_smoothing: Label smoothing for cross-entropy
        """
        super().__init__()
        self.logits_weight = logits_weight
        self.embeddings_weight = embeddings_weight
        self.target_weight = target_weight

        self.logits_loss = LossDistillationLogits(temperature=temperature)
        self.embeddings_loss = LossDistillationEmbeddings(
            loss_type=embedding_loss_type,
            normalize=normalize_embeddings,
        )
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        student_output: ModelOutput,
        teacher_output: ModelOutput,
        targets: tp.Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> DFLossDistillationWithTarget:
        logits_loss = self.logits_loss(student_output, teacher_output)

        # Embeddings loss (optional)
        if student_output.embeddings is not None and teacher_output.embeddings is not None:
            embeddings_loss = self.embeddings_loss(student_output, teacher_output)
            emb_loss_value = embeddings_loss.value
        else:
            emb_loss_value = torch.tensor(0.0, device=student_output.logits.device)

        # Target loss
        if targets is not None:
            target_loss = self.ce_loss(student_output.logits, targets)
        else:
            target_loss = torch.tensor(0.0, device=student_output.logits.device)

        total_loss = (
            self.logits_weight * logits_loss.value +
            self.embeddings_weight * emb_loss_value +
            self.target_weight * target_loss
        )

        return DFLossDistillationWithTarget(
            value=total_loss,
            value_logits=logits_loss.value,
            value_embeddings=emb_loss_value,
            value_target=target_loss,
        )
