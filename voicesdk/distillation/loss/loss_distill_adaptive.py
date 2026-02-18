import typing as tp

import torch
from torch import nn

from ._types import LossDistillationBase, ModelOutput
from .dataflow import DFLossDistillationWithTarget
from .loss_distill_embeddings import LossDistillationEmbeddings
from .loss_distill_logits import LossDistillationLogits


class LossDistillationAdaptive(LossDistillationBase):
    """
    Adaptive distillation loss that shifts weight towards logits/embeddings
    when target logloss is low (model is confident).

    The idea: when the model achieves low target loss, it means the basic
    classification is working well, so we can focus more on knowledge transfer.
    """

    def __init__(
        self,
        base_logits_weight: float = 0.3,
        base_embeddings_weight: float = 0.3,
        base_target_weight: float = 0.4,
        max_distill_boost: float = 0.5,
        logloss_threshold: float = 0.5,
        start_adaptive_step: int = 1000,
        temperature: float = 4.0,
        embedding_loss_type: str = 'mse',
        normalize_embeddings: bool = True,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            base_logits_weight: Base weight for logits distillation
            base_embeddings_weight: Base weight for embeddings distillation
            base_target_weight: Base weight for target loss
            max_distill_boost: Maximum boost to distillation weights when logloss is low
            logloss_threshold: Threshold below which adaptive weighting kicks in
            start_adaptive_step: Step at which to start adaptive weighting
            temperature: Softmax temperature for logits distillation
            embedding_loss_type: Loss type for embeddings
            normalize_embeddings: Whether to normalize embeddings
            label_smoothing: Label smoothing for cross-entropy
        """
        super().__init__()
        self.base_logits_weight = base_logits_weight
        self.base_embeddings_weight = base_embeddings_weight
        self.base_target_weight = base_target_weight
        self.max_distill_boost = max_distill_boost
        self.logloss_threshold = logloss_threshold
        self.start_adaptive_step = start_adaptive_step

        self.logits_loss = LossDistillationLogits(temperature=temperature)
        self.embeddings_loss = LossDistillationEmbeddings(
            loss_type=embedding_loss_type,
            normalize=normalize_embeddings,
        )
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def _compute_adaptive_weights(
        self,
        target_loss_value: float,
        step: int,
    ) -> tp.Tuple[float, float, float]:
        """
        Compute adaptive weights based on current target loss.

        The lower the target loss (below threshold), the more weight
        we shift from target loss to distillation losses.
        """
        if step < self.start_adaptive_step or target_loss_value >= self.logloss_threshold:
            return self.base_logits_weight, self.base_embeddings_weight, self.base_target_weight

        # Calculate boost factor: higher when logloss is lower
        # boost_factor ranges from 0 (at threshold) to 1 (at 0 logloss)
        boost_factor = max(0.0, 1.0 - target_loss_value / self.logloss_threshold)
        boost = boost_factor * self.max_distill_boost

        # Redistribute weights: reduce target weight, increase distillation weights
        target_weight = max(0.1, self.base_target_weight - boost)

        # Split the boost between logits and embeddings proportionally
        total_distill = self.base_logits_weight + self.base_embeddings_weight
        if total_distill > 0:
            logits_ratio = self.base_logits_weight / total_distill
            embeddings_ratio = self.base_embeddings_weight / total_distill
        else:
            logits_ratio = embeddings_ratio = 0.5

        remaining = 1.0 - target_weight
        logits_weight = remaining * logits_ratio
        embeddings_weight = remaining * embeddings_ratio

        return logits_weight, embeddings_weight, target_weight

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

        # Get adaptive weights
        logits_weight, embeddings_weight, target_weight = self._compute_adaptive_weights(
            target_loss.item() if targets is not None else float('inf'),
            step,
        )

        total_loss = (
            logits_weight * logits_loss.value +
            embeddings_weight * emb_loss_value +
            target_weight * target_loss
        )

        return DFLossDistillationWithTarget(
            value=total_loss,
            value_logits=logits_loss.value,
            value_embeddings=emb_loss_value,
            value_target=target_loss,
        )
