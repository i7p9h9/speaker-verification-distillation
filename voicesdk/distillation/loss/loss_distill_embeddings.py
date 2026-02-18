import typing as tp

import torch
import torch.nn.functional as F
from torch import nn

from ._types import LossDistillationBase, ModelOutput
from .dataflow import DFLossDistillationEmbeddings


class LossDistillationEmbeddings(LossDistillationBase):
    """
    Knowledge distillation loss using embeddings.
    Supports single embeddings, lists of embeddings, and multi-dimensional embeddings.
    """

    def __init__(
        self,
        loss_type: str = 'mse',
        normalize: bool = True,
        layer_weights: tp.Optional[tp.List[float]] = None,
    ):
        """
        Args:
            loss_type: Type of loss ('mse', 'cosine', 'l1')
            normalize: Whether to normalize embeddings before computing loss
            layer_weights: Weights for each layer if using list of embeddings
        """
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        self.layer_weights = layer_weights

        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'cosine':
            self.loss_fn = lambda x, y: 1 - F.cosine_similarity(x, y, dim=-1).mean()  # pylint: disable=not-callable
        elif loss_type == 'cosine_embedding':
            self.loss_fn = lambda x, y: F.cosine_embedding_loss(x, y, target=1)
        elif loss_type == 'cosine_log':
            self.loss_fn = lambda x, y: -torch.log(
                torch.clamp((1 + F.cosine_similarity(x, y, dim=-1)) / 2, min=1e-6)
            ).mean()

        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _compute_single_loss(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for a single embedding pair."""
        if self.normalize:
            student_emb = F.normalize(student_emb, p=2, dim=-1)
            teacher_emb = F.normalize(teacher_emb, p=2, dim=-1)

        return self.loss_fn(student_emb, teacher_emb)

    def forward(
        self,
        student_output: ModelOutput,
        teacher_output: ModelOutput,
        targets: tp.Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> DFLossDistillationEmbeddings:
        student_emb = student_output.embeddings
        teacher_emb = teacher_output.embeddings

        if student_emb is None or teacher_emb is None:
            raise ValueError("Embeddings are required for EmbeddingsDistillationLoss")

        # Handle list of embeddings (multi-layer)
        if isinstance(student_emb, list):
            if not isinstance(teacher_emb, list):
                raise ValueError("Both student and teacher must have list embeddings")

            if len(student_emb) != len(teacher_emb):
                raise ValueError("Number of embedding layers must match")

            weights = self.layer_weights or [1.0] * len(student_emb)
            total_loss = torch.tensor(0.0, device=student_emb[0].device)

            for _, (s_emb, t_emb, w) in enumerate(zip(student_emb, teacher_emb, weights)):
                total_loss = total_loss + w * self._compute_single_loss(s_emb, t_emb)

            loss = total_loss / sum(weights)
        else:
            loss = self._compute_single_loss(student_emb, teacher_emb)

        return DFLossDistillationEmbeddings(value=loss)
