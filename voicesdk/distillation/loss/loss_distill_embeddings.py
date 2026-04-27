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
            self.loss_fn = lambda x, y: F.mse_loss(x, y, reduction='none').mean(dim=tuple(range(1, x.ndim)))
        elif loss_type == 'l1':
            self.loss_fn = lambda x, y: F.l1_loss(x, y, reduction='none').mean(dim=tuple(range(1, x.ndim)))
        elif loss_type == 'cosine':
            self.loss_fn = lambda x, y: 1 - F.cosine_similarity(x, y, dim=-1)  # pylint: disable=not-callable
        elif loss_type == 'cosine_embedding':
            # cosine_embedding_loss with reduction='none' returns per-sample values
            self.loss_fn = lambda x, y: F.cosine_embedding_loss(
                x, y,
                target=torch.ones(x.shape[0], device=x.device),
                reduction='none',
            )
        elif loss_type == 'cosine_log':
            self.loss_fn = lambda x, y: -torch.log(
                torch.clamp((1 + F.cosine_similarity(x, y, dim=-1)) / 2, min=1e-6)
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _compute_single_loss(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample loss for a single embedding pair.

        Returns:
            Tensor of shape [B] with per-sample loss values.
        """
        if self.normalize:
            student_emb = F.normalize(student_emb, p=2, dim=-1)
            teacher_emb = F.normalize(teacher_emb, p=2, dim=-1)

        # loss_fn returns shape [B]
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
            weights_sum = sum(weights)

            # per_layer_losses: list of [B] tensors, one per layer
            per_layer_losses: tp.List[torch.Tensor] = [
                self._compute_single_loss(s_emb, t_emb)
                for s_emb, t_emb in zip(student_emb, teacher_emb)
            ]

            # Weighted sum across layers -> [B]
            loss_values = sum(
                w * layer_loss
                for w, layer_loss in zip(weights, per_layer_losses)
            ) / weights_sum  # type: ignore[assignment]

            value = loss_values.mean()

        else:
            # loss_values: [B], value: scalar
            loss_values = self._compute_single_loss(student_emb, teacher_emb)
            value = loss_values.mean()

        return DFLossDistillationEmbeddings(value=value, loss_batch=loss_values)
