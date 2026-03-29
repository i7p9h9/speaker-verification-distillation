import torch
import torch.nn as nn
import torch.nn.functional as F

from ._types import AMSoftmaxOutput


class AMSoftmaxLoss(nn.Module):
    """Additive Margin Softmax (AM-Softmax / CosFace) loss.

    Args:
        embedding_dim: Dimensionality of input embeddings.
        num_classes:   Number of target classes.
        scale:         Scaling factor (s). Typical values: 30–64.
        margin:        Additive cosine margin (m). Typical values: 0.2–0.4.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.35,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin

        # Weight matrix — one unit-norm prototype per class
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _cosine_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between embeddings and class prototypes.

        Args:
            embeddings: (B, D) — NOT required to be unit-norm on input.

        Returns:
            cosine: (B, C) — values in [-1, 1].
        """
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # (B, D)
        weight_norm = F.normalize(self.weight, p=2, dim=1)      # (C, D)
        return F.linear(embeddings_norm, weight_norm)            # (B, C)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> AMSoftmaxOutput:
        """Compute AM-Softmax loss.

        Args:
            embeddings: (B, D) float tensor — raw speaker/sample embeddings.
            labels:     (B,)   long tensor  — ground-truth class indices.

        Returns:
            AMSoftmaxOutput with scalar `loss` and per-sample `loss_values`.
        """
        cosine = self._cosine_similarity(embeddings)  # (B, C)

        # Subtract margin only from the ground-truth class score
        # cosine_m[i, labels[i]] = cosine[i, labels[i]] - margin
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        cosine_with_margin = cosine - one_hot * self.margin

        # Scale and compute cross-entropy per sample
        logits = cosine_with_margin * self.scale            # (B, C)
        loss_values = F.cross_entropy(logits, labels, reduction="none")  # (B,)

        return AMSoftmaxOutput(
            loss=loss_values.mean(),
            loss_values=loss_values,
        )
