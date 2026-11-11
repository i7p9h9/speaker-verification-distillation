import torch
import torch.nn.functional as F
from torch import nn

from ._types import AMSoftmaxOutput


class SoftmaxLoss(nn.Module):
    """Plain softmax classification head: logits = x W^T + b, cross-entropy.

    Drop-in replacement of ``AMSoftmaxLoss`` (same call signature, same ``weight`` (C, D) parameter and
    ``predict``/output container) but without embedding/weight normalisation, scale or margin.
    ``predict`` returns the raw logits. Its state dict has ``weight`` and ``bias`` (AM-Softmax has no bias).
    """

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.xavier_uniform_(self.weight)

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.linear(embeddings, self.weight, self.bias)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> AMSoftmaxOutput:
        logits = self.predict(embeddings)  # (B, C)
        loss_values = F.cross_entropy(logits, labels, reduction="none")
        cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1)).detach()  # monitoring only
        return AMSoftmaxOutput(loss=loss_values.mean(), loss_values=loss_values, cosine=cosine, logits=logits)
