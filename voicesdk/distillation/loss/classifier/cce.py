import typing as tp

import torch
from torch import nn

from voicesdk.distillation.loss import CCELossOutput, ModelOutput


class LossCCE(nn.Module):
    """Categorical cross-entropy loss with optional embedding pass-through.

    Args:
        label_smoothing: Label smoothing factor in ``[0, 1)``.
        ignore_index:    Class index to ignore (e.g. padding token).
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self._ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        output: ModelOutput,
        targets: torch.Tensor,
    ) -> CCELossOutput:
        """Compute cross-entropy loss.

        Args:
            output:  Model output containing ``logits`` of shape
                     ``[N, C]`` or ``[N, T, C]``.
            targets: Ground-truth class indices of shape ``[N]``
                     or ``[N, T]``.

        Returns:
            :class:`CCELossOutput` with ``value`` and optional ``embedding``.
        """
        logits = output.logits

        # Flatten sequence dimension if present: [N, T, C] -> [N*T, C]
        if logits.dim() == 3:
            n, t, c = logits.shape
            logits = logits.reshape(n * t, c)
            targets = targets.reshape(n * t)

        loss = self._ce(logits, targets)

        # Derive a mean-pooled embedding when available
        embedding: tp.Optional[torch.Tensor] = None
        if output.embeddings is not None:
            embs = output.embeddings
            if isinstance(embs, list):
                embs = embs[-1]  # use last layer by convention
            if embs.dim() == 3:
                embs = embs.mean(dim=1)
            embedding = embs.detach()

        return CCELossOutput(value=loss, embedding=embedding)
