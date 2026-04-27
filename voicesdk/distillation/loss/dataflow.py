from dataclasses import dataclass, field

import torch

from ._types import DFLossBase


# =============================================================================
# Loss Dataclasses
# =============================================================================
@dataclass
class DFLossDistillationLogits(DFLossBase):
    """Loss computed from logits distillation only."""


@dataclass
class DFLossDistillationEmbeddings(DFLossBase):
    """Loss computed from embeddings distillation only."""


@dataclass
class DFLossDistillationMixed(DFLossBase):
    """
    Loss computed from both logits and embeddings distillation.

    Attributes:
        value: Combined total loss
        value_logits: Loss contribution from logits
        value_embeddings: Loss contribution from embeddings
    """
    value_logits: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    value_embeddings: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))


@dataclass
class DFLossDistillationWithTarget(DFLossBase):
    """
    Loss with logits, embeddings, and target cross-entropy.

    Attributes:
        value: Combined total loss
        value_logits: Loss contribution from logits distillation
        value_embeddings: Loss contribution from embeddings distillation
        value_target: Loss contribution from target (ground truth) cross-entropy
    """
    value_logits: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    value_embeddings: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    value_target: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
