from dataclasses import dataclass

import torch


@dataclass
class AMSoftmaxOutput:
    """Output container for AMSoftmax loss computation."""
    loss: torch.Tensor        # scalar mean loss over the batch
    loss_values: torch.Tensor # per-sample loss values, shape (batch_size,)