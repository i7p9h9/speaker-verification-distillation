import typing as tp

import numpy as np
import torch

from torch.utils import nn
from torch.utils.data import DataLoader, Dataset

from voicesdk.distillation.validation import ValidatorBase
from .utils.extract_embeddings import extract_embeddings, extract_embeddings_with_aggregation


class EmbeddingValidatorBase(ValidatorBase):
    """
    Base class for validators that work with embeddings.
    
    Provides common functionality for embedding extraction and trial evaluation.
    """

    def __init__(
        self,
        name: str,
        dataset: tp.Optional[Dataset] = None,
        dataloader: tp.Optional[DataLoader] = None,
        use_weighted_aggregation: bool = False,
        normalize_embeddings: bool = True,
        **kwargs,
    ):
        """
        Args:
            name: Validator name
            dataset: Dataset for validation (used with aggregation)
            dataloader: DataLoader for validation (used for batch extraction)
            use_weighted_aggregation: If True, use weighted segment aggregation
            normalize_embeddings: If True, normalize embeddings
            **kwargs: Additional arguments for ValidatorBase
        """
        super().__init__(name=name, **kwargs)
        self.dataset = dataset
        self.dataloader = dataloader
        self.use_weighted_aggregation = use_weighted_aggregation
        self.normalize_embeddings = normalize_embeddings

    def _extract_embeddings(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> np.ndarray:
        """Extract embeddings using configured method."""
        if self.use_weighted_aggregation and self.dataset is not None:
            return extract_embeddings_with_aggregation(
                model=model,
                dataset=self.dataset,
                device=device,
                normalize=self.normalize_embeddings,
            )
        elif self.dataloader is not None:
            return extract_embeddings(
                model=model,
                dataloader=self.dataloader,
                device=device,
                normalize=self.normalize_embeddings,
            )
        else:
            raise ValueError("Either dataset or dataloader must be provided")
