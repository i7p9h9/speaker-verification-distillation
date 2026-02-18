import json
import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from voicesdk.distillation import ModelOutput


class HeadBase(nn.Module, ABC):
    """
    Abstract base class for classification head layers.

    Provides common functionality for speaker mapping management
    and defines the interface for classification heads.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        speaker_to_idx: tp.Optional[tp.Dict[str, int]] = None,
    ):
        """
        Args:
            embedding_dim: Input embedding dimension
            num_classes: Number of classes
            speaker_to_idx: Mapping from speaker ID to class index
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Store speaker mapping
        if speaker_to_idx is not None:
            self._speaker_to_idx = speaker_to_idx
            self._idx_to_speaker = {v: k for k, v in speaker_to_idx.items()}
        else:
            self._speaker_to_idx = {}
            self._idx_to_speaker = {}

    @property
    def speaker_to_idx(self) -> tp.Dict[str, int]:
        """Mapping from speaker ID to class index."""
        return self._speaker_to_idx

    @property
    def idx_to_speaker(self) -> tp.Dict[int, str]:
        """Mapping from class index to speaker ID."""
        return self._idx_to_speaker

    @abstractmethod
    def forward(self, embeddings: torch.Tensor) -> ModelOutput:
        """
        Compute logits from embeddings.

        Args:
            embeddings: Input embeddings (batch, embedding_dim)

        Returns:
            ModelOutput with logits and embeddings
        """
        pass

    def save_speaker_mapping(self, path: str) -> None:
        """Save speaker mapping to JSON file."""
        with open(path, "w") as f:
            json.dump(
                {
                    "speaker_to_idx": self._speaker_to_idx,
                    "idx_to_speaker": {str(k): v for k, v in self._idx_to_speaker.items()},
                },
                f,
                indent=2,
            )

    def load_speaker_mapping(self, path: str) -> None:
        """Load speaker mapping from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        self._speaker_to_idx = data["speaker_to_idx"]
        self._idx_to_speaker = {int(k): v for k, v in data["idx_to_speaker"].items()}

    def _ensure_batch_dim(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Ensure embeddings have batch dimension."""
        if embeddings.dim() == 1:
            return embeddings.unsqueeze(0)
        return embeddings


class HeadModelWrapper(nn.Module):
    """
    Wrapper that combines a base embedding model with a classification head.

    By default, runs in inference mode with no gradients and both modules in eval mode.
    Training mode must be explicitly enabled with clear notification.
    """

    def __init__(
        self,
        model_base: nn.Module,
        head: HeadBase,
        enable_grad: bool = False,
        enable_train: bool = False,
    ):
        """
        Args:
            model_base: Base model that produces embeddings
            head: Classification head (HeadBase subclass)
            enable_grad: If True, enables gradient computation. 
                         WARNING: Must be explicitly set to True for training.
            enable_train: If True, sets both modules to train mode.
                          WARNING: Must be explicitly set to True for training.
        """
        super().__init__()

        self.model_base = model_base
        self.head = head
        self._grad_enabled = enable_grad
        self._train_enabled = enable_train

        # Notify if training/grad is enabled
        if enable_grad:
            print(
                "[HeadModelWrapper] WARNING: Gradient computation is ENABLED. "
                "This is intended for training only."
            )
        if enable_train:
            print(
                "[HeadModelWrapper] WARNING: Training mode is ENABLED. "
                "Both model_base and head are set to train()."
            )

        # Apply default mode
        self._apply_mode()

    def _apply_mode(self) -> None:
        """Apply eval/train mode based on configuration."""
        if self._train_enabled:
            self.model_base.train()
            self.head.train()
        else:
            self.model_base.eval()
            self.head.eval()

    def set_inference_mode(self) -> None:
        """Set wrapper to inference mode (no grad, eval)."""
        self._grad_enabled = False
        self._train_enabled = False
        self._apply_mode()
        print("[HeadModelWrapper] Switched to INFERENCE mode (no grad, eval).")

    def set_training_mode(self, enable_grad: bool = True) -> None:
        """
        Set wrapper to training mode.

        Args:
            enable_grad: Whether to enable gradient computation
        """
        self._train_enabled = True
        self._grad_enabled = enable_grad
        self._apply_mode()
        print(
            f"[HeadModelWrapper] WARNING: Switched to TRAINING mode "
            f"(grad={'ENABLED' if enable_grad else 'DISABLED'}, train=ENABLED)."
        )

    def forward(self, *args, **kwargs) -> ModelOutput:
        """
        Forward pass through base model and head.

        Args:
            *args: Positional arguments passed to model_base
            **kwargs: Keyword arguments passed to model_base

        Returns:
            ModelOutput with logits and embeddings
        """
        if self._grad_enabled:
            embeddings = self.model_base(*args, **kwargs)
            return self.head(embeddings)
        else:
            with torch.no_grad():
                embeddings = self.model_base(*args, **kwargs)
                return self.head(embeddings)

    def train(self, mode: bool = True) -> "HeadModelWrapper":
        """
        Override train() to require explicit configuration.

        Use set_training_mode() instead for explicit control.
        """
        if mode and not self._train_enabled:
            raise RuntimeError(
                "Cannot enable training mode via train(). "
                "Use set_training_mode() for explicit control with proper notification."
            )
        return super().train(mode)

    def get_embeddings(self, *args, **kwargs) -> torch.Tensor:
        """
        Get embeddings from base model without classification.

        Args:
            *args: Positional arguments passed to model_base
            **kwargs: Keyword arguments passed to model_base

        Returns:
            Embeddings tensor from model_base
        """
        if self._grad_enabled:
            return self.model_base(*args, **kwargs)
        else:
            with torch.no_grad():
                return self.model_base(*args, **kwargs)

    @property
    def is_training_mode(self) -> bool:
        """Check if wrapper is in training mode."""
        return self._train_enabled

    @property
    def is_grad_enabled(self) -> bool:
        """Check if gradient computation is enabled."""
        return self._grad_enabled


class HeadClassificationLDA(HeadBase):
    """
    LDA-based classification head layer.

    Computes logits based on negative squared distance to class means in LDA space.

    Can be initialized with pre-computed LDA parameters or empty (load state_dict later).
    """

    def __init__(
        self,
        embedding_dim: tp.Optional[int] = None,
        num_classes: tp.Optional[int] = None,
        n_components: tp.Optional[int] = None,
        projection: tp.Optional[np.ndarray] = None,
        class_means: tp.Optional[np.ndarray] = None,
        global_mean: tp.Optional[np.ndarray] = None,
        speaker_to_idx: tp.Optional[tp.Dict[str, int]] = None,
    ):
        """
        Args:
            embedding_dim: Input embedding dimension (required if projection not provided)
            num_classes: Number of classes (required if class_means not provided)
            n_components: LDA components (required if projection not provided)
            projection: Pre-computed LDA projection matrix (n_components, embedding_dim)
            class_means: Pre-computed class means in LDA space (num_classes, n_components)
            global_mean: Pre-computed global mean (embedding_dim,)
            speaker_to_idx: Mapping from speaker ID to class index
        """
        # Infer dimensions from provided arrays
        if projection is not None:
            n_components_inferred, embedding_dim_inferred = projection.shape
            n_components = n_components or n_components_inferred
            embedding_dim = embedding_dim or embedding_dim_inferred

        if class_means is not None:
            num_classes_inferred = class_means.shape[0]
            num_classes = num_classes or num_classes_inferred

        # Validate required dimensions
        if embedding_dim is None or n_components is None:
            raise ValueError("Either projection or (embedding_dim, n_components) must be provided")
        if num_classes is None:
            raise ValueError("Either class_means or num_classes must be provided")

        super().__init__(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            speaker_to_idx=speaker_to_idx,
        )

        self.n_components = n_components

        # Initialize projection matrix
        if projection is not None:
            self.register_buffer("projection", torch.from_numpy(projection.astype(np.float32)))
        else:
            self.register_buffer("projection", torch.zeros(n_components, embedding_dim))

        # Initialize class means
        if class_means is not None:
            self.register_buffer("class_means", torch.from_numpy(class_means.astype(np.float32)))
        else:
            self.register_buffer("class_means", torch.zeros(num_classes, n_components))

        # Initialize global mean
        if global_mean is not None:
            self.register_buffer("global_mean", torch.from_numpy(global_mean.astype(np.float32)))
        else:
            self.register_buffer("global_mean", torch.zeros(embedding_dim))

    def forward(self, embeddings: torch.Tensor) -> ModelOutput:
        """
        Compute logits from embeddings.

        Args:
            embeddings: Input embeddings (batch, embedding_dim)

        Returns:
            ModelOutput with logits and embeddings
        """
        embeddings = self._ensure_batch_dim(embeddings)

        # Center embeddings
        centered = embeddings - self.global_mean.unsqueeze(0)

        # Project to LDA space: (batch, n_components)
        lda_features = torch.matmul(centered, self.projection.T)

        # Compute negative squared distances to class means as logits
        # Higher = closer = more likely
        diff = lda_features.unsqueeze(1) - self.class_means.unsqueeze(0)
        logits = -torch.sum(diff**2, dim=-1)

        return ModelOutput(logits=logits, embeddings=embeddings)


class HeadClassificationCentroids(HeadBase):
    """
    Centroid-based classification head layer.

    Uses cosine similarity to class centroids as logits.
    Centroids are stored as nn.Embedding for efficient lookup and serialization.

    Can be initialized with pre-computed centroids or empty (load state_dict later).
    """

    def __init__(
        self,
        embedding_dim: tp.Optional[int] = None,
        num_classes: tp.Optional[int] = None,
        centroids: tp.Optional[np.ndarray] = None,
        speaker_to_idx: tp.Optional[tp.Dict[str, int]] = None,
        temperature: float = 1.0,
        normalize_input: bool = True,
    ):
        """
        Args:
            embedding_dim: Embedding dimension (required if centroids not provided)
            num_classes: Number of classes (required if centroids not provided)
            centroids: Pre-computed centroids (num_classes, embedding_dim)
            speaker_to_idx: Mapping from speaker ID to class index
            temperature: Temperature for scaling cosine similarities
            normalize_input: Whether to normalize input embeddings
        """
        # Infer dimensions from provided arrays
        if centroids is not None:
            num_classes_inferred, embedding_dim_inferred = centroids.shape
            num_classes = num_classes or num_classes_inferred
            embedding_dim = embedding_dim or embedding_dim_inferred

        # Validate required dimensions
        if embedding_dim is None or num_classes is None:
            raise ValueError("Either centroids or (embedding_dim, num_classes) must be provided")

        super().__init__(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            speaker_to_idx=speaker_to_idx,
        )

        self.temperature = temperature
        self.normalize_input = normalize_input

        # Initialize centroids as Embedding layer
        self.centroids = nn.Embedding(num_classes, embedding_dim)
        self.centroids.weight.requires_grad = False  # Freeze by default

        if centroids is not None:
            self.centroids.weight.data = torch.from_numpy(centroids.astype(np.float32))

    @property
    def weight(self) -> torch.Tensor:
        """Get centroids weight matrix for compatibility."""
        return self.centroids.weight

    def forward(self, embeddings: torch.Tensor) -> ModelOutput:
        """
        Compute logits from embeddings using cosine similarity to centroids.

        Args:
            embeddings: Input embeddings (batch, embedding_dim)

        Returns:
            ModelOutput with logits and embeddings
        """
        embeddings = self._ensure_batch_dim(embeddings)

        # Normalize input embeddings
        if self.normalize_input:
            embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        else:
            embeddings_norm = embeddings

        # Normalize centroids
        centroids_norm = F.normalize(self.centroids.weight, p=2, dim=-1)

        # Cosine similarity: (batch, num_classes)
        logits = torch.matmul(embeddings_norm, centroids_norm.T) / self.temperature

        return ModelOutput(logits=logits, embeddings=embeddings)

    def get_centroid(self, speaker_id: str) -> torch.Tensor:
        """Get centroid for a specific speaker."""
        idx = self._speaker_to_idx[speaker_id]
        return self.centroids.weight[idx]
