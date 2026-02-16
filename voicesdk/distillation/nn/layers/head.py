import json
import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from voicesdk.distillation import ModelOutput


class HeadClassificationLDA(nn.Module):
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
        super().__init__()

        # Store speaker mapping as buffer (JSON string for serialization)
        if speaker_to_idx is not None:
            self._speaker_to_idx = speaker_to_idx
            self._idx_to_speaker = {v: k for k, v in speaker_to_idx.items()}
        else:
            self._speaker_to_idx = {}
            self._idx_to_speaker = {}

        # Initialize from provided arrays or create empty buffers
        if projection is not None:
            self.register_buffer("projection", torch.from_numpy(projection.astype(np.float32)))
            n_components, embedding_dim = projection.shape
        else:
            if embedding_dim is None or n_components is None:
                raise ValueError("Either projection or (embedding_dim, n_components) must be provided")
            self.register_buffer("projection", torch.zeros(n_components, embedding_dim))

        if class_means is not None:
            self.register_buffer("class_means", torch.from_numpy(class_means.astype(np.float32)))
            num_classes = class_means.shape[0]
        else:
            if num_classes is None or n_components is None:
                raise ValueError("Either class_means or (num_classes, n_components) must be provided")
            self.register_buffer("class_means", torch.zeros(num_classes, n_components))

        if global_mean is not None:
            self.register_buffer("global_mean", torch.from_numpy(global_mean.astype(np.float32)))
        else:
            if embedding_dim is None:
                embedding_dim = self.projection.shape[1]
            self.register_buffer("global_mean", torch.zeros(embedding_dim))

        # Store dimensions
        self.embedding_dim = self.projection.shape[1]
        self.n_components = self.projection.shape[0]
        self.num_classes = self.class_means.shape[0]

    @property
    def speaker_to_idx(self) -> tp.Dict[str, int]:
        return self._speaker_to_idx

    @property
    def idx_to_speaker(self) -> tp.Dict[int, str]:
        return self._idx_to_speaker

    def forward(self, embeddings: torch.Tensor) -> ModelOutput:
        """
        Compute logits from embeddings.

        Args:
            embeddings: Input embeddings (batch, embedding_dim)

        Returns:
            ModelOutput with logits and embeddings
        """
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        # Center embeddings
        centered = embeddings - self.global_mean.unsqueeze(0)

        # Project to LDA space: (batch, n_components)
        lda_features = torch.matmul(centered, self.projection.T)

        # Compute negative squared distances to class means as logits
        # Higher = closer = more likely
        diff = lda_features.unsqueeze(1) - self.class_means.unsqueeze(0)
        logits = -torch.sum(diff**2, dim=-1)

        return ModelOutput(logits=logits, embeddings=embeddings)

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


class HeadClassificationCentroids(nn.Module):
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
        super().__init__()

        self.temperature = temperature
        self.normalize_input = normalize_input

        # Store speaker mapping
        if speaker_to_idx is not None:
            self._speaker_to_idx = speaker_to_idx
            self._idx_to_speaker = {v: k for k, v in speaker_to_idx.items()}
        else:
            self._speaker_to_idx = {}
            self._idx_to_speaker = {}

        # Initialize centroids as Embedding layer
        if centroids is not None:
            num_classes, embedding_dim = centroids.shape
            self.centroids = nn.Embedding(num_classes, embedding_dim)
            self.centroids.weight.data = torch.from_numpy(centroids.astype(np.float32))
            self.centroids.weight.requires_grad = False  # Freeze by default
        else:
            if embedding_dim is None or num_classes is None:
                raise ValueError("Either centroids or (embedding_dim, num_classes) must be provided")
            self.centroids = nn.Embedding(num_classes, embedding_dim)
            self.centroids.weight.requires_grad = False

        self.embedding_dim = self.centroids.embedding_dim
        self.num_classes = self.centroids.num_embeddings

    @property
    def speaker_to_idx(self) -> tp.Dict[str, int]:
        return self._speaker_to_idx

    @property
    def idx_to_speaker(self) -> tp.Dict[int, str]:
        return self._idx_to_speaker

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
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

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
