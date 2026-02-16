"""
Teacher Model Preparation for Distillation

This script prepares a teacher model by computing classification heads
from embeddings extracted on the training dataset (VoxCeleb2).

Two methods are supported:
1. LDA-based: Computes Linear Discriminant Analysis projection as the output layer
2. Centroid-based: Uses weighted average embeddings per speaker as class centers

The process:
1. Extract embeddings for all audio files
2. Aggregate embeddings per utterance (weighted by duration)
3. Compute speaker representations from utterances
4. Build classification head (LDA or centroids)
"""

import json
import os
import typing as tp
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from voicesdk.distillation import ModelOutput

# Import your dataset
from voicesdk.distillation.data.pipe import WavDataset
from voicesdk.nn.arch import ResNetTF

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TeacherPrepConfig:
    """Configuration for teacher model preparation."""

    # Data paths
    data_dir: str = "data/vox2"
    output_dir: str = "teacher_prepared"

    # Model settings
    embedding_dim: int = 256

    # Audio settings
    sample_rate: int = 16000
    norm_type: str = "std"

    # Processing
    batch_size: int = 1  # Since length_segment_ms=None, each file is one sample
    num_workers: int = 4
    device: str = "cuda"

    # Method: 'lda' or 'centroid'
    method: str = "lda"

    # LDA settings
    lda_n_components: tp.Optional[int] = None  # None = num_classes - 1
    lda_solver: str = "svd"  # 'svd', 'lsqr', 'eigen'

    # Centroid settings
    centroid_normalize: bool = True

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Model Placeholder
# =============================================================================


def read_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    return dict(hparams)


def get_model() -> nn.Module:
    """
    Load your teacher model here.

    Returns:
        Teacher model that outputs embeddings

    Example implementation:
        model = YourModel.load_from_checkpoint('teacher.pt')
        model.eval()
        return model
    """
    cfg = read_yaml(os.path.join(__file__, "data/cfg-models/rn100_v016_flr_vox4_v2.yaml"))
    state_dict_teacher = torch.load("data/ckpt/rn100_v016_flr_vox4_v2/model.pt", map_location=torch.device("cpu"))

    model_teacher = ResNetTF(**cfg["model_args"])
    model_teacher.eval()
    model_teacher.load_state_dict(state_dict_teacher, strict=True)

    return model_teacher


# =============================================================================
# Data Collection
# =============================================================================


@dataclass
class FileInfo:
    """Information about a single audio file."""

    filepath: str
    speaker_id: str
    utterance_id: str
    duration: float = 0.0


@dataclass
class UtteranceEmbedding:
    """Aggregated embedding for an utterance."""

    speaker_id: str
    utterance_id: str
    embedding: np.ndarray
    total_duration: float
    num_files: int


def discover_voxceleb2_files(data_dir: str) -> tp.List[FileInfo]:
    """
    Discover all wav files in VoxCeleb2 directory structure.

    Expected structure: data_dir/speaker_id/utterance_id/*.wav

    Args:
        data_dir: Root directory containing VoxCeleb2 data

    Returns:
        List of FileInfo for each wav file
    """
    files = []
    data_path = Path(data_dir)

    print(f"Discovering files in {data_dir}...")

    for speaker_dir in tqdm(sorted(data_path.iterdir()), desc="Scanning speakers"):
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name

        for utterance_dir in speaker_dir.iterdir():
            if not utterance_dir.is_dir():
                continue

            utterance_id = utterance_dir.name

            for wav_file in utterance_dir.glob("*.wav"):
                files.append(
                    FileInfo(
                        filepath=str(wav_file),
                        speaker_id=speaker_id,
                        utterance_id=f"{speaker_id}/{utterance_id}",
                    )
                )

    print(f"Found {len(files)} files from {len(set(f.speaker_id for f in files))} speakers")
    return files


def group_files_by_utterance(files: tp.List[FileInfo]) -> tp.Dict[str, tp.List[FileInfo]]:
    """Group files by utterance ID."""
    utterance_groups = defaultdict(list)
    for file_info in files:
        utterance_groups[file_info.utterance_id].append(file_info)
    return dict(utterance_groups)


def group_files_by_speaker(files: tp.List[FileInfo]) -> tp.Dict[str, tp.List[FileInfo]]:
    """Group files by speaker ID."""
    speaker_groups = defaultdict(list)
    for file_info in files:
        speaker_groups[file_info.speaker_id].append(file_info)
    return dict(speaker_groups)


# =============================================================================
# Embedding Extraction
# =============================================================================


@dataclass
class ExtractedEmbedding:
    """Single extracted embedding with metadata."""

    filepath: str
    speaker_id: str
    utterance_id: str
    embedding: np.ndarray
    duration: float


def extract_embeddings_for_files(
    model: nn.Module,
    files: tp.List[FileInfo],
    config: TeacherPrepConfig,
) -> tp.List[ExtractedEmbedding]:
    """
    Extract embeddings for all files.

    Uses length_segment_ms=None for full-length extraction (best quality).

    Args:
        model: Teacher model
        files: List of file info
        config: Configuration

    Returns:
        List of extracted embeddings with metadata
    """
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    # Create dataset with full-length segments
    filepaths = [f.filepath for f in files]
    dataset = WavDataset(
        wav_scp=filepaths,
        norm_type=config.norm_type,
        length_segment_ms=None,  # Full length for best quality
        segments_step_ms=None,
        sample_rate=config.sample_rate,
    )

    embeddings = []

    print("Extracting embeddings...")
    for idx in tqdm(range(len(dataset)), desc="Extracting"):
        sample = dataset[idx]
        file_info = files[idx]

        # Get segments tensor
        # When length_segment_ms=None, segments shape is [1, audio_length]
        segments = sample.segments
        if isinstance(segments, np.ndarray):
            segments = torch.from_numpy(segments)

        segments = segments.to(device)

        # Handle batch dimension
        if segments.dim() == 1:
            segments = segments.unsqueeze(0)

        with torch.no_grad():
            output = model(segments)

            # Get embeddings
            if hasattr(output, "embeddings") and output.embeddings is not None:
                if isinstance(output.embeddings, list):
                    emb = output.embeddings[-1]
                else:
                    emb = output.embeddings
            elif hasattr(output, "logits"):
                emb = output.logits
            else:
                emb = output

            # Average over segments if needed
            if emb.dim() > 2:
                emb = emb.mean(dim=1)
            elif emb.dim() == 2 and emb.size(0) > 1:
                # Multiple segments - weighted average by duration
                weights = torch.tensor(sample.segments_weights, device=device, dtype=emb.dtype)
                emb = (emb * weights.unsqueeze(1)).sum(dim=0, keepdim=True)

            emb = emb.squeeze(0).cpu().numpy()

        embeddings.append(
            ExtractedEmbedding(
                filepath=file_info.filepath,
                speaker_id=file_info.speaker_id,
                utterance_id=file_info.utterance_id,
                embedding=emb,
                duration=sample.total_duration,
            )
        )

    return embeddings


def aggregate_embeddings_by_utterance(
    embeddings: tp.List[ExtractedEmbedding],
) -> tp.List[UtteranceEmbedding]:
    """
    Aggregate embeddings per utterance using weighted average.

    Weights are based on audio duration.

    Args:
        embeddings: List of extracted embeddings

    Returns:
        List of aggregated utterance embeddings
    """
    # Group by utterance
    utterance_groups: tp.Dict[str, tp.List[ExtractedEmbedding]] = defaultdict(list)
    for emb in embeddings:
        utterance_groups[emb.utterance_id].append(emb)

    aggregated = []

    print("Aggregating embeddings by utterance...")
    for utterance_id, group in tqdm(utterance_groups.items(), desc="Aggregating"):
        # Get speaker ID (same for all in group)
        speaker_id = group[0].speaker_id

        # Weighted average by duration
        total_duration = sum(e.duration for e in group)

        if total_duration > 0:
            weighted_sum = np.zeros_like(group[0].embedding)
            for e in group:
                weight = e.duration / total_duration
                weighted_sum += weight * e.embedding
            aggregated_emb = weighted_sum
        else:
            # Fallback to simple average
            aggregated_emb = np.mean([e.embedding for e in group], axis=0)

        aggregated.append(
            UtteranceEmbedding(
                speaker_id=speaker_id,
                utterance_id=utterance_id,
                embedding=aggregated_emb,
                total_duration=total_duration,
                num_files=len(group),
            )
        )

    print(f"Aggregated to {len(aggregated)} utterances")
    return aggregated


# =============================================================================
# Classification Head: LDA
# =============================================================================


@dataclass
class LDAClassificationHead:
    """LDA-based classification head."""

    # LDA transformation matrix: (n_components, embedding_dim)
    projection: np.ndarray

    # Class means in LDA space: (num_classes, n_components)
    class_means: np.ndarray

    # Speaker ID to class index mapping
    speaker_to_idx: tp.Dict[str, int]

    # Class index to speaker ID mapping
    idx_to_speaker: tp.Dict[int, str]

    # Global mean
    global_mean: np.ndarray

    def save(self, path: str) -> None:
        """Save LDA head to file."""
        np.savez(
            path,
            projection=self.projection,
            class_means=self.class_means,
            global_mean=self.global_mean,
            speaker_to_idx=json.dumps(self.speaker_to_idx),
            idx_to_speaker=json.dumps({int(k): v for k, v in self.idx_to_speaker.items()}),
        )

    @classmethod
    def load(cls, path: str) -> "LDAClassificationHead":
        """Load LDA head from file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            projection=data["projection"],
            class_means=data["class_means"],
            global_mean=data["global_mean"],
            speaker_to_idx=json.loads(str(data["speaker_to_idx"])),
            idx_to_speaker={int(k): v for k, v in json.loads(str(data["idx_to_speaker"])).items()},
        )


def compute_lda_head(
    utterance_embeddings: tp.List[UtteranceEmbedding],
    config: TeacherPrepConfig,
) -> LDAClassificationHead:
    """
    Compute LDA-based classification head.

    Args:
        utterance_embeddings: Aggregated utterance embeddings
        config: Configuration

    Returns:
        LDA classification head
    """
    # Prepare data
    X = np.array([e.embedding for e in utterance_embeddings])
    speakers = [e.speaker_id for e in utterance_embeddings]

    # Create speaker to index mapping
    unique_speakers = sorted(set(speakers))
    speaker_to_idx = {s: i for i, s in enumerate(unique_speakers)}
    idx_to_speaker = {i: s for s, i in speaker_to_idx.items()}

    y = np.array([speaker_to_idx[s] for s in speakers])

    print(f"Computing LDA with {len(unique_speakers)} classes and {len(X)} samples...")

    # Compute LDA
    n_components = config.lda_n_components
    if n_components is None:
        n_components = min(len(unique_speakers) - 1, X.shape[1])

    lda = LinearDiscriminantAnalysis(
        n_components=n_components,
        solver=config.lda_solver,
    )

    lda.fit(X, y)

    # Get projection matrix (LDA scalings)
    # Shape: (embedding_dim, n_components)
    projection = lda.scalings_[:, :n_components].T  # (n_components, embedding_dim)

    # Compute class means in LDA space
    X_lda = lda.transform(X)
    class_means = np.zeros((len(unique_speakers), n_components))

    for class_idx in range(len(unique_speakers)):
        mask = y == class_idx
        if mask.sum() > 0:
            class_means[class_idx] = X_lda[mask].mean(axis=0)

    print(f"LDA projection shape: {projection.shape}")
    print(f"Class means shape: {class_means.shape}")

    return LDAClassificationHead(
        projection=projection,
        class_means=class_means,
        speaker_to_idx=speaker_to_idx,
        idx_to_speaker=idx_to_speaker,
        global_mean=lda.xbar_,
    )


# =============================================================================
# Classification Head: Centroids
# =============================================================================


@dataclass
class CentroidClassificationHead:
    """Centroid-based classification head."""

    # Class centroids: (num_classes, embedding_dim)
    centroids: np.ndarray

    # Speaker ID to class index mapping
    speaker_to_idx: tp.Dict[str, int]

    # Class index to speaker ID mapping
    idx_to_speaker: tp.Dict[int, str]

    # Whether centroids are normalized
    normalized: bool

    def save(self, path: str) -> None:
        """Save centroid head to file."""
        np.savez(
            path,
            centroids=self.centroids,
            normalized=self.normalized,
            speaker_to_idx=json.dumps(self.speaker_to_idx),
            idx_to_speaker=json.dumps({int(k): v for k, v in self.idx_to_speaker.items()}),
        )

    @classmethod
    def load(cls, path: str) -> "CentroidClassificationHead":
        """Load centroid head from file."""
        data = np.load(path, allow_pickle=True)
        return cls(
            centroids=data["centroids"],
            normalized=bool(data["normalized"]),
            speaker_to_idx=json.loads(str(data["speaker_to_idx"])),
            idx_to_speaker={int(k): v for k, v in json.loads(str(data["idx_to_speaker"])).items()},
        )


def compute_centroid_head(
    utterance_embeddings: tp.List[UtteranceEmbedding],
    config: TeacherPrepConfig,
) -> CentroidClassificationHead:
    """
    Compute centroid-based classification head.

    Centroids are computed as weighted average of utterance embeddings,
    where weights are based on utterance duration.

    Args:
        utterance_embeddings: Aggregated utterance embeddings
        config: Configuration

    Returns:
        Centroid classification head
    """
    # Group by speaker
    speaker_utterances: tp.Dict[str, tp.List[UtteranceEmbedding]] = defaultdict(list)
    for utt in utterance_embeddings:
        speaker_utterances[utt.speaker_id].append(utt)

    # Create speaker to index mapping
    unique_speakers = sorted(speaker_utterances.keys())
    speaker_to_idx = {s: i for i, s in enumerate(unique_speakers)}
    idx_to_speaker = {i: s for s, i in speaker_to_idx.items()}

    embedding_dim = utterance_embeddings[0].embedding.shape[0]
    centroids = np.zeros((len(unique_speakers), embedding_dim))

    print(f"Computing centroids for {len(unique_speakers)} speakers...")

    for speaker_id, utterances in tqdm(speaker_utterances.items(), desc="Computing centroids"):
        class_idx = speaker_to_idx[speaker_id]

        # Weighted average by utterance duration
        total_duration = sum(u.total_duration for u in utterances)

        if total_duration > 0:
            weighted_sum = np.zeros(embedding_dim)
            for u in utterances:
                weight = u.total_duration / total_duration
                weighted_sum += weight * u.embedding
            centroid = weighted_sum
        else:
            # Fallback to simple average
            centroid = np.mean([u.embedding for u in utterances], axis=0)

        if config.centroid_normalize:
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        centroids[class_idx] = centroid

    print(f"Centroids shape: {centroids.shape}")

    return CentroidClassificationHead(
        centroids=centroids,
        speaker_to_idx=speaker_to_idx,
        idx_to_speaker=idx_to_speaker,
        normalized=config.centroid_normalize,
    )


# =============================================================================
# Teacher Wrapper Models
# =============================================================================


class LDATeacherWrapper(nn.Module):
    """
    Wrapper that adds LDA classification head to teacher model.
    """

    def __init__(
        self,
        base_model: nn.Module,
        lda_head: LDAClassificationHead,
        return_embeddings: bool = True,
    ):
        """
        Args:
            base_model: Base teacher model that outputs embeddings
            lda_head: LDA classification head
            return_embeddings: Whether to return embeddings in output
        """
        super().__init__()
        self.base_model = base_model
        self.return_embeddings = return_embeddings

        # Convert LDA parameters to torch
        self.register_buffer("projection", torch.from_numpy(lda_head.projection.astype(np.float32)))
        self.register_buffer("class_means", torch.from_numpy(lda_head.class_means.astype(np.float32)))
        self.register_buffer("global_mean", torch.from_numpy(lda_head.global_mean.astype(np.float32)))

        self.speaker_to_idx = lda_head.speaker_to_idx
        self.idx_to_speaker = lda_head.idx_to_speaker
        self.num_classes = len(self.speaker_to_idx)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # Get base embeddings
        with torch.no_grad():
            base_output = self.base_model(x)

            if hasattr(base_output, "embeddings") and base_output.embeddings is not None:
                if isinstance(base_output.embeddings, list):
                    embeddings = base_output.embeddings[-1]
                else:
                    embeddings = base_output.embeddings
            elif hasattr(base_output, "logits"):
                embeddings = base_output.logits
            else:
                embeddings = base_output

        # Ensure correct shape
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        # Center embeddings
        centered = embeddings - self.global_mean.unsqueeze(0)

        # Project to LDA space
        # projection: (n_components, embedding_dim)
        # centered: (batch, embedding_dim)
        lda_features = torch.matmul(centered, self.projection.T)  # (batch, n_components)

        # Compute distances to class means
        # class_means: (num_classes, n_components)
        # lda_features: (batch, n_components)
        diff = lda_features.unsqueeze(1) - self.class_means.unsqueeze(0)  # (batch, num_classes, n_components)
        distances = -torch.sum(diff**2, dim=-1)  # (batch, num_classes) - negative squared distance as logits

        logits = distances  # Higher = closer = more likely

        return ModelOutput(
            logits=logits,
            embeddings=embeddings if self.return_embeddings else None,
        )


class CentroidTeacherWrapper(nn.Module):
    """
    Wrapper that adds centroid-based classification head to teacher model.

    Uses cosine similarity to centroids as logits.
    """

    def __init__(
        self,
        base_model: nn.Module,
        centroid_head: CentroidClassificationHead,
        return_embeddings: bool = True,
        temperature: float = 1.0,
    ):
        """
        Args:
            base_model: Base teacher model that outputs embeddings
            centroid_head: Centroid classification head
            return_embeddings: Whether to return embeddings in output
            temperature: Temperature for scaling cosine similarities
        """
        super().__init__()
        self.base_model = base_model
        self.return_embeddings = return_embeddings
        self.temperature = temperature

        # Convert centroids to torch
        self.register_buffer("centroids", torch.from_numpy(centroid_head.centroids.astype(np.float32)))

        self.normalized = centroid_head.normalized
        self.speaker_to_idx = centroid_head.speaker_to_idx
        self.idx_to_speaker = centroid_head.idx_to_speaker
        self.num_classes = len(self.speaker_to_idx)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # Get base embeddings
        with torch.no_grad():
            base_output = self.base_model(x)

            if hasattr(base_output, "embeddings") and base_output.embeddings is not None:
                if isinstance(base_output.embeddings, list):
                    embeddings = base_output.embeddings[-1]
                else:
                    embeddings = base_output.embeddings
            elif hasattr(base_output, "logits"):
                embeddings = base_output.logits
            else:
                embeddings = base_output

        # Ensure correct shape
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        # Normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)

        # Centroids should already be normalized if config.centroid_normalize=True
        if not self.normalized:
            centroids_norm = F.normalize(self.centroids, p=2, dim=-1)
        else:
            centroids_norm = self.centroids

        # Compute cosine similarity to all centroids
        # embeddings_norm: (batch, embedding_dim)
        # centroids_norm: (num_classes, embedding_dim)
        logits = torch.matmul(embeddings_norm, centroids_norm.T) / self.temperature

        return ModelOutput(
            logits=logits,
            embeddings=embeddings if self.return_embeddings else None,
        )


# =============================================================================
# Main Preparation Function
# =============================================================================


def prepare_teacher(config: TeacherPrepConfig) -> nn.Module:
    """
    Prepare teacher model with classification head.

    Args:
        config: Configuration

    Returns:
        Wrapped teacher model with classification head
    """
    print("=" * 60)
    print("Teacher Model Preparation")
    print("=" * 60)
    print(f"Method: {config.method}")
    print(f"Data directory: {config.data_dir}")
    print(f"Output directory: {config.output_dir}")
    print()

    # 1. Load model
    print("Loading model...")
    model = get_model()
    if model is None:
        raise ValueError("get_model() returned None. Please implement model loading.")

    # 2. Discover files
    files = discover_voxceleb2_files(config.data_dir)

    # 3. Extract embeddings
    embeddings = extract_embeddings_for_files(model, files, config)

    # 4. Aggregate by utterance
    utterance_embeddings = aggregate_embeddings_by_utterance(embeddings)

    # 5. Compute classification head
    output_path = Path(config.output_dir)

    if config.method == "lda":
        lda_head = compute_lda_head(utterance_embeddings, config)
        lda_head.save(str(output_path / "lda_head.npz"))
        wrapped_model = LDATeacherWrapper(model, lda_head)
        print(f"LDA head saved to {output_path / 'lda_head.npz'}")

    elif config.method == "centroid":
        centroid_head = compute_centroid_head(utterance_embeddings, config)
        centroid_head.save(str(output_path / "centroid_head.npz"))
        wrapped_model = CentroidTeacherWrapper(model, centroid_head)
        print(f"Centroid head saved to {output_path / 'centroid_head.npz'}")

    elif config.method == "both":
        lda_head = compute_lda_head(utterance_embeddings, config)
        lda_head.save(str(output_path / "lda_head.npz"))
        centroid_head = compute_centroid_head(utterance_embeddings, config)
        centroid_head.save(str(output_path / "centroid_head.npz"))

    else:
        raise ValueError(f"Unknown method: {config.method}")

    # 6. Save wrapped model state dict
    torch.save(wrapped_model.state_dict(), output_path / "teacher_wrapped.pt")
    print(f"Wrapped model saved to {output_path / 'teacher_wrapped.pt'}")

    # 7. Save config
    config_dict = {
        "method": config.method,
        "embedding_dim": config.embedding_dim,
        "num_classes": wrapped_model.num_classes,
        "speaker_to_idx": wrapped_model.speaker_to_idx,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print()
    print("=" * 60)
    print("Preparation complete!")
    print(f"Number of classes: {wrapped_model.num_classes}")
    print("=" * 60)

    return wrapped_model


def load_prepared_teacher(
    output_dir: str,
    base_model: nn.Module,
) -> nn.Module:
    """
    Load a previously prepared teacher model.

    Args:
        output_dir: Directory with saved head
        base_model: Base teacher model

    Returns:
        Wrapped teacher model
    """
    output_path = Path(output_dir)

    # Load config
    with open(output_path / "config.json", "r") as f:
        config_dict = json.load(f)

    method = config_dict["method"]

    if method == "lda":
        lda_head = LDAClassificationHead.load(str(output_path / "lda_head.npz"))
        return LDATeacherWrapper(base_model, lda_head)
    elif method == "centroid":
        centroid_head = CentroidClassificationHead.load(str(output_path / "centroid_head.npz"))
        return CentroidTeacherWrapper(base_model, centroid_head)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare teacher model for distillation")
    parser.add_argument("--data-dir", type=str, default="data/vox2", help="Path to VoxCeleb2 data")
    parser.add_argument("--output-dir", type=str, default="teacher_prepared", help="Output directory")
    parser.add_argument(
        "--method", type=str, choices=["lda", "centroid"], default="lda", help="Classification head method"
    )
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--norm-type", type=str, default="std", help="Audio normalization type")

    args = parser.parse_args()

    config = TeacherPrepConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        method=args.method,
        embedding_dim=args.embedding_dim,
        device=args.device,
        norm_type=args.norm_type,
    )

    prepare_teacher(config)


if __name__ == "__main__":
    main()
