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
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from voicesdk.distillation.data.pipe import WavDataset
from voicesdk.distillation.nn.layers.head import (
    HeadClassificationCentroids,
    HeadClassificationLDA,
)
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
    device: str = "cuda"

    # Method: 'lda' or 'centroid'
    method: str = "lda"

    # LDA settings
    lda_n_components: tp.Optional[int] = None  # None = num_classes - 1
    lda_solver: str = "svd"  # 'svd', 'lsqr', 'eigen'

    # Centroid settings
    centroid_normalize: bool = True
    centroid_temperature: float = 1.0

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
    """
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    filepaths = [f.filepath for f in files]
    dataset = WavDataset(
        wav_scp=filepaths,
        norm_type=config.norm_type,
        length_segment_ms=None,
        segments_step_ms=None,
        sample_rate=config.sample_rate,
    )

    embeddings = []

    print("Extracting embeddings...")
    for idx in tqdm(range(len(dataset)), desc="Extracting"):
        sample = dataset[idx]
        file_info = files[idx]

        segments = sample.segments
        if isinstance(segments, np.ndarray):
            segments = torch.from_numpy(segments)

        segments = segments.to(device)

        if segments.dim() == 1:
            segments = segments.unsqueeze(0)

        with torch.no_grad():
            output = model(segments)

            if hasattr(output, "embeddings") and output.embeddings is not None:
                if isinstance(output.embeddings, list):
                    emb = output.embeddings[-1]
                else:
                    emb = output.embeddings
            elif hasattr(output, "logits"):
                emb = output.logits
            else:
                emb = output

            if emb.dim() > 2:
                emb = emb.mean(dim=1)
            elif emb.dim() == 2 and emb.size(0) > 1:
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
    """Aggregate embeddings per utterance using weighted average by duration."""
    utterance_groups: tp.Dict[str, tp.List[ExtractedEmbedding]] = defaultdict(list)
    for emb in embeddings:
        utterance_groups[emb.utterance_id].append(emb)

    aggregated = []

    print("Aggregating embeddings by utterance...")
    for utterance_id, group in tqdm(utterance_groups.items(), desc="Aggregating"):
        speaker_id = group[0].speaker_id
        total_duration = sum(e.duration for e in group)

        if total_duration > 0:
            weighted_sum = np.zeros_like(group[0].embedding)
            for e in group:
                weight = e.duration / total_duration
                weighted_sum += weight * e.embedding
            aggregated_emb = weighted_sum
        else:
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
# Head Computation
# =============================================================================


def compute_lda_head(
    utterance_embeddings: tp.List[UtteranceEmbedding],
    config: TeacherPrepConfig,
) -> HeadClassificationLDA:
    """Compute LDA-based classification head."""
    X = np.array([e.embedding for e in utterance_embeddings])
    speakers = [e.speaker_id for e in utterance_embeddings]

    unique_speakers = sorted(set(speakers))
    speaker_to_idx = {s: i for i, s in enumerate(unique_speakers)}

    y = np.array([speaker_to_idx[s] for s in speakers])

    print(f"Computing LDA with {len(unique_speakers)} classes and {len(X)} samples...")

    n_components = config.lda_n_components
    if n_components is None:
        n_components = min(len(unique_speakers) - 1, X.shape[1])

    lda = LinearDiscriminantAnalysis(
        n_components=n_components,
        solver=config.lda_solver,
    )
    lda.fit(X, y)

    projection = lda.scalings_[:, :n_components].T

    X_lda = lda.transform(X)
    class_means = np.zeros((len(unique_speakers), n_components))

    for class_idx in range(len(unique_speakers)):
        mask = y == class_idx
        if mask.sum() > 0:
            class_means[class_idx] = X_lda[mask].mean(axis=0)

    print(f"LDA projection shape: {projection.shape}")
    print(f"Class means shape: {class_means.shape}")

    return HeadClassificationLDA(
        projection=projection,
        class_means=class_means,
        global_mean=lda.xbar_,
        speaker_to_idx=speaker_to_idx,
    )


def compute_centroid_head(
    utterance_embeddings: tp.List[UtteranceEmbedding],
    config: TeacherPrepConfig,
) -> HeadClassificationCentroids:
    """Compute centroid-based classification head."""
    speaker_utterances: tp.Dict[str, tp.List[UtteranceEmbedding]] = defaultdict(list)
    for utt in utterance_embeddings:
        speaker_utterances[utt.speaker_id].append(utt)

    unique_speakers = sorted(speaker_utterances.keys())
    speaker_to_idx = {s: i for i, s in enumerate(unique_speakers)}

    embedding_dim = utterance_embeddings[0].embedding.shape[0]
    centroids = np.zeros((len(unique_speakers), embedding_dim))

    print(f"Computing centroids for {len(unique_speakers)} speakers...")

    for speaker_id, utterances in tqdm(speaker_utterances.items(), desc="Computing centroids"):
        class_idx = speaker_to_idx[speaker_id]
        total_duration = sum(u.total_duration for u in utterances)

        if total_duration > 0:
            weighted_sum = np.zeros(embedding_dim)
            for u in utterances:
                weight = u.total_duration / total_duration
                weighted_sum += weight * u.embedding
            centroid = weighted_sum
        else:
            centroid = np.mean([u.embedding for u in utterances], axis=0)

        if config.centroid_normalize:
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        centroids[class_idx] = centroid

    print(f"Centroids shape: {centroids.shape}")

    return HeadClassificationCentroids(
        centroids=centroids,
        speaker_to_idx=speaker_to_idx,
        temperature=config.centroid_temperature,
        normalize_input=True,
    )


# =============================================================================
# Main Preparation Function
# =============================================================================


def prepare_teacher_head(config: TeacherPrepConfig) -> nn.Module:
    """
    Prepare classification head for teacher model.

    Returns:
        Classification head layer (HeadClassificationLDA or HeadClassificationCentroids)
    """
    print("=" * 60)
    print("Teacher Classification Head Preparation")
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
        head = compute_lda_head(utterance_embeddings, config)
        head_name = "head_lda"
    elif config.method == "centroid":
        head = compute_centroid_head(utterance_embeddings, config)
        head_name = "head_centroid"
    else:
        raise ValueError(f"Unknown method: {config.method}")

    # 6. Save head state dict
    torch.save(head.state_dict(), output_path / f"{head_name}.pt")
    print(f"Head state dict saved to {output_path / f'{head_name}.pt'}")

    # 7. Save speaker mapping
    head.save_speaker_mapping(str(output_path / f"{head_name}_speakers.json"))
    print(f"Speaker mapping saved to {output_path / f'{head_name}_speakers.json'}")

    # 8. Save config
    config_dict = {
        "method": config.method,
        "embedding_dim": head.embedding_dim,
        "num_classes": head.num_classes,
    }
    if config.method == "lda":
        config_dict["n_components"] = head.n_components
    else:
        config_dict["temperature"] = head.temperature

    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print()
    print("=" * 60)
    print("Preparation complete!")
    print(f"Number of classes: {head.num_classes}")
    print("=" * 60)

    return head


def load_head(
    head_path: str,
    method: str,
    speaker_mapping_path: tp.Optional[str] = None,
) -> nn.Module:
    """
    Load a previously saved classification head.

    Args:
        head_path: Path to head state dict
        method: 'lda' or 'centroid'
        speaker_mapping_path: Optional path to speaker mapping JSON

    Returns:
        Classification head layer
    """
    state_dict = torch.load(head_path, map_location="cpu")

    if method == "lda":
        # Infer dimensions from state dict
        projection = state_dict["projection"]
        class_means = state_dict["class_means"]
        n_components, embedding_dim = projection.shape
        num_classes = class_means.shape[0]

        head = HeadClassificationLDA(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            n_components=n_components,
        )
    elif method == "centroid":
        # Infer dimensions from state dict
        centroids = state_dict["centroids.weight"]
        num_classes, embedding_dim = centroids.shape

        head = HeadClassificationCentroids(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    head.load_state_dict(state_dict)

    if speaker_mapping_path:
        head.load_speaker_mapping(speaker_mapping_path)

    return head


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare teacher classification head for distillation")
    parser.add_argument("--data-dir", type=str, default="data/vox2", help="Path to VoxCeleb2 data")
    parser.add_argument("--output-dir", type=str, default="teacher_prepared", help="Output directory")
    parser.add_argument(
        "--method", type=str, choices=["lda", "centroid"], default="lda", help="Classification head method"
    )
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--norm-type", type=str, default="std", help="Audio normalization type")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for centroid method")

    args = parser.parse_args()

    config = TeacherPrepConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        method=args.method,
        embedding_dim=args.embedding_dim,
        device=args.device,
        norm_type=args.norm_type,
        centroid_temperature=args.temperature,
    )

    prepare_teacher_head(config)


if __name__ == "__main__":
    main()
