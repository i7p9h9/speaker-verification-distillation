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
    norm_type: str = 'std'

    # Processing
    device: str = 'cuda'

    # Method: 'lda' or 'centroid'
    method: str = 'lda'

    # Data limits (None = no limit)
    max_files_per_utterance: tp.Optional[int] = None
    max_utterances_per_speaker: tp.Optional[int] = None

    # LDA settings
    lda_n_components: tp.Optional[int] = None  # None = num_classes - 1
    lda_solver: str = 'svd'  # 'svd', 'lsqr', 'eigen'

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
    cfg = read_yaml(os.path.join(os.path.dirname(__file__), "data/cfg-models/rn100_v016_flr_vox4_v2.yaml"))
    state_dict_teacher = torch.load(
        os.path.join(os.path.dirname(__file__), "data/ckpt/rn100_v016_flr_vox4_v2/model.pt"),
        map_location=torch.device("cpu"),
    )

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


def discover_voxceleb2_files(
    data_dir: str,
    max_files_per_utterance: tp.Optional[int] = None,
    max_utterances_per_speaker: tp.Optional[int] = None,
) -> tp.Dict[str, tp.Dict[str, tp.List[FileInfo]]]:
    """
    Discover all wav files in VoxCeleb2 directory structure.

    Expected structure: data_dir/speaker_id/utterance_id/*.wav

    Returns:
        Nested dict: speaker_id -> utterance_id -> list of FileInfo
    """
    data_path = Path(data_dir)

    # Structure: speaker_id -> utterance_id -> [files]
    speaker_utterances: tp.Dict[str, tp.Dict[str, tp.List[FileInfo]]] = defaultdict(lambda: defaultdict(list))

    print(f"Discovering files in {data_dir}...")

    total_files = 0
    total_utterances = 0

    for speaker_dir in tqdm(sorted(data_path.iterdir()), desc="Scanning speakers"):
        if not speaker_dir.is_dir():
            continue

        speaker_id = speaker_dir.name
        utterance_count = 0

        for utterance_dir in sorted(speaker_dir.iterdir()):
            if not utterance_dir.is_dir():
                continue

            # Check utterance limit
            if max_utterances_per_speaker is not None and utterance_count >= max_utterances_per_speaker:
                break

            utterance_id = f"{speaker_id}/{utterance_dir.name}"

            wav_files = sorted(utterance_dir.glob("*.wav"))

            # Apply file limit per utterance
            if max_files_per_utterance is not None:
                wav_files = wav_files[:max_files_per_utterance]

            for wav_file in wav_files:
                speaker_utterances[speaker_id][utterance_id].append(FileInfo(
                    filepath=str(wav_file),
                    speaker_id=speaker_id,
                    utterance_id=utterance_id,
                ))
                total_files += 1

            if wav_files:
                utterance_count += 1
                total_utterances += 1

    num_speakers = len(speaker_utterances)
    print(f"Found {total_files} files, {total_utterances} utterances from {num_speakers} speakers")

    if max_files_per_utterance is not None:
        print(f"  (limited to {max_files_per_utterance} files per utterance)")
    if max_utterances_per_speaker is not None:
        print(f"  (limited to {max_utterances_per_speaker} utterances per speaker)")

    return dict(speaker_utterances)


# =============================================================================
# Embedding Extraction with Online Aggregation
# =============================================================================

def extract_utterance_embeddings(
    model: nn.Module,
    speaker_utterances: tp.Dict[str, tp.Dict[str, tp.List[FileInfo]]],
    config: TeacherPrepConfig,
) -> tp.List[UtteranceEmbedding]:
    """
    Extract embeddings for all files with online aggregation per utterance.

    Memory-efficient: aggregates embeddings immediately per utterance,
    never storing all file embeddings in memory.

    Uses length_segment_ms=None for full-length extraction (best quality).
    """
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    utterance_embeddings: tp.List[UtteranceEmbedding] = []

    # Count total utterances for progress bar
    total_utterances = sum(len(utts) for utts in speaker_utterances.values())

    print("Extracting embeddings with online aggregation...")

    with tqdm(total=total_utterances, desc="Processing utterances") as pbar:
        for speaker_id, utterances in speaker_utterances.items():
            for utterance_id, files in utterances.items():
                # Process all files for this utterance and aggregate immediately
                utterance_emb = _extract_and_aggregate_utterance(
                    model=model,
                    files=files,
                    config=config,
                    device=device,
                )

                if utterance_emb is not None:
                    utterance_embeddings.append(utterance_emb)

                pbar.update(1)

    print(f"Extracted {len(utterance_embeddings)} utterance embeddings")
    return utterance_embeddings


def _extract_and_aggregate_utterance(
    model: nn.Module,
    files: tp.List[FileInfo],
    config: TeacherPrepConfig,
    device: torch.device,
) -> tp.Optional[UtteranceEmbedding]:
    """
    Extract embeddings for a single utterance and aggregate them.

    Weighted average by audio duration.
    """
    if not files:
        return None

    speaker_id = files[0].speaker_id
    utterance_id = files[0].utterance_id

    filepaths = [f.filepath for f in files]

    # Create dataset for this utterance's files
    dataset = WavDataset(
        wav_scp=filepaths,
        norm_type=config.norm_type,
        length_segment_ms=None,
        segments_step_ms=None,
        sample_rate=config.sample_rate,
    )

    # Accumulate weighted embeddings
    weighted_sum: tp.Optional[np.ndarray] = None
    total_duration = 0.0
    num_files = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]

        segments = sample.segments
        if isinstance(segments, np.ndarray):
            segments = torch.from_numpy(segments)

        segments = segments.to(device)

        if segments.dim() == 1:
            segments = segments.unsqueeze(0)

        with torch.no_grad():
            output = model(segments)

            if hasattr(output, 'embeddings') and output.embeddings is not None:
                if isinstance(output.embeddings, list):
                    emb = output.embeddings[-1]
                else:
                    emb = output.embeddings
            elif hasattr(output, 'logits'):
                emb = output.logits
            else:
                emb = output

            # Handle multi-segment output
            if emb.dim() > 2:
                emb = emb.mean(dim=1)
            elif emb.dim() == 2 and emb.size(0) > 1:
                weights = torch.tensor(
                    sample.segments_weights,
                    device=device,
                    dtype=emb.dtype
                )
                emb = (emb * weights.unsqueeze(1)).sum(dim=0, keepdim=True)

            emb = emb.squeeze(0).cpu().numpy()

        duration = sample.total_duration

        # Accumulate weighted embedding
        if weighted_sum is None:
            weighted_sum = emb * duration
        else:
            weighted_sum += emb * duration

        total_duration += duration
        num_files += 1

    if weighted_sum is None or total_duration == 0:
        return None

    # Compute weighted average
    aggregated_emb = weighted_sum / total_duration

    return UtteranceEmbedding(
        speaker_id=speaker_id,
        utterance_id=utterance_id,
        embedding=aggregated_emb,
        total_duration=total_duration,
        num_files=num_files,
    )


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
    if config.max_files_per_utterance:
        print(f"Max files per utterance: {config.max_files_per_utterance}")
    if config.max_utterances_per_speaker:
        print(f"Max utterances per speaker: {config.max_utterances_per_speaker}")
    print()

    # 1. Load model
    print("Loading model...")
    model = get_model()
    if model is None:
        raise ValueError("get_model() returned None. Please implement model loading.")

    # 2. Discover files (grouped by speaker/utterance)
    speaker_utterances = discover_voxceleb2_files(
        config.data_dir,
        max_files_per_utterance=config.max_files_per_utterance,
        max_utterances_per_speaker=config.max_utterances_per_speaker,
    )

    # 3. Extract embeddings with online aggregation per utterance
    utterance_embeddings = extract_utterance_embeddings(model, speaker_utterances, config)

    # 4. Compute classification head
    output_path = Path(config.output_dir)

    if config.method == "lda":
        head = compute_lda_head(utterance_embeddings, config)
        head_name = "head_lda"
    elif config.method == "centroid":
        head = compute_centroid_head(utterance_embeddings, config)
        head_name = "head_centroid"
    elif config.method == "both":
        head_lda = compute_lda_head(utterance_embeddings, config)
        head_centroid = compute_centroid_head(utterance_embeddings, config)

        head_name_lda = "head_lda"
        head_name_centroid = "head_centroid"

        torch.save(head_lda.state_dict(), output_path / f"{head_name_lda}.pt")
        head_lda.save_speaker_mapping(str(output_path / f"{head_name_lda}_speakers.json"))
        print(f"LDA head saved to {output_path / f'{head_name_lda}.pt'}")

        torch.save(head_centroid.state_dict(), output_path / f"{head_name_centroid}.pt")
        head_centroid.save_speaker_mapping(str(output_path / f"{head_name_centroid}_speakers.json"))
        print(f"Centroid head saved to {output_path / f'{head_name_centroid}.pt'}")

        return head_lda, head_centroid
    else:
        raise ValueError(f"Unknown method: {config.method}")

    # 5. Save head state dict
    torch.save(head.state_dict(), output_path / f'{head_name}.pt')
    print(f"Head state dict saved to {output_path / f'{head_name}.pt'}")

    # 6. Save speaker mapping
    head.save_speaker_mapping(str(output_path / f'{head_name}_speakers.json'))
    print(f"Speaker mapping saved to {output_path / f'{head_name}_speakers.json'}")

    # 7. Save config
    config_dict = {
        'method': config.method,
        'embedding_dim': head.embedding_dim,
        'num_classes': head.num_classes,
        'max_files_per_utterance': config.max_files_per_utterance,
        'max_utterances_per_speaker': config.max_utterances_per_speaker,
    }
    if config.method == 'lda':
        config_dict['n_components'] = head.n_components
    else:
        config_dict['temperature'] = head.temperature

    with open(output_path / 'config.json', 'w') as f:
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
    state_dict = torch.load(head_path, map_location='cpu')

    if method == 'lda':
        # Infer dimensions from state dict
        projection = state_dict['projection']
        class_means = state_dict['class_means']
        n_components, embedding_dim = projection.shape
        num_classes = class_means.shape[0]

        head = HeadClassificationLDA(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            n_components=n_components,
        )
    elif method == 'centroid':
        # Infer dimensions from state dict
        centroids = state_dict['centroids.weight']
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

    parser = argparse.ArgumentParser(description='Prepare teacher classification head for distillation')
    parser.add_argument('--data-dir', type=str, default='data/vox2',
                        help='Path to VoxCeleb2 data')
    parser.add_argument('--output-dir', type=str, default='teacher_prepared',
                        help='Output directory')
    parser.add_argument('--method', type=str, choices=['lda', 'centroid', 'both'], default='lda',
                        help='Classification head method')
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--norm-type', type=str, default='std',
                        help='Audio normalization type')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for centroid method')
    parser.add_argument('--max-files-per-utterance', type=int, default=None,
                        help='Maximum files per utterance (default: no limit)')
    parser.add_argument('--max-utterances-per-speaker', type=int, default=None,
                        help='Maximum utterances per speaker (default: no limit)')

    args = parser.parse_args()

    config = TeacherPrepConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        method=args.method,
        embedding_dim=args.embedding_dim,
        device=args.device,
        norm_type=args.norm_type,
        centroid_temperature=args.temperature,
        max_files_per_utterance=args.max_files_per_utterance,
        max_utterances_per_speaker=args.max_utterances_per_speaker,
    )

    prepare_teacher_head(config)


if __name__ == '__main__':
    main()