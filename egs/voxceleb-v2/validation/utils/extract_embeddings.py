import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_logits: bool = False,
    normalize: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Extract embeddings from a model for all samples in dataloader.

    Args:
        model: Model to extract embeddings from
        dataloader: DataLoader with samples
        device: Device to run inference on
        use_logits: If True, use logits instead of embeddings
        normalize: If True, L2 normalize embeddings
        show_progress: If True, show progress bar

    Returns:
        Array of embeddings with shape (num_samples, embedding_dim)
    """
    model.eval()
    embeddings_list = []

    iterator = tqdm(dataloader, desc="Extracting embeddings") if show_progress else dataloader

    with torch.no_grad():
        for batch in iterator:
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            output = model(inputs)

            if use_logits:
                emb = output.logits
            elif output.embeddings is not None:
                if isinstance(output.embeddings, list):
                    # Use last layer embeddings
                    emb = output.embeddings[-1]
                else:
                    emb = output.embeddings
            else:
                raise ValueError("Model output has no embeddings and use_logits=False")

            if normalize:
                emb = F.normalize(emb, p=2, dim=-1)

            embeddings_list.append(emb.cpu().numpy())

    return np.concatenate(embeddings_list, axis=0)


def extract_embeddings_with_aggregation(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    weights_attr: str = 'segments_weights',
    normalize: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Extract embeddings with weighted aggregation (e.g., for variable-length segments).

    This is useful for speaker verification where each sample may have multiple
    segments that need to be aggregated into a single embedding.

    Args:
        model: Model to extract embeddings from
        dataset: Dataset where each sample has segments and weights
        device: Device to run inference on
        weights_attr: Attribute name for segment weights in sample
        normalize: If True, L2 normalize final embeddings
        show_progress: If True, show progress bar

    Returns:
        Array of aggregated embeddings with shape (num_samples, embedding_dim)
    """
    state_traing: bool = model.training

    model.eval()
    embeddings_list = []

    iterator = tqdm(dataset, desc="Extracting embeddings") if show_progress else dataset

    for sample in iterator:
        # Get segments tensor
        if hasattr(sample, 'segments'):
            segments = sample.segments
        elif isinstance(sample, dict):
            segments = sample['segments']
        else:
            segments = sample[0]

        # Get weights
        if hasattr(sample, weights_attr):
            weights = getattr(sample, weights_attr)
        elif isinstance(sample, dict):
            weights = sample.get(weights_attr, None)
        else:
            weights = None

        if isinstance(segments, np.ndarray):
            segments = torch.from_numpy(segments)

        segments = segments.to(device)

        with torch.no_grad():
            output = model(segments)

            if output.embeddings is not None:
                if isinstance(output.embeddings, list):
                    emb = output.embeddings[-1]
                else:
                    emb = output.embeddings
            else:
                emb = output.logits

        emb_np = emb.cpu().numpy()

        # Aggregate with weights
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                weights = weights.numpy()
            aggregated = np.matmul(np.asarray(weights), emb_np)
        else:
            aggregated = emb_np.mean(axis=0)

        if normalize:
            aggregated = aggregated / (np.linalg.norm(aggregated) + 1e-8)

        embeddings_list.append(aggregated)

    if state_traing:
        model.train()

    return np.stack(embeddings_list, axis=0)
