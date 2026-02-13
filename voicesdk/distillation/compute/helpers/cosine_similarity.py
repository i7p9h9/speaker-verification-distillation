import typing as tp
import numpy as np
import torch


ArrayOrTensor = tp.Union[np.ndarray, torch.Tensor]


def cosine_similarity(
    vec1: ArrayOrTensor,
    vec2: ArrayOrTensor,
    keepdims: bool = False,
    eps: float = 1e-10,
    axis: int = -1,
) -> ArrayOrTensor:
    """
    Compute cosine similarity between two arrays/tensors along specified axis.

    Supports both numpy arrays and pytorch tensors with arbitrary dimensions.
    Uses einsum for efficient computation.

    Args:
        vec1: First array or tensor
        vec2: Second array or tensor (must be broadcastable with vec1)
        keepdims: If True, retains the reduced dimension with size 1
        eps: Small epsilon for numerical stability
        axis: Axis along which to compute similarity

    Returns:
        Cosine similarity with shape determined by input shape and axis.
        If input shape is (2, 3, 4) and axis=-1:
            - keepdims=False: output shape (2, 3)
            - keepdims=True: output shape (2, 3, 1)
    """
    is_torch = isinstance(vec1, torch.Tensor)

    if is_torch:
        return _cosine_similarity_torch(vec1, vec2, keepdims, eps, axis)
    else:
        return _cosine_similarity_numpy(vec1, vec2, keepdims, eps, axis)


def _build_einsum_subscripts(ndim: int, axis: int) -> str:
    """
    Build einsum subscripts for dot product along specified axis.

    For ndim=3 and axis=-1 (normalized to 2):
        Returns "ijk,ijk->ij" (contracts over k)
    """
    # Normalize negative axis
    if axis < 0:
        axis = ndim + axis

    # Generate subscript letters
    letters = "abcdefghijklmnopqrstuvwxyz"
    input_subscripts = letters[:ndim]

    # Output subscripts exclude the contracted axis
    output_subscripts = input_subscripts[:axis] + input_subscripts[axis + 1:]

    return f"{input_subscripts},{input_subscripts}->{output_subscripts}"


def _cosine_similarity_numpy(
    vec1: np.ndarray,
    vec2: np.ndarray,
    keepdims: bool,
    eps: float,
    axis: int,
) -> np.ndarray:
    """Numpy implementation using einsum."""
    ndim = vec1.ndim
    subscripts = _build_einsum_subscripts(ndim, axis)

    # Compute dot product using einsum
    dot_product = np.einsum(subscripts, vec1, vec2)

    # Compute norms along the specified axis
    norm1 = np.sqrt(np.einsum(subscripts, vec1, vec1))
    norm2 = np.sqrt(np.einsum(subscripts, vec2, vec2))

    # Compute cosine similarity with numerical stability
    similarity = dot_product / np.maximum(norm1 * norm2, eps)

    if keepdims:
        similarity = np.expand_dims(similarity, axis=axis)

    return similarity


def _cosine_similarity_torch(
    vec1: torch.Tensor,
    vec2: torch.Tensor,
    keepdims: bool,
    eps: float,
    axis: int,
) -> torch.Tensor:
    """PyTorch implementation using einsum."""
    ndim = vec1.ndim
    subscripts = _build_einsum_subscripts(ndim, axis)

    # Compute dot product using einsum
    dot_product = torch.einsum(subscripts, vec1, vec2)

    # Compute norms along the specified axis
    norm1 = torch.sqrt(torch.einsum(subscripts, vec1, vec1))
    norm2 = torch.sqrt(torch.einsum(subscripts, vec2, vec2))

    # Compute cosine similarity with numerical stability
    similarity = dot_product / torch.clamp(norm1 * norm2, min=eps)

    if keepdims:
        similarity = similarity.unsqueeze(axis)

    return similarity


if __name__ == "__main__":
    # Test with numpy
    print("=== NumPy Tests ===")
    np_vec1 = np.random.randn(2, 3, 4)
    np_vec2 = np.random.randn(2, 3, 4)

    result_np = cosine_similarity(np_vec1, np_vec2, axis=-1)
    print(f"Input shape: {np_vec1.shape}")
    print(f"Output shape (keepdims=False): {result_np.shape}")

    result_np_keepdims = cosine_similarity(np_vec1, np_vec2, axis=-1, keepdims=True)
    print(f"Output shape (keepdims=True): {result_np_keepdims.shape}")

    # Test with pytorch
    print("\n=== PyTorch Tests ===")
    torch_vec1 = torch.randn(2, 3, 4)
    torch_vec2 = torch.randn(2, 3, 4)

    result_torch = cosine_similarity(torch_vec1, torch_vec2, axis=-1)
    print(f"Input shape: {tuple(torch_vec1.shape)}")
    print(f"Output shape (keepdims=False): {tuple(result_torch.shape)}")

    result_torch_keepdims = cosine_similarity(torch_vec1, torch_vec2, axis=-1, keepdims=True)
    print(f"Output shape (keepdims=True): {tuple(result_torch_keepdims.shape)}")

    # Verify correctness with 1D case
    print("\n=== Verification (1D) ===")
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    print(f"Parallel vectors: {cosine_similarity(v1, v2)}")  # Should be 1.0

    v3 = np.array([0.0, 1.0, 0.0])
    print(f"Orthogonal vectors: {cosine_similarity(v1, v3)}")  # Should be 0.0

    v4 = np.array([-1.0, 0.0, 0.0])
    print(f"Opposite vectors: {cosine_similarity(v1, v4)}")  # Should be -1.0