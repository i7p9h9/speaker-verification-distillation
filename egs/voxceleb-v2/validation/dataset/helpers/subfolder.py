import os
import typing as tp
from pathlib import Path

from aggregated_dataset import WeightedDataset
from torch.utils.data import Dataset

# Factory type: receives the absolute path to a subfolder, returns a Dataset
DatasetFactory = tp.Callable[[str], Dataset]


def datasets_from_subfolders(
    root: tp.Union[str, Path],
    factory: DatasetFactory,
    *,
    weight: tp.Optional[float] = None,
    recursive: bool = False,
    extensions: tp.Optional[tp.FrozenSet[str]] = None,
    ignore_empty: bool = True,
    name_fn: tp.Optional[tp.Callable[[Path], str]] = None,
    prefix: str = ""
) -> tp.List[WeightedDataset]:
    """
    Scan *root* for immediate subdirectories and return one WeightedDataset
    per subdirectory, ready to be passed to ``LabeledSource.dataset``.

    Parameters
    ----------
    root:
        Parent directory whose immediate subdirectories become datasets.
    factory:
        Callable that receives the absolute path to a subdirectory (as ``str``)
        and returns a :class:`~torch.utils.data.Dataset`.
        Example::

            factory=lambda path: VoxDataset(reader=reader_train, root=path)

    weight:
        Sampling weight assigned to every produced WeightedDataset.
        ``None`` (default) → all-None, which resolves to equal probability
        inside :class:`AggregatedDataset`.
    recursive:
        If ``True``, scan all subdirectories at any depth instead of only
        immediate children.
    extensions:
        When provided, a subfolder is considered *non-empty* only if it
        contains at least one file with one of these extensions (e.g.
        ``frozenset({".wav", ".flac", ".mp3"})``).
        When ``None`` (default), any file counts.
    ignore_empty:
        Skip subdirectories that contain no files (after optional extension
        filtering).  Default ``True``.
    name_fn:
        Optional callable that maps a subfolder :class:`~pathlib.Path` to the
        dataset name string.  Defaults to ``subfolder.name`` (the directory's
        basename).

    Returns
    -------
    List[WeightedDataset]
        One entry per discovered subfolder, sorted by name for reproducibility.

    Raises
    ------
    FileNotFoundError
        If *root* does not exist.
    ValueError
        If no subdirectories (or no non-empty ones) were found under *root*.

    Example
    -------
    ::

        sources = [
            LabeledSource(
                dataset=datasets_from_subfolders(
                    root=TRAIN_CODECS,
                    factory=lambda path: VoxDataset(reader=reader_train, root=path),
                    extensions=frozenset({".wav", ".flac"}),
                ),
                label=1,
                weight=1.0,
                name="codecs",
            ),
        ]
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    if recursive:
        candidates = sorted(p for p in root.rglob("*") if p.is_dir())
    else:
        candidates = sorted(p for p in root.iterdir() if p.is_dir())

    if not candidates:
        raise ValueError(f"No subdirectories found under {root}")

    result: tp.List[WeightedDataset] = []
    skipped: tp.List[Path] = []

    for subfolder in candidates:
        if ignore_empty and not _has_files(subfolder, extensions):
            skipped.append(subfolder)
            continue

        ds_name = name_fn(subfolder) if name_fn is not None else subfolder.name
        ds = factory(str(subfolder))

        result.append(WeightedDataset(dataset=ds, weight=weight, name=f"{prefix}{ds_name}"))

    if skipped:
        skipped_names = ", ".join(p.name for p in skipped)
        print(f"[datasets_from_subfolders] Skipped empty dirs: {skipped_names}")

    if not result:
        raise ValueError(
            f"All subdirectories under {root} were empty (or filtered out). "
            "Set ignore_empty=False to include them anyway."
        )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_files(
    directory: Path,
    extensions: tp.Optional[tp.FrozenSet[str]],
) -> bool:
    """Return True if *directory* contains at least one matching file."""
    for entry in os.scandir(directory):
        if not entry.is_file():
            continue
        if extensions is None:
            return True
        if Path(entry.name).suffix.lower() in extensions:
            return True
    return False
