import csv
import dataclasses
import random
import typing as tp
from pathlib import Path

from voicesdk.distillation.data import AudioReaderBase, WavDataset
from voicesdk.utils.find_files import find_files_recursive


@dataclasses.dataclass
class SpoofLiveDatasets:
    dataset_live: WavDataset
    dataset_spoof: WavDataset


def _sample(items: tp.List[str], limit: tp.Optional[int], seed: int) -> tp.List[str]:
    """Return up to *limit* items sampled without replacement (reproducible)."""
    if limit is None or limit >= len(items):
        return items
    rng = random.Random(seed)
    return rng.sample(items, limit)


def get_asv_spoof_17(
    dir_main: str,
    reader: AudioReaderBase,
    limit_spoof: tp.Optional[int] = None,
    limit_live: tp.Optional[int] = None,
    seed: int = 42,
) -> SpoofLiveDatasets:
    """
    Load ASVspoof 2017 dataset.

    Expected layout::

        dir_main/
            meta.csv          # header: path,type,spk,phrase,env,playback_device,recording_device
            eval/
                E_1000001.wav
                ...

    Args:
        dir_main:    Root directory of the dataset.
        reader:      AudioReaderBase instance shared by both sub-datasets.
        limit_spoof: Maximum number of spoof files to include (None = all).
        limit_live:  Maximum number of live/bonafide files to include (None = all).
        seed:        Random seed used when sampling is needed.

    Returns:
        SpoofLiveDatasets with .dataset_live and .dataset_spoof.
    """
    meta_path = Path(dir_main) / "meta.csv"
    live_paths: tp.List[str] = []
    spoof_paths: tp.List[str] = []

    with open(meta_path, newline="", encoding="utf-8") as f:
        reader_csv = csv.DictReader(f)
        for row in reader_csv:
            full_path = str(Path(dir_main) / row["path"])
            if row["type"].strip().lower() == "replay":
                live_paths.append(full_path)
            else:
                spoof_paths.append(full_path)

    live_paths = _sample(live_paths, limit_live, seed)
    spoof_paths = _sample(spoof_paths, limit_spoof, seed)

    return SpoofLiveDatasets(
        dataset_live=WavDataset(live_paths, reader),
        dataset_spoof=WavDataset(spoof_paths, reader),
    )


def get_asv_spoof_19(
    dir_wav: str,
    file_meta: str,
    reader: AudioReaderBase,
    limit_spoof: tp.Optional[int] = None,
    limit_live: tp.Optional[int] = None,
    seed: int = 42,
) -> SpoofLiveDatasets:
    """
    Load ASVspoof 2019 dataset.

    Meta file format (space-separated, no header)::

        LA_0078 LA_D_9377948 - VC_4 spoof
        LA_0069 LA_D_1047731 - -   bonafide

    The filename is taken from column index 1; ``.wav`` is appended and the
    file is looked up directly inside *dir_wav*.

    Args:
        dir_wav:     Directory that contains all ``.wav`` files flat.
        file_meta:   Path to the protocol / meta text file.
        reader:      AudioReaderBase instance shared by both sub-datasets.
        limit_spoof: Maximum number of spoof files to include (None = all).
        limit_live:  Maximum number of bonafide files to include (None = all).
        seed:        Random seed used when sampling is needed.

    Returns:
        SpoofLiveDatasets with .dataset_live and .dataset_spoof.
    """
    live_paths: tp.List[str] = []
    spoof_paths: tp.List[str] = []

    with open(file_meta, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # parts[1] — utterance id, parts[-1] — label
            utt_id = parts[1]
            label = parts[-1].lower()
            full_path = str(Path(dir_wav) / f"{utt_id}.wav")
            if label == "bonafide":
                live_paths.append(full_path)
            else:
                spoof_paths.append(full_path)

    live_paths = _sample(live_paths, limit_live, seed)
    spoof_paths = _sample(spoof_paths, limit_spoof, seed)

    return SpoofLiveDatasets(
        dataset_live=WavDataset(live_paths, reader),
        dataset_spoof=WavDataset(spoof_paths, reader),
    )


def get_splitted(
    dir_live: str,
    dir_spoof: str,
    reader: AudioReaderBase,
    limit_spoof: tp.Optional[int] = None,
    limit_live: tp.Optional[int] = None,
    seed: int = 42,
) -> SpoofLiveDatasets:
    """
    Load a dataset that is already split into two directories.

    Searches both directories recursively for ``.wav`` files.

    Args:
        dir_live:    Directory (or tree) containing bonafide recordings.
        dir_spoof:   Directory (or tree) containing spoof recordings.
        reader:      AudioReaderBase instance shared by both sub-datasets.
        limit_spoof: Maximum number of spoof files to include (None = all).
        limit_live:  Maximum number of live files to include (None = all).
        seed:        Random seed used when sampling is needed.

    Returns:
        SpoofLiveDatasets with .dataset_live and .dataset_spoof.
    """
    live_paths: tp.List[str] = list(find_files_recursive(dir_live, extension=".wav"))
    spoof_paths: tp.List[str] = list(find_files_recursive(dir_spoof, extension=".wav"))

    live_paths = _sample(live_paths, limit_live, seed)
    spoof_paths = _sample(spoof_paths, limit_spoof, seed)

    return SpoofLiveDatasets(
        dataset_live=WavDataset(live_paths, reader),
        dataset_spoof=WavDataset(spoof_paths, reader),
    )
