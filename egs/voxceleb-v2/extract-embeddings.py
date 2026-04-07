import random
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from voicesdk.distillation.data import AudioReaderBase, AudioReaderRandom, AudioSegments
from voicesdk.nn.arch import ReDimNetWrap, ResNetTF
from voicesdk.nn.loss import AMSoftmaxLoss
from voicesdk.utils.find_files import find_files_recursive

@dataclass
class AudioSample:
    """Single sample returned by WavLimitDataset."""
    segments: "AudioSegments"
    path: Path
    spk_id: str


class WavLimitDataset(Dataset):
    """
    Dataset that maps a list of wav file paths to AudioSample instances.
    Limits the number of utterances per speaker.
    """

    def __init__(
        self,
        wav_scp: tp.List[str],
        reader: "AudioReaderBase",
        spk_level: int = -3,
        max_utt: int = 10,
        random_select: bool = False,
        seed: tp.Optional[int] = None,
    ):
        """
        Args:
            wav_scp: List of paths to wav files
            reader: AudioReaderBase instance that defines how each file is read
            spk_level: Index in path.parts used to extract speaker ID
            max_utt: Maximum number of utterances kept per speaker
            random_select: If True, select utterances randomly; otherwise take
                           the first max_utt after sorting
            seed: Optional random seed used when random_select=True
        """
        self._spk_level = spk_level
        self._max_utt = max_utt
        self._random_select = random_select
        self._rng = random.Random(seed)

        self.reader = reader
        self.wav_list: tp.List[Path] = self._filter(
            [Path(x) for x in wav_scp]
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_spk(self, path: Path) -> str:
        return path.parts[self._spk_level]

    def _filter(self, paths: tp.List[Path]) -> tp.List[Path]:
        """Group paths by speaker, apply per-speaker limit, return flat list."""
        spk2utt: tp.Dict[str, tp.List[Path]] = {}
        for path in paths:
            spk = self._extract_spk(path)
            spk2utt.setdefault(spk, []).append(path)

        filtered: tp.List[Path] = []
        for spk, utt_paths in spk2utt.items():
            if self._random_select:
                selected = self._rng.sample(
                    utt_paths, min(self._max_utt, len(utt_paths))
                )
            else:
                selected = sorted(utt_paths)[: self._max_utt]
            filtered.extend(selected)

        return filtered

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.wav_list)

    def __getitem__(self, idx: int) -> AudioSample:
        path = self.wav_list[idx]
        spk_id = self._extract_spk(path)
        segments = self.reader.read(path)
        return AudioSample(segments=segments, path=path, spk_id=spk_id)


# ----------------------------------------------------------------------
# Collate
# ----------------------------------------------------------------------

@dataclass
class AudioBatch:
    """Output of collate_audio_samples."""
    audio: torch.Tensor          # (B, 1, T)  — padded & stacked
    paths: tp.List[Path]
    spk_ids: tp.List[str]


def collate_audio_samples(batch: tp.List[AudioSample]) -> AudioBatch:
    """
    Collate a list of AudioSample into a tensorised batch.

    Audio tensors are zero-padded along the time dimension to the
    length of the longest segment in the batch.

    Args:
        batch: List of AudioSample produced by WavLimitDataset.__getitem__

    Returns:
        AudioBatch with:
            audio   — FloatTensor of shape (B, 1, T_max)
            paths   — list of Path objects
            spk_ids — list of speaker ID strings
    """
    tensors: tp.List[torch.Tensor] = []
    for sample in batch:
        # segments.segments has shape (num_crops, T); take the first crop
        arr: np.ndarray = sample.segments.segments[0]          # (T,)
        tensors.append(torch.from_numpy(arr).float())           # (T,)

    # Pad to the longest sequence in the batch
    max_len = max(t.shape[-1] for t in tensors)
    padded = torch.zeros(len(tensors), 1, max_len)
    for i, t in enumerate(tensors):
        padded[i, 0, : t.shape[-1]] = t

    return AudioBatch(
        audio=padded,
        paths=[s.path for s in batch],
        spk_ids=[s.spk_id for s in batch],
    )

weights_path = Path('data/exps/vox2-emb-cosine-001/student-last.ckpt')

def read_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    return dict(hparams)

def get_model() -> nn.Module:
    cfg = read_yaml("data/cfg-models/redimnet_L.yaml")
    # _model_backbone = ResNetTF
    _model_backbone = ReDimNetWrap
    model = _model_backbone(**cfg["model_args"])

    state_dict_loaded = torch.load(weights_path)
    model.load_state_dict(state_dict_loaded)
    # state_dict_student = model.state_dict()
    # for k in state_dict_student:
    #     state_dict_student[k].copy_(state_dict_loaded["state_dict"][f"student_model.model_base.{k}"])

    return model

TRAIN_VOX = "/media/ssd/voice/datasets/vox2/dev-16k/aac/"

reader = AudioReaderRandom(length_segment_ms=3000, norm_type="std")
wav_scp = find_files_recursive(TRAIN_VOX)
ds = WavLimitDataset(wav_scp, reader, spk_level=-3, max_utt=5, random_select=False)
loader = DataLoader(ds, batch_size=100, collate_fn=collate_audio_samples, drop_last=False, shuffle=False)

model = get_model().eval().cuda()

save_path = weights_path.parent / "embedings" / "vox2"
save_path.mkdir(parents=True)

for n_batch, batch in tqdm(enumerate(loader), total=len(loader)):
    with torch.no_grad():
        embs = model(batch.audio.cuda())
        np.save(save_path / f"batch-emb-{n_batch:02d}.npy", embs.cpu().numpy())
        np.save(save_path / f"batch-spk-{n_batch:02d}.npy", batch.spk_ids)