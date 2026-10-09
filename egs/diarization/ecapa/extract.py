"""Single-step extraction of speaker embeddings and attention-based VAD logits.

Implements the extraction step of Thienpondt and Demuynck, "Speaker Embeddings
With Weakly Supervised Voice Activity Detection For Efficient Speaker
Diarization" (Odyssey 2024), Section 4: a sliding window (2 s, step 1 s) is run
over every file and one forward pass gives both the window embedding and the
frame-level VAD logits ``v_t`` (channel-averaged pre-softmax attention scores,
equation (5)).  Overlapping ``v_t`` values are averaged onto a global frame grid.

Files are read and windowed by :class:`AudioReaderFull` in DataLoader workers;
windows of several files are collated into one batch.  The reader pads the last
partial window by tiling, so frames past the end of the file are dropped.

Works with any model that has a ``pooling`` submodule returning
``(pooled, attention_logits)`` with logits of shape ``(B, C, T)``, i.e.
:class:`ECAPA2` and :class:`ECAPA_TDNN_VAD`.  Logits are captured with a forward
hook, so the model's ``forward`` may return the embedding only.

For every audio file ``<wav-dir>/<rel>.wav`` an ``<out-dir>/<rel>.npz`` is
written with:

    embeddings      (W, D) float32  window embeddings
    window_starts   (W,)   float32  window start times, seconds
    window_ends     (W,)   float32  window end times (clipped to the file), seconds
    window_vad      (W, T) float32  per-window frame VAD logits
    vad             (N,)   float32  overlap-averaged VAD logits, frame i at i * frame_shift
    frame_shift     ()     float32  seconds
    duration        ()     float32  seconds

Usage:
    python egs/diarization/ecapa/extract.py --wav-dir data/wav --out-dir data/npz \
        --arch ECAPA2 --config cfg.yaml --ckpt model.ckpt --device cuda
"""

from __future__ import annotations

import argparse
import typing as tp
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from voicesdk.dataset import LabeledSample
from voicesdk.distillation.data import AudioReaderFull, collate_batch_labeled_segments_fn
from voicesdk.nn.arch import ECAPA2, ECAPA_TDNN_VAD

ARCHS: tp.Dict[str, tp.Type[nn.Module]] = {
    "ECAPA2": ECAPA2,
    "ECAPA_TDNN_VAD": ECAPA_TDNN_VAD,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--arch", choices=sorted(ARCHS), default="ECAPA2")
    p.add_argument("--config", type=Path, default=None, help="yaml with 'model_args' (or plain kwargs)")
    p.add_argument("--ckpt", type=Path, default=None, help="state_dict or lightning checkpoint")
    p.add_argument("--ckpt-prefix", default="", help="key prefix to strip, e.g. 'student_model.model_base.'")
    p.add_argument("--ext", default=".wav")
    # reader
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--force-resample", action="store_true", help="resample files with another sample rate")
    p.add_argument("--norm-type", default="std", help="'std' (z-score per file) or peak normalization")
    p.add_argument("--win-ms", type=int, default=2000)
    p.add_argument("--step-ms", type=int, default=1000)
    p.add_argument("--frame-shift", type=float, default=0.01, help="model frame shift, seconds")
    # batching
    p.add_argument("--files-per-batch", type=int, default=8, help="files collated into one loader batch")
    p.add_argument("--batch-size", type=int, default=128, help="max windows per forward pass")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# def load_model(args: argparse.Namespace) -> nn.Module:
#     kwargs: dict = {}
#     if args.config is not None:
#         with open(args.config) as f:
#             cfg = yaml.safe_load(f) or {}
#         kwargs = cfg.get("model_args", cfg)
#     model = ARCHS[args.arch](**kwargs)

#     if args.ckpt is not None:
#         state = torch.load(args.ckpt, map_location="cpu")
#         state = state.get("state_dict", state)
#         if args.ckpt_prefix:
#             state = {k[len(args.ckpt_prefix) :]: v for k, v in state.items() if k.startswith(args.ckpt_prefix)}
#         model.load_state_dict(state)

#     if not hasattr(model, "pooling"):
#         raise ValueError(f"{args.arch} has no 'pooling' module to read attention logits from")
#     return model.eval().to(args.device)


def load_model(args: argparse.Namespace) -> nn.Module:
    model = ECAPA_TDNN_VAD(
        channels=1024,
        feat_type="tf",
        embed_dim=256,
    )

    if args.ckpt is not None:
        state_dict_loaded = torch.load(args.ckpt)
        state_dict_student = model.state_dict()
        for k in state_dict_student:
            state_dict_student[k].copy_(state_dict_loaded["state_dict"][f"student_model.model_base.{k}"])
    return model.eval().to(args.device)


class AttentionCatcher:
    """Stores the attention logits returned by ``model.pooling``."""

    def __init__(self, pooling: nn.Module) -> None:
        self.logits: torch.Tensor | None = None
        self.handle = pooling.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: tuple, output: tuple) -> None:
        self.logits = output[1]


class FileDataset(Dataset):
    """``LabeledSample(sample=AudioSegments, label=file index, dataset_name=path)``."""

    def __init__(self, files: tp.List[Path], reader: AudioReaderFull) -> None:
        self.files = files
        self.reader = reader

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> LabeledSample:
        path = self.files[idx]
        return LabeledSample(sample=self.reader.read(str(path)), label=idx, dataset_name=str(path))


@torch.no_grad()
def forward_windows(
    windows: torch.Tensor,
    model: nn.Module,
    catcher: AttentionCatcher,
    args: argparse.Namespace,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Embeddings ``(N, D)`` and channel-averaged VAD logits ``(N, T)``."""
    embeddings, vad = [], []
    for i in range(0, len(windows), args.batch_size):
        batch = windows[i : i + args.batch_size].float().to(args.device, non_blocking=True)
        emb = model(batch)
        emb = emb.embedding if isinstance(emb, tuple) else emb
        assert catcher.logits is not None
        embeddings.append(emb.float().cpu().numpy())
        vad.append(catcher.logits.float().mean(dim=1).cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(vad)


def pack_file(
    embeddings: np.ndarray,
    window_vad: np.ndarray,
    duration: float,
    args: argparse.Namespace,
) -> tp.Dict[str, np.ndarray]:
    win, step, shift = args.win_ms / 1000, args.step_ms / 1000, args.frame_shift
    starts = np.arange(len(embeddings)) * step

    # Average overlapping window logits on a global frame grid.
    num_frames = int(np.ceil(duration / shift))
    vad_sum = np.zeros(num_frames, dtype=np.float64)
    vad_cnt = np.zeros(num_frames, dtype=np.float64)
    for start, logits in zip(starts, window_vad):
        idx = int(round(start / shift)) + np.arange(len(logits))
        keep = idx < num_frames
        np.add.at(vad_sum, idx[keep], logits[keep])
        np.add.at(vad_cnt, idx[keep], 1.0)
    vad = vad_sum / np.maximum(vad_cnt, 1.0)

    return {
        "embeddings": embeddings.astype(np.float32),
        "window_starts": starts.astype(np.float32),
        "window_ends": np.minimum(starts + win, duration).astype(np.float32),
        "window_vad": window_vad.astype(np.float32),
        "vad": vad.astype(np.float32),
        "frame_shift": np.float32(shift),
        "duration": np.float32(duration),
    }


def main() -> None:
    args = parse_args()
    files = sorted(args.wav_dir.rglob(f"*{args.ext}"))
    out_paths = [(args.out_dir / f.relative_to(args.wav_dir)).with_suffix(".npz") for f in files]
    todo = [i for i, out in enumerate(out_paths) if args.overwrite or not out.exists()]
    if not files:
        raise SystemExit(f"no *{args.ext} files in {args.wav_dir}")
    if not todo:
        return
    files, out_paths = [files[i] for i in todo], [out_paths[i] for i in todo]

    reader = AudioReaderFull(
        norm_type=args.norm_type,
        length_segment_ms=args.win_ms,
        segments_step_ms=args.step_ms,
        sample_rate=args.sample_rate,
        force_resample=args.force_resample,
    )
    loader = DataLoader(
        FileDataset(files, reader),
        batch_size=args.files_per_batch,
        num_workers=args.num_workers,
        collate_fn=collate_batch_labeled_segments_fn,
        pin_memory=args.device.startswith("cuda"),
    )

    model = load_model(args)
    catcher = AttentionCatcher(model.pooling)

    with tqdm(total=len(files), desc="extract") as progress:
        for batch in loader:
            embeddings, window_vad = forward_windows(batch.segments, model, catcher, args)
            segment_to_sample = batch.segment_to_sample.numpy()
            for sample_idx, file_idx in enumerate(batch.labels.tolist()):
                rows = segment_to_sample == sample_idx
                duration = sf.info(str(files[file_idx])).duration
                result = pack_file(embeddings[rows], window_vad[rows], duration, args)
                out_paths[file_idx].parent.mkdir(parents=True, exist_ok=True)
                np.savez(out_paths[file_idx], **result)
                progress.update()


if __name__ == "__main__":
    main()
