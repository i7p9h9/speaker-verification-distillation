"""Single-step extraction of frame-level speaker features and attention-based VAD logits.

Implements the extraction step of Thienpondt and Demuynck, "Speaker Embeddings
With Weakly Supervised Voice Activity Detection For Efficient Speaker
Diarization" (Odyssey 2024), Figure 1 and Section 4: a sliding window (2 s,
step 1 s) is run over every file, and one forward pass gives

* the frame-level features ``H`` (``C x T``, the input of the attentive pooling),
* the attention responses ``E`` (``C x T``), averaged over channels into the
  frame-level VAD logits ``v_t`` (equation (5)),
* the window embedding ``d``.

``H`` and ``v_t`` of overlapping windows are averaged onto a global frame grid.
Its step is the model frame shift (10 ms for ECAPA, 80 ms for ResNetTF; measured
on one window unless ``--frame-shift`` is given) or, if the window step is not a
multiple of it, the gcd of both: 80 ms frames with a 1 s step give a 40 ms grid
with every frame repeated twice (``--step-ms 960`` keeps 80 ms).

Files are read and windowed by :class:`AudioReaderFull` in DataLoader workers;
windows of several files are collated into one batch.  The reader pads the last
partial window by tiling, so frames past the end of the file are dropped.

Works with any model that has a ``pooling`` submodule taking ``H`` and returning
``(pooled, attention_logits)`` with logits of shape ``(B, C, T)``, i.e.
:class:`ECAPA2`, :class:`ECAPA_TDNN_VAD`, the original ECAPA2 (``ecapa2_orig.py``) and
``ResNetTF`` with ``SelfAttentionTF`` pooling (``resnettf_vad.py``, ``--config``
is its yaml with ``model_args``).  ``H`` and ``E`` are captured with a
forward hook, so the model's ``forward`` may return anything with the embedding
first (a tensor or :class:`ECAPA2Output`).

For every audio file ``<wav-dir>/<rel>.wav`` an ``<out-dir>/<rel>.npz`` is
written with:

    frames          (N, C) float16  overlap-averaged frame-level features H,
                                    frame i at i * frame_shift (--frames-dtype);
                                    only with --save-frames
    vad             (N,)   float32  overlap-averaged VAD logits v_t
    embeddings      (W, D) float32  window embeddings
    window_starts   (W,)   float32  window start times, seconds
    window_ends     (W,)   float32  window end times (clipped to the file), seconds
    window_vad      (W, T) float32  per-window VAD logits, model frames
    frame_shift     ()     float32  grid step, seconds
    duration        ()     float32  seconds

In float16 ``frames`` takes about 1.1 GB per hour of audio at C = 1536 on a
10 ms grid, 0.55 GB at C = 3072 on a 40 ms grid, so it is written only with
``--save-frames``; it is needed by the LDA VAD (``lda_vad.py``, ``diarize.py
--vad-lda``) only.

Usage:
    python egs/diarization/ecapa/extract.py --wav-dir data/wav --out-dir data/npz \
        --arch ECAPA2 --config cfg.yaml --ckpt model.ckpt --device cuda
"""

from __future__ import annotations

import argparse
import math
import typing as tp
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from ecapa2_orig import OriginalECAPA2
from resnettf_vad import ResNetTFVAD
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--arch",
        choices=sorted([*ARCHS, "ECAPA2_ORIG", "ResNetTF"]),
        default="ECAPA2",
        help="ECAPA2_ORIG: pre-trained Jenthe/ECAPA2, --ckpt is its ecapa2.pt (default: HF download); "
        "ResNetTF: needs --config (yaml with model_args)",
    )
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
    p.add_argument(
        "--frame-shift",
        type=float,
        default=None,
        help="model frame shift, seconds (default: from the output length of one window, to 10 ms)",
    )
    # batching
    p.add_argument("--files-per-batch", type=int, default=8, help="files collated into one loader batch")
    p.add_argument("--batch-size", type=int, default=128, help="max windows per forward pass")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--frames-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="storage type of the frame-level features",
    )
    p.add_argument("--save-frames", action="store_true", help="also write the frame-level features H (large)")
    p.add_argument("--overwrite", action="store_true")
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


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
    if args.arch == "ECAPA2_ORIG":
        return OriginalECAPA2(args.ckpt).eval().to(args.device)
    if args.arch == "ResNetTF":
        if args.config is None:
            raise SystemExit("--arch ResNetTF needs --config")
        return ResNetTFVAD.from_config(args.config, args.ckpt, args.ckpt_prefix).eval().to(args.device)

    model = ECAPA_TDNN_VAD(
        channels=1024,
        feat_type="tf",
        embed_dim=256,
        is_use_std_pool=False,
        is_use_pooling_global_context=False,
        is_channel_dependent_attention=True,
    )

    # exp-003
    # is_use_std_pool=False,
    # is_use_pooling_global_context=False,
    # is_channel_dependent_attention=True,

    # exp-004
    # is_use_std_pool=False,
    # is_use_pooling_global_context=True,
    # is_channel_dependent_attention=True,

    # exp-005
    # is_use_std_pool=False,
    # is_use_pooling_global_context=False,
    # is_channel_dependent_attention=False,

    if args.ckpt is not None:
        state_dict_loaded = torch.load(args.ckpt)
        state_dict_student = model.state_dict()
        for k in state_dict_student:
            state_dict_student[k].copy_(state_dict_loaded["state_dict"][f"student_model.model_base.{k}"])
    return model.eval().to(args.device)


class PoolingCatcher:
    """Stores the input of ``model.pooling`` (frame-level features ``H``) and
    the attention logits ``E`` it returns, both ``(B, C, T)``."""

    def __init__(self, pooling: nn.Module) -> None:
        self.frames: torch.Tensor | None = None
        self.logits: torch.Tensor | None = None
        self.handle = pooling.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: tuple, output: tuple) -> None:
        self.frames = inputs[0]
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


class FileAccumulator:
    """Overlap-averages window outputs of one file onto its global frame grid.

    The grid step ``args.frame_shift`` divides both the model frame shift and the
    window step; every model frame covers ``args.frame_repeat`` grid frames.
    Frame ``i`` of window ``w`` lands at grid index ``w * step / shift + i``;
    frames past the end of the file (tiled padding of the last window) are dropped.
    """

    def __init__(self, num_windows: int, duration: float, args: argparse.Namespace) -> None:
        self.duration = duration
        self.args = args
        self.num_frames = int(np.ceil(duration / args.frame_shift))
        self.offset = int(round(args.step_ms / 1000 / args.frame_shift))
        self.embeddings: tp.List[np.ndarray | None] = [None] * num_windows
        self.window_vad: tp.List[np.ndarray | None] = [None] * num_windows
        self.frames_sum: np.ndarray | None = None  # (N, C), allocated on the first window
        self.vad_sum = np.zeros(self.num_frames, dtype=np.float64)
        self.count = np.zeros(self.num_frames, dtype=np.float64)

    def add(self, window: int, embedding: np.ndarray, frames: np.ndarray | None, vad: np.ndarray) -> None:
        """``embedding (D,)``, ``frames (C, T)`` (None without --save-frames), ``vad (T,)``
        of window number ``window``."""
        self.embeddings[window] = embedding
        self.window_vad[window] = vad
        if self.args.frame_repeat > 1:
            frames = None if frames is None else np.repeat(frames, self.args.frame_repeat, axis=1)
            vad = np.repeat(vad, self.args.frame_repeat)
        if frames is not None and self.frames_sum is None:
            self.frames_sum = np.zeros((self.num_frames, frames.shape[0]), dtype=np.float32)
        begin = window * self.offset
        end = min(begin + len(vad), self.num_frames)
        if end <= begin:
            return
        if frames is not None:
            self.frames_sum[begin:end] += frames[:, : end - begin].T
        self.vad_sum[begin:end] += vad[: end - begin]
        self.count[begin:end] += 1.0

    def result(self) -> tp.Dict[str, np.ndarray]:
        assert all(e is not None for e in self.embeddings)
        win, step = self.args.win_ms / 1000, self.args.step_ms / 1000
        starts = np.arange(len(self.embeddings)) * step
        count = np.maximum(self.count, 1.0)
        if self.frames_sum is not None:
            self.frames_sum /= count[:, None].astype(np.float32)
        vad = self.vad_sum / count
        # A window yields one frame less than its length, so the last frame of a
        # file can be left uncovered; copy the nearest covered frame there.
        covered = np.flatnonzero(self.count > 0)
        if 0 < len(covered) < self.num_frames:
            nearest = covered[np.clip(np.searchsorted(covered, np.arange(self.num_frames)), 0, len(covered) - 1)]
            if self.frames_sum is not None:
                self.frames_sum = self.frames_sum[nearest]
            vad = vad[nearest]
        out = {
            "embeddings": np.stack(self.embeddings).astype(np.float32),
            "window_starts": starts.astype(np.float32),
            "window_ends": np.minimum(starts + win, self.duration).astype(np.float32),
            "window_vad": np.stack(self.window_vad).astype(np.float32),
            "vad": vad.astype(np.float32),
            "frame_shift": np.float32(self.args.frame_shift),
            "duration": np.float32(self.duration),
        }
        if self.frames_sum is not None:
            out["frames"] = self.frames_sum.astype(self.args.frames_dtype)
        return out


@torch.no_grad()
def forward_windows(
    windows: torch.Tensor,
    model: nn.Module,
    catcher: PoolingCatcher,
    args: argparse.Namespace,
) -> tp.Iterator[tp.Tuple[int, np.ndarray, np.ndarray | None, np.ndarray]]:
    """Yields ``(first window, embeddings (B, D), frames H (B, C, T) or None without
    --save-frames, VAD logits (B, T))`` per chunk of ``--batch-size`` windows, so
    that frames are never held for a whole batch."""
    for i in range(0, len(windows), args.batch_size):
        batch = windows[i : i + args.batch_size].float().to(args.device, non_blocking=True)
        out = model(batch)
        emb = out.embedding if isinstance(out, tuple) else out
        assert catcher.frames is not None and catcher.logits is not None
        yield (
            i,
            emb.float().cpu().numpy(),
            catcher.frames.float().cpu().numpy() if args.save_frames else None,
            catcher.logits.float().mean(dim=1).cpu().numpy(),
        )


@torch.no_grad()
def set_frame_grid(model: nn.Module, catcher: PoolingCatcher, args: argparse.Namespace) -> None:
    """Sets ``args.frame_shift`` (overlap-averaging grid) and ``args.frame_repeat``.

    The model frame shift is ``--frame-shift`` or measured on one window.  The
    grid step is the gcd of it and ``--step-ms`` in ms, so windows land on the
    grid exactly (e.g. 80 ms frames with a 1000 ms step give a 40 ms grid).
    """
    if args.frame_shift is None:
        model(torch.randn(1, args.sample_rate * args.win_ms // 1000, device=args.device))
        num_frames = catcher.logits.shape[-1]
        args.frame_shift = round(args.win_ms / num_frames / 10) / 100
    model_ms = round(args.frame_shift * 1000)
    grid_ms = math.gcd(model_ms, args.step_ms)
    args.frame_repeat = model_ms // grid_ms
    args.frame_shift = grid_ms / 1000
    print(f"model frame shift {model_ms} ms, grid {grid_ms} ms")


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
    catcher = PoolingCatcher(model.pooling)
    set_frame_grid(model, catcher, args)

    with tqdm(total=len(files), desc="extract") as progress:
        for batch in loader:
            segment_to_sample = batch.segment_to_sample.numpy()
            file_ids = batch.labels.tolist()
            # Window number of every segment within its own file.
            first = {s: int(np.argmax(segment_to_sample == s)) for s in range(len(file_ids))}
            accumulators = [
                FileAccumulator(int((segment_to_sample == s).sum()), sf.info(str(files[file_idx])).duration, args)
                for s, file_idx in enumerate(file_ids)
            ]
            for offset, embeddings, frames, vad in forward_windows(batch.segments, model, catcher, args):
                for j in range(len(embeddings)):
                    sample = int(segment_to_sample[offset + j])
                    window = offset + j - first[sample]
                    accumulators[sample].add(window, embeddings[j], None if frames is None else frames[j], vad[j])
            for acc, file_idx in zip(accumulators, file_ids):
                out_paths[file_idx].parent.mkdir(parents=True, exist_ok=True)
                # write + rename, so an interrupted run never leaves a truncated .npz
                tmp = out_paths[file_idx].with_suffix(".npz.tmp")
                with open(tmp, "wb") as f:
                    np.savez(f, **acc.result())
                tmp.replace(out_paths[file_idx])
                progress.update()


if __name__ == "__main__":
    main()
