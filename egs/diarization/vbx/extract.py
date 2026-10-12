"""x-vectors of the speech regions of a directory of audio files, as in VBx.

    python egs/diarization/vbx/extract.py --wav-dir WAVS --out-dir XVEC \
        --model resnet101-16k --vad-silero SILERO_DIR [--vad-silero-prob --onset 0.5 ...]

For every ``<wav-dir>/<rel>.wav`` writes ``<out-dir>/<rel>.npz`` with

    embeddings  (N, D) float32  x-vectors
    starts      (N,)   float64  window start, s
    ends        (N,)   float64  window end, s
    speech      (K, 2) float64  the speech regions they were extracted from, s

The windows follow the reference (``VBx/predict.py``) per speech region:

* ``fbank`` models (ResNet101): 1.44 s windows (``--seg-len`` frames) every
  0.24 s (``--seg-jump``), plus a last shorter window of at least 0.1 s;
  regions up to 10 ms are skipped.  The window times are rounded to ms.
* ``pcm`` models (redimnet): windows of the model's input size (1 s) every
  ``--pcm-hop`` s, each normalized to zero mean and unit variance; regions
  shorter than one window are skipped (no x-vector, so no speaker label).

Audio is mixed to mono and resampled (resampy, ``kaiser_best``) to the model's
rate if needed.  The ONNX models run in torch (``onnx_torch.py``), batched, in
fp32 without TF32.  Existing outputs are kept unless ``--overwrite``.

Speech comes from one of ``--vad-silero DIR`` (Silero segments, or with
``--vad-silero-prob`` its probabilities thresholded by ``--onset`` ``--offset``
``--merge-gap`` ``--min-dur``), ``--vad-rttm-dir`` (oracle) or ``--vad-lab-dir``.
"""

from __future__ import annotations

import argparse
import typing as tp
from pathlib import Path

import numpy as np
import resampy
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import vad
    from features import Fbank, add_dither
    from models import PRESETS, preset
    from onnx_torch import input_shape, load_onnx
except ImportError:
    from . import vad
    from .features import Fbank, add_dither
    from .models import PRESETS, preset
    from .onnx_torch import input_shape, load_onnx


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    p.add_argument("--wav-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ext", default=".wav")
    p.add_argument("--model", choices=sorted(PRESETS), default=None, help="preset, see models.py")
    p.add_argument("--weights", type=Path, default=None, help="ONNX extractor (instead of --model)")
    p.add_argument("--frontend", choices=["fbank", "pcm"], default=None, help="with --weights")
    p.add_argument("--sample-rate", type=int, default=None, help="with --weights")
    p.add_argument("--seg-len", type=int, default=144, help="fbank: window, frames")
    p.add_argument("--seg-jump", type=int, default=24, help="fbank: hop, frames")
    p.add_argument("--pcm-hop", type=float, default=0.24, help="pcm: hop, s")
    vad.add_arguments(p)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    return p


def parse_args(argv: tp.List[str] | None = None) -> argparse.Namespace:
    p = build_parser()
    args = p.parse_args(argv)
    vad.check_arguments(p, args)
    if (args.model is None) == (args.weights is None):
        p.error("give --model or --weights")
    if args.model is not None:
        cfg = preset(args.model)
        args.weights, args.frontend, args.sample_rate = cfg["weights"], cfg["frontend"], cfg["sample_rate"]
    elif args.frontend is None or args.sample_rate is None:
        p.error("--weights needs --frontend and --sample-rate")
    return args


# ---------------------------------------------------------------------------
# Audio -> windows (in DataLoader workers)
# ---------------------------------------------------------------------------
class Item(tp.NamedTuple):
    out: Path
    speech: np.ndarray  # (K, 2) s
    windows: np.ndarray  # fbank: (N, seg_len, 64) features; pcm: (N, L) normalized audio
    tails: tp.List[np.ndarray]  # fbank: shorter last windows, (T_i, 64)
    starts: np.ndarray  # (N + len(tails),), windows then tails
    ends: np.ndarray
    order: np.ndarray  # position of each (window or tail) in the reference order


def read_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=True)
    signal = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if sr != sample_rate:
        signal = resampy.resample(signal, sr, sample_rate, filter="kaiser_best")
    return signal


def fbank_windows(signal: np.ndarray, sr: int, speech: np.ndarray, args: argparse.Namespace) -> Item:
    """``extract_xvectors_melbank``; windows in the reference order are listed
    as (kind, index) to restore it after the full windows were batched."""
    fbank = Fbank(sr)
    labs = (speech * sr).astype(int)
    np.random.seed(3)  # for reproducibility
    signal = add_dither((signal * 2**15).astype(int))
    seg_len, seg_jump = args.seg_len, args.seg_jump

    windows, tails, w_times, t_times, order = [], [], [], [], []
    for lab_start, lab_end in labs:
        seg = signal[lab_start:lab_end]
        if seg.shape[0] <= 0.01 * sr:
            continue
        fea = fbank(seg)
        slen = len(fea)
        start = -seg_jump
        for start in range(0, slen - seg_len, seg_jump):
            windows.append(fea[start : start + seg_len])
            begin = round(lab_start / float(sr) + start / 100.0, 3)
            w_times.append((begin, round(lab_start / float(sr) + start / 100.0 + seg_len / 100.0, 3)))
            order.append((0, len(windows) - 1))
        if slen - start - seg_jump >= 10:
            tails.append(fea[start + seg_jump : slen])
            t_times.append((round(lab_start / float(sr) + (start + seg_jump) / 100.0, 3), round(lab_end / float(sr), 3)))
            order.append((1, len(tails) - 1))
    windows_arr = np.stack(windows) if windows else np.zeros((0, seg_len, 64), np.float32)
    times = np.array(w_times + t_times, dtype=np.float64).reshape(-1, 2)
    # position in the concatenation [windows, tails] of the i-th window in reference order
    concat = np.array([i if kind == 0 else len(windows) + i for kind, i in order], dtype=int)
    return Item(Path(), speech, windows_arr, tails, times[:, 0], times[:, 1], concat)


def pcm_windows(signal: np.ndarray, sr: int, speech: np.ndarray, length: int, args: argparse.Namespace) -> Item:
    """``extract_xvectors_pcm``."""
    labs = (speech * sr).astype(int)
    hop = int(args.pcm_hop * sr)
    windows, times = [], []
    for lab_start, lab_end in labs:
        seg = signal[lab_start:lab_end]
        if seg.shape[0] < length:
            continue
        offset = lab_start / float(sr)
        for start in range(0, len(seg) - length + 1, hop):
            chunk = seg[start : start + length]
            std = np.std(chunk)
            windows.append((chunk - np.mean(chunk)) / std if std != 0 else chunk)
            times.append((round(offset + start / float(sr), 3), round(offset + (start + length) / float(sr), 3)))
    windows_arr = np.stack(windows).astype(np.float32) if windows else np.zeros((0, length), np.float32)
    times_arr = np.array(times, dtype=np.float64).reshape(-1, 2)
    return Item(Path(), speech, windows_arr, [], times_arr[:, 0], times_arr[:, 1], np.arange(len(windows)))


class Files(Dataset):
    def __init__(self, pairs: tp.List[tp.Tuple[Path, Path]], args: argparse.Namespace, length: int) -> None:
        self.pairs, self.args, self.length = pairs, args, length
        self.speech: vad.Speech | None = None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Item:
        if self.speech is None:
            self.speech = vad.Speech(self.args)
        wav, out = self.pairs[i]
        speech = np.array(self.speech(wav.stem), dtype=np.float64).reshape(-1, 2)
        sr = self.args.sample_rate
        signal = read_audio(wav, sr)
        if self.args.frontend == "fbank":
            item = fbank_windows(signal, sr, speech, self.args)
        else:
            item = pcm_windows(signal, sr, speech, self.length, self.args)
        return item._replace(out=out)


def _identity(item: Item) -> Item:
    return item


# ---------------------------------------------------------------------------
# Windows -> x-vectors
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed(model: torch.nn.Module, item: Item, args: argparse.Namespace) -> np.ndarray:
    def run(x: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(x)).to(args.device)
        if args.frontend == "fbank":
            x = x.transpose(1, 2)  # (B, 64, T)
        return model(x).reshape(len(x), -1).float().cpu().numpy()

    out = [run(item.windows[i : i + args.batch_size]) for i in range(0, len(item.windows), args.batch_size)]
    out += [run(tail[None]) for tail in item.tails]
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def save(item: Item, emb: np.ndarray) -> None:
    emb, starts, ends = emb[item.order], item.starts[item.order], item.ends[item.order]
    ok = ~np.isnan(emb).any(axis=1) if emb.size else np.ones(len(starts), dtype=bool)
    if not ok.all():
        print(f"{item.out.stem}: {int((~ok).sum())} NaN x-vectors dropped")
    item.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = item.out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, embeddings=emb[ok].astype(np.float32), starts=starts[ok], ends=ends[ok], speech=item.speech)
    tmp.replace(item.out)


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    wavs = sorted(args.wav_dir.rglob(f"*{args.ext}"))
    pairs = [(w, (args.out_dir / w.relative_to(args.wav_dir)).with_suffix(".npz")) for w in wavs]
    speech = vad.Speech(args)
    missing = [w for w, _ in pairs if not speech.has(w.stem)]
    if missing:
        print(f"{len(missing)} files without VAD skipped: {' '.join(w.stem for w in missing[:5])}{' ...' * (len(missing) > 5)}")
    pairs = [(w, o) for w, o in pairs if speech.has(w.stem)]
    todo = [(w, o) for w, o in pairs if args.overwrite or not o.exists()]
    print(f"{len(pairs)} files, {len(pairs) - len(todo)} done, model {args.model or args.weights}")
    if not todo:
        return
    length = input_shape(args.weights)[-1]
    if args.frontend == "pcm" and not isinstance(length, int):
        raise SystemExit(f"{args.weights}: pcm model without a fixed input length")
    model = load_onnx(args.weights).to(args.device)

    loader = DataLoader(
        Files(todo, args, length if isinstance(length, int) else 0),
        batch_size=None,
        collate_fn=_identity,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers else None,
        multiprocessing_context="spawn" if args.num_workers else None,
    )
    for item in tqdm(loader, total=len(todo), desc="x-vectors"):
        save(item, embed(model, item, args))


if __name__ == "__main__":
    main()
