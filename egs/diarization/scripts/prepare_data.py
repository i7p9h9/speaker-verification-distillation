"""Bring a diarization dataset into the common layout of the recipes (ecapa, vbx).

    python egs/diarization/scripts/prepare_data.py DATASET ROOT [--force] [--channel 0] [--workers 8]

writes, next to the data,

    <root>/prepared/<set>/wav/<id>.wav    symlink to the audio
    <root>/prepared/<set>/rttm/<id>.rttm  reference with file id <id>
    <root>/prepared/audio/<id>.wav        mono 16-bit copies of multichannel / non-wav audio
    <root>/prepared/.ready                json: version, dataset, set sizes

``<id>`` is the audio file name without extension (dots replaced by ``_``), and
every RTTM is rewritten with it, so references with other or broken file ids
(VoxConverse ``<file_id>``, AliMeeting) score against the outputs directly.
Only recordings with both audio and reference go into a set; sets without any
are left out (e.g. the AMI dev/train meetings without audio).  The layout is
built in ``prepared.tmp`` and renamed, so ``.ready`` exists only once it is
complete; nothing outside ``prepared/`` is changed.  A ready dataset is left as
is unless ``--force``.

Multichannel audio (AISHELL-4, AliMeeting: 8-channel arrays) is reduced to
``--channel`` (an index, default 0, or ``mean``).

Datasets (ROOT is the directory with the listed contents):

    ami           rttm/{dev,train,test}/<meeting>.rttm, wav/<meeting>/audio/<meeting>.<stream>.wav
                  (--ami-stream, default Mix-Headset); sets dev, train, test
    callhome      meta.csv (path,start_sec,speaker,end_sec), data_converted/*.wav; set all
    callhome2     sre2000-key/fullref.rttm (the Kaldi callhome_diarization key; the rttm/
                  dir of the archive misses the first turn of every file), wav/*.wav,
                  lists/part2.txt; sets part1, part2 (the Kaldi halves), all
    voxconverse   rttm/*.rttm, wav/*.wav (test), or the original layout
                  {dev,test}/audio/*.wav + voxconverse/{dev,test}/*.rttm; sets test (, dev)
    voxconverse8k meta.csv, voxconverse_test_wav_8k/audio/*.wav; set test
    aishell4      rttm/<id>.rttm, wav/<id>.flac, lists/<set>.txt; sets of the lists with audio
    alimeeting    rttm/<session>.rttm, wav/<session>_<mic>.wav; set test
    msdwild       rttm/<id>.rttm, wav/<id>.wav, lists/<set>.txt; sets of the lists with audio
    icsi          meta.csv, Signals_8k_cut/<meeting>/<meeting>_<k>.interaction.wav; set all
    robusta       meta.csv, wavs_8k/*.wav; set all
    switchboard   meta.csv, NIST2000-Disk6-Diarization/.../data_converted/*.wav; set all
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import typing as tp
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

VERSION = 1
PREPARED = "prepared"
READY = ".ready"

# (start_sec, end_sec, speaker)
Turn = tp.Tuple[float, float, str]


class Rec(tp.NamedTuple):
    id: str
    audio: Path
    turns: tp.List[Turn]


Sets = tp.Dict[str, tp.List[Rec]]


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def file_id(path: Path | str) -> str:
    name = Path(path).name
    return name[: -len(Path(name).suffix)].replace(".", "_")


def read_rttm(path: Path) -> tp.Dict[str, tp.List[Turn]]:
    """``{file_id: turns}`` of the SPEAKER lines."""
    turns: tp.Dict[str, tp.List[Turn]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                start = float(parts[3])
                turns[parts[1]].append((start, start + float(parts[4]), parts[7]))
    return dict(turns)


def rttm_turns(path: Path) -> tp.List[Turn]:
    """All turns of one RTTM file, whatever file id it gives."""
    return [t for turns in read_rttm(path).values() for t in turns]


def read_meta(root: Path) -> tp.Dict[Path, tp.List[Turn]]:
    """``meta.csv`` (columns path, start_sec, end_sec, speaker) -> {audio: turns}."""
    turns: tp.Dict[Path, tp.List[Turn]] = defaultdict(list)
    with open(root / "meta.csv", newline="") as f:
        for row in csv.DictReader(f):
            turns[root / row["path"]].append((float(row["start_sec"]), float(row["end_sec"]), row["speaker"]))
    return dict(turns)


def read_list(path: Path) -> tp.List[str]:
    return [line.strip() for line in open(path) if line.strip()]


def from_meta(root: Path, set_name: str) -> Sets:
    return {set_name: [Rec(file_id(a), a, t) for a, t in read_meta(root).items()]}


def from_rttm_dir(rttm_dir: Path, audio: tp.Callable[[str], Path]) -> tp.List[Rec]:
    return [Rec(p.stem, audio(p.stem), rttm_turns(p)) for p in sorted(rttm_dir.glob("*.rttm"))]


def from_lists(root: Path, recs: tp.List[Rec]) -> Sets:
    by_id = {r.id: r for r in recs}
    return {
        p.stem: [by_id[i] for i in read_list(p) if i in by_id] for p in sorted((root / "lists").glob("*.txt"))
    }


# ---------------------------------------------------------------------------
# Datasets: root -> {set: recordings} (audio may be missing, filtered later)
# ---------------------------------------------------------------------------
def ami(root: Path, args: argparse.Namespace) -> Sets:
    stream = args.ami_stream
    return {
        d.name: from_rttm_dir(d, lambda m: root / "wav" / m / "audio" / f"{m}.{stream}.wav")
        for d in sorted((root / "rttm").iterdir())
        if d.is_dir()
    }


def callhome(root: Path, args: argparse.Namespace) -> Sets:
    return from_meta(root, "all")


def callhome2(root: Path, args: argparse.Namespace) -> Sets:
    ref = read_rttm(root / "sre2000-key" / "fullref.rttm")
    recs = [Rec(i, root / "wav" / f"{i}.wav", t) for i, t in sorted(ref.items())]
    part2 = set(read_list(root / "lists" / "part2.txt"))
    return {
        "part1": [r for r in recs if r.id not in part2],
        "part2": [r for r in recs if r.id in part2],
        "all": recs,
    }


def voxconverse(root: Path, args: argparse.Namespace) -> Sets:
    if (root / "rttm").is_dir():
        return {"test": from_rttm_dir(root / "rttm", lambda i: root / "wav" / f"{i}.wav")}
    return {
        s: from_rttm_dir(root / "voxconverse" / s, lambda i, s=s: root / s / "audio" / f"{i}.wav")
        for s in ("dev", "test")
    }


def voxconverse8k(root: Path, args: argparse.Namespace) -> Sets:
    return from_meta(root, "test")


def aishell4(root: Path, args: argparse.Namespace) -> Sets:
    return from_lists(root, from_rttm_dir(root / "rttm", lambda i: root / "wav" / f"{i}.flac"))


def alimeeting(root: Path, args: argparse.Namespace) -> Sets:
    # wav/R8002_M8002_MS802.wav <-> rttm/R8002_M8002.rttm
    audio = {"_".join(p.stem.split("_")[:2]): p for p in (root / "wav").glob("*.wav")}
    recs = from_rttm_dir(root / "rttm", lambda i: audio.get(i, root / "wav" / f"{i}.wav"))
    return {"test": recs}


def msdwild(root: Path, args: argparse.Namespace) -> Sets:
    return from_lists(root, from_rttm_dir(root / "rttm", lambda i: root / "wav" / f"{i}.wav"))


def icsi(root: Path, args: argparse.Namespace) -> Sets:
    return from_meta(root, "all")


def robusta(root: Path, args: argparse.Namespace) -> Sets:
    return from_meta(root, "all")


def switchboard(root: Path, args: argparse.Namespace) -> Sets:
    return from_meta(root, "all")


DATASETS: tp.Dict[str, tp.Callable[[Path, argparse.Namespace], Sets]] = {
    f.__name__: f
    for f in (
        ami,
        callhome,
        callhome2,
        voxconverse,
        voxconverse8k,
        aishell4,
        alimeeting,
        msdwild,
        icsi,
        robusta,
        switchboard,
    )
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def needs_copy(path: Path) -> bool:
    return path.suffix.lower() != ".wav" or sf.info(str(path)).channels > 1


def to_mono(job: tp.Tuple[Path, Path, str]) -> None:
    src, dst, channel = job
    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1) if channel == "mean" else audio[:, int(channel)]
    tmp = dst.with_name(dst.name + ".tmp")
    sf.write(str(tmp), np.clip(mono, -1, 1), sr, subtype="PCM_16", format="WAV")
    tmp.replace(dst)


def write_rttm(path: Path, rec_id: str, turns: tp.Iterable[Turn]) -> None:
    with open(path, "w") as f:
        for start, end, spk in sorted(turns):
            spk = "_".join(spk.split())
            f.write(f"SPEAKER {rec_id} 1 {start:.6f} {end - start:.6f} <NA> <NA> {spk} <NA> <NA>\n")


def prepare(name: str, root: Path, args: argparse.Namespace) -> tp.Dict[str, int]:
    sets = DATASETS[name](root, args)
    out = root / PREPARED
    tmp = root / (PREPARED + ".tmp")
    # converted audio of an interrupted run is reused
    audio_dir = tmp / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for d in tmp.iterdir():
        if d != audio_dir:
            shutil.rmtree(d) if d.is_dir() else d.unlink()

    # recordings with audio and a non-empty reference, the same id once per dataset
    kept: Sets = {}
    audio_of: tp.Dict[str, Path] = {}
    for set_name, recs in sets.items():
        ok = []
        for r in recs:
            turns = [t for t in r.turns if t[1] > t[0]]
            if not r.audio.is_file() or not turns:
                continue
            if audio_of.setdefault(r.id, r.audio) != r.audio:
                raise SystemExit(f"{name}: id {r.id} of both {audio_of[r.id]} and {r.audio}")
            ok.append(r._replace(turns=turns))
        print(f"{set_name}: {len(ok)} of {len(recs)} recordings with audio and reference")
        if ok:
            kept[set_name] = ok

    ids = {r.id for recs in kept.values() for r in recs}
    sources = {i: a.resolve() for i, a in audio_of.items() if i in ids}
    with mp.Pool(args.workers) as pool:
        copy = dict(zip(sources, pool.map(needs_copy, [sources[i] for i in sources], chunksize=16)))
        jobs = [(sources[i], audio_dir / f"{i}.wav", args.channel) for i in sources if copy[i]]
        jobs = [j for j in jobs if not j[1].exists()]
        if jobs:
            print(f"{len(jobs)} files to mono wav (channel {args.channel})")
            pool.map(to_mono, jobs, chunksize=1)

    for set_name, recs in kept.items():
        (tmp / set_name / "wav").mkdir(parents=True)
        (tmp / set_name / "rttm").mkdir()
        for r in recs:
            target = (out / "audio" / f"{r.id}.wav") if copy[r.id] else sources[r.id]
            os.symlink(target, tmp / set_name / "wav" / f"{r.id}.wav")
            write_rttm(tmp / set_name / "rttm" / f"{r.id}.rttm", r.id, r.turns)
    if not any(audio_dir.iterdir()):
        audio_dir.rmdir()

    sizes = {s: len(recs) for s, recs in kept.items()}
    info = {"version": VERSION, "dataset": name, "channel": args.channel, "sets": sizes}
    (tmp / READY).write_text(json.dumps(info, indent=1) + "\n")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return sizes


def ready(root: Path) -> bool:
    try:
        return json.loads((root / PREPARED / READY).read_text())["version"] == VERSION
    except (OSError, ValueError, KeyError):
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=sorted(DATASETS))
    p.add_argument("root", type=Path)
    p.add_argument("--force", action="store_true", help="prepare again even if ready")
    p.add_argument("--channel", default="0", help="channel of multichannel audio: index or 'mean'")
    p.add_argument("--ami-stream", default="Mix-Headset")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.channel != "mean" and not args.channel.isdigit():
        p.error("--channel: an index or 'mean'")
    root = args.root.resolve()
    if not args.force and ready(root):
        print(f"{root / PREPARED}: ready")
        return
    sizes = prepare(args.dataset, root, args)
    if not sizes:
        raise SystemExit(f"{args.dataset}: no recording with audio and reference in {root}")
    print(f"{root / PREPARED}: " + ", ".join(f"{s} {n}" for s, n in sizes.items()))


if __name__ == "__main__":
    main()
