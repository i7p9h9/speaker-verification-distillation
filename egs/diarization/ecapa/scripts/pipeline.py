"""extract.py -> diarize.py -> score_der.py for several sets, re-running only what changed.

    pipeline.py --exp-dir DIR [--dataset NAME] --set NAME WAV_DIR [REF_DIR] [--set ...]
                [--stages extract,diarize,score] [--force] [--collars 0 0.25] [--skip-overlap]
                [--oracle-vad] [--oracle-speakers] [--silero] [--silero-root DIR] [--tag NAME]
                [extract.py / diarize.py options]

Options of ``extract.py`` and ``diarize.py`` are given directly and routed to
the script that has them (``--ckpt``, ``--arch``, ``--batch-size`` -> extract;
``--onset``, ``--offset``, ``--num-speakers`` -> diarize).  Paths that the
pipeline sets (``--wav-dir``, ``--out-dir``, ``--npz-dir``, ``--overwrite``)
are not accepted.

Layout, per set (``<root> = <exp-dir>/<dataset>``, or ``<exp-dir>`` without ``--dataset``):

    <root>/<set>/npz/*.npz               extract.py output
    <root>/<set>/npz/.config.json        its parameters
    <root>/<set>/silero/*.npz            Silero VAD cache (with --silero; or <silero-root>/<set>/silero)
    <root>/<set>/diar-<hash>/rttm/       diarize.py output, one dir per parameter set
    <root>/<set>/diar-<hash>/config.json its parameters (the hash covers them and the npz)
    <root>/<set>/diar-<hash>/scores.csv  per-file DER and VAD metrics (scores-skip-overlap.csv)
    <root>/results.tsv                   one line per (set, diar-<hash>, scoring, tag), all runs so far

A stage is skipped when it has completed with the same parameters.  Files given
as parameters (``--ckpt``, ``--config``, ``--vad-lda``) are compared by md5, so
an updated ``last.ckpt`` triggers a new extraction; options that do not change
the result (batch sizes, workers, device) are ignored.  The frame features
are written only with ``--save-frames`` (needed by ``--vad-lda``).  An interrupted
extraction with unchanged parameters resumes; with changed ones it starts over.
Scoring is cheap and always runs for sets with a reference.

``--oracle-vad`` / ``--oracle-speakers`` pass the set's reference to
``diarize.py`` as ``--oracle-vad-dir`` / ``--oracle-speakers-dir``.

``--silero`` diarizes with the Silero VAD segments (``--vad-silero``; add
``--vad-silero-prob`` to threshold its probabilities with ``--onset`` etc.).
Silero runs only for the files missing from ``<set>/silero``, i.e. once per set
(it depends on the audio only); ``--silero-root DIR`` keeps the caches in
``DIR/<dataset>/<set>/silero`` instead, to share them between experiments.

``--tag NAME`` labels the results.tsv lines of this run (e.g. the VAD variant
of a benchmark, see ``../../scripts/aggregate.py``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import typing as tp
from pathlib import Path

ECAPA_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ECAPA_DIR))

import diarize  # noqa: E402
import extract  # noqa: E402
import silero_vad  # noqa: E402
from score_der import COLUMNS, format_table, score_dirs  # noqa: E402

STAGES = ("extract", "diarize", "score")
# set by the pipeline itself
RESERVED = {"--wav-dir", "--out-dir", "--npz-dir", "--overwrite", "--help", "-h"}
# do not change the result
EXTRACT_RUNTIME = {"out_dir", "overwrite", "batch_size", "files_per_batch", "num_workers", "device"}
DIARIZE_RUNTIME = {"npz_dir", "out_dir"}


def parse_args() -> tp.Tuple[argparse.Namespace, tp.List[str], tp.List[str]]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    p.add_argument("--exp-dir", type=Path, required=True)
    p.add_argument("--dataset", default="")
    p.add_argument(
        "--set",
        dest="sets",
        nargs="+",
        action="append",
        required=True,
        metavar="NAME WAV_DIR [REF_DIR]",
    )
    p.add_argument("--stages", default=",".join(STAGES), help="comma-separated subset of " + ",".join(STAGES))
    p.add_argument("--force", action="store_true", help="re-run the selected stages even if up to date")
    p.add_argument("--collars", type=float, nargs="+", default=[0.0, 0.25])
    p.add_argument("--skip-overlap", action="store_true", help="score without the reference overlap regions")
    p.add_argument("--oracle-vad", action="store_true")
    p.add_argument("--oracle-speakers", action="store_true")
    p.add_argument("--silero", action="store_true", help="Silero VAD instead of the model's v_t")
    p.add_argument("--silero-workers", type=int, default=4)
    p.add_argument("--silero-root", type=Path, default=None, help="Silero caches DIR/<dataset>/<set>/silero")
    p.add_argument("--tag", default="", help="label of the results.tsv lines")
    args, rest = p.parse_known_args()

    for spec in args.sets:
        if len(spec) not in (2, 3):
            p.error(f"--set takes NAME WAV_DIR [REF_DIR], got {spec}")
    args.stages = [s for s in args.stages.split(",") if s]
    if unknown := set(args.stages) - set(STAGES):
        p.error(f"unknown stages {sorted(unknown)}")

    routed = route(rest, {"extract": extract.build_parser(), "diarize": diarize.build_parser()}, p)
    return args, routed["extract"], routed["diarize"]


def route(
    argv: tp.List[str], parsers: tp.Mapping[str, argparse.ArgumentParser], main: argparse.ArgumentParser
) -> tp.Dict[str, tp.List[str]]:
    """Split ``argv`` by which parser knows each option; values follow their option."""
    routed: tp.Dict[str, tp.List[str]] = {name: [] for name in parsers}
    current = None
    for token in argv:
        if token.startswith("--"):
            option = token.split("=", 1)[0]
            if option in RESERVED:
                main.error(f"{option} is set by the pipeline")
            owners = [name for name, parser in parsers.items() if option in parser._option_string_actions]
            if len(owners) != 1:
                main.error(f"{option}: {'unknown option' if not owners else 'ambiguous, in ' + ' and '.join(owners)}")
            current = owners[0]
        elif current is None:
            main.error(f"unexpected argument {token}")
        routed[current].append(token)
    return routed


# ---------------------------------------------------------------------------
# Parameters and state
# ---------------------------------------------------------------------------
def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameters(ns: argparse.Namespace, runtime: tp.Set[str]) -> tp.Tuple[dict, dict]:
    """``(key, info)``: ``key`` decides whether results are reusable, ``info`` is
    for people.  Files are keyed by md5, other paths by their absolute path."""
    key, info = {}, {}
    for name, value in sorted(vars(ns).items()):
        if name in runtime:
            continue
        if isinstance(value, Path):
            info[name] = str(value)
            value = {"md5": md5(value)} if value.is_file() else str(value.resolve())
        key[name] = value
    return key, info


def non_default(parser: argparse.ArgumentParser, ns: argparse.Namespace, required: tp.List[str]) -> str:
    """Compact ``name=value`` list of the options that differ from the defaults."""
    defaults = vars(parser.parse_args(required))
    items = []
    for name, value in sorted(vars(ns).items()):
        if value != defaults.get(name) and name not in DIARIZE_RUNTIME:
            items.append(f"{name}={value.name if isinstance(value, Path) else value}")
    return " ".join(items) or "defaults"


def digest(obj: tp.Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]


def load_state(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run(script: str, argv: tp.List[str]) -> None:
    cmd = [sys.executable, str(ECAPA_DIR / script), *argv]
    print("+", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def stage_extract(wav_dir: Path, npz_dir: Path, extract_args: tp.List[str], force: bool) -> None:
    argv = ["--wav-dir", str(wav_dir), "--out-dir", str(npz_dir), *extract_args]
    key, info = parameters(extract.build_parser().parse_args(argv), EXTRACT_RUNTIME)
    # without frames the npz are a subset of the earlier ones (which always had them)
    if not key["save_frames"]:
        del key["save_frames"]
    state_path = npz_dir / ".config.json"
    state = load_state(state_path)
    same = state is not None and state.get("key") == key and not force
    if same and state.get("done"):
        print(f"extract: up to date ({npz_dir})")
        return
    if same:
        print(f"extract: resuming ({npz_dir})")
    save_state(state_path, {"key": key, "info": info, "done": False})
    run("extract.py", argv if same else [*argv, "--overwrite"])
    save_state(state_path, {"key": key, "info": info, "done": True, "time": time.strftime("%F %T")})


def stage_silero(wav_dir: Path, out_dir: Path, ext: str, num_workers: int) -> None:
    todo = silero_vad.pending(wav_dir, out_dir, ext)
    if not todo:
        print(f"silero: up to date ({out_dir})")
        return
    print(f"silero: {len(todo)} files", flush=True)
    run("silero_vad.py", ["--wav-dir", str(wav_dir), "--out-dir", str(out_dir), "--ext", ext,
                          "--num-workers", str(num_workers)])


def diarize_dir(set_dir: Path, npz_dir: Path, diarize_args: tp.List[str]) -> tp.Tuple[Path, dict]:
    """Output dir and state of the diarization with these parameters on these npz."""
    argv = ["--npz-dir", str(npz_dir), "--out-dir", "-", *diarize_args]
    parser = diarize.build_parser()
    ns = parser.parse_args(argv)
    key, info = parameters(ns, DIARIZE_RUNTIME)
    npz_state = load_state(npz_dir / ".config.json") or {}
    npz_key = npz_state.get("key", {"npz_dir": str(npz_dir.resolve())})
    full_key = {"diarize": key, "extract": npz_key}
    state = {
        "key": full_key,
        "info": info,
        "params": non_default(parser, ns, argv[:4]),
        "extract_info": npz_state.get("info", {}),
    }
    return set_dir / f"diar-{digest(full_key)}", state


def stage_diarize(npz_dir: Path, out_dir: Path, state: dict, diarize_args: tp.List[str], force: bool) -> None:
    old = load_state(out_dir / "config.json")
    if old is not None and old.get("key") == state["key"] and old.get("done") and not force:
        print(f"diarize: up to date ({out_dir.name}: {state['params']})")
        return
    save_state(out_dir / "config.json", {**state, "done": False})
    run("diarize.py", ["--npz-dir", str(npz_dir), "--out-dir", str(out_dir / "rttm"), *diarize_args])
    save_state(out_dir / "config.json", {**state, "done": True, "time": time.strftime("%F %T")})


def stage_score(
    ref_dir: Path, out_dir: Path, collars: tp.List[float], skip_overlap: bool
) -> tp.Tuple[tp.List[str], tp.List[str]]:
    """Writes ``scores.csv`` (``scores-skip-overlap.csv``); returns the header and
    the ``*TOTAL*`` row (with the file count)."""
    header, rows = score_dirs(ref_dir, out_dir / "rttm", collars, skip_overlap, ignore_missing=True)
    with open(out_dir / scores_name(skip_overlap), "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    return ["files", *header[1:]], [str(len(rows) - 1), *rows[-1][1:]]


def scores_name(skip_overlap: bool) -> str:
    return "scores-skip-overlap.csv" if skip_overlap else "scores.csv"


def update_results(path: Path, entries: tp.List[tp.Dict[str, str]]) -> None:
    """Upsert ``entries`` into the tsv by ``(set, diar, scoring, tag)``."""
    rows: tp.Dict[tp.Tuple[str, ...], tp.Dict[str, str]] = {}
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                rows[row["set"], row["diar"], row.get("scoring", ""), row.get("tag", "")] = row
    for entry in entries:
        rows[entry["set"], entry["diar"], entry["scoring"], entry["tag"]] = entry
    fields: tp.List[str] = []
    for row in rows.values():
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t", restval="")
        writer.writeheader()
        writer.writerows(rows.values())


def model_tag(info: dict, key: dict) -> str:
    ckpt = key.get("ckpt")
    tag = str(key.get("arch", "?"))
    if isinstance(ckpt, dict):
        tag += f" {Path(info.get('ckpt', '')).name}@{ckpt['md5'][:8]}"
    return tag


def main() -> None:
    args, extract_args, diarize_args = parse_args()
    root = args.exp_dir / args.dataset if args.dataset else args.exp_dir
    silero_root = (args.silero_root / args.dataset if args.dataset else args.silero_root) if args.silero_root else root

    table_header: tp.List[str] = []
    table: tp.List[tp.List[str]] = []
    results: tp.List[tp.Dict[str, str]] = []
    for name, wav_dir, *ref in args.sets:
        ref_dir = Path(ref[0]) if ref else None
        set_dir = root / name
        npz_dir = set_dir / "npz"
        print(f"=== {args.dataset or root.name}/{name}", flush=True)

        set_diarize_args = list(diarize_args)
        if ref_dir is not None and args.oracle_vad:
            set_diarize_args += ["--oracle-vad-dir", str(ref_dir)]
        if ref_dir is not None and args.oracle_speakers:
            set_diarize_args += ["--oracle-speakers-dir", str(ref_dir)]
        if args.silero:
            set_diarize_args += ["--vad-silero", str(silero_root / name / "silero")]

        if "extract" in args.stages:
            stage_extract(Path(wav_dir), npz_dir, extract_args, args.force)
        out_dir, state = diarize_dir(set_dir, npz_dir, set_diarize_args)
        if "diarize" in args.stages:
            if args.silero:
                ext = extract.build_parser().parse_args(["--wav-dir", "-", "--out-dir", "-", *extract_args]).ext
                stage_silero(Path(wav_dir), silero_root / name / "silero", ext, args.silero_workers)
            stage_diarize(npz_dir, out_dir, state, set_diarize_args, args.force)
        if "score" not in args.stages or ref_dir is None:
            continue
        if not (out_dir / "rttm").is_dir():
            print(f"score: no {out_dir / 'rttm'}, run the diarize stage", file=sys.stderr)
            continue
        header, total = stage_score(ref_dir, out_dir, args.collars, args.skip_overlap)
        print(f"score: {out_dir / scores_name(args.skip_overlap)}")
        table_header = ["set", "diar", *header]
        table.append([name, out_dir.name, *total])
        results.append(
            {
                "set": name,
                "diar": out_dir.name,
                "model": model_tag(state["extract_info"], state["key"]["extract"]),
                "diarize": state["params"],
                "scoring": "skip-overlap" if args.skip_overlap else "",
                "tag": args.tag,
                **dict(zip(header, total)),
                "time": time.strftime("%F %T"),
            }
        )

    if not table:
        return
    update_results(root / "results.tsv", results)
    print()
    overlap = ", overlap skipped" if args.skip_overlap else ""
    for c in args.collars:
        cols = [f"{n}[collar={c:g}]" for n in COLUMNS]
        idx = [table_header.index(col) for col in cols]
        print(f"collar {c:g}{overlap}  ({results[0]['model']}; {results[0]['diarize']})")
        print(format_table(["set", "diar", "files", *COLUMNS], [[r[0], r[1], r[2], *[r[i] for i in idx]] for r in table]))
        print()
    print(f"all runs: {root / 'results.tsv'}")


if __name__ == "__main__":
    main()
