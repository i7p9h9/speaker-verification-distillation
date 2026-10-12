"""Silero VAD -> extract.py -> diarize.py -> DER / VAD scores for several sets,
re-running only what changed.

    pipeline.py --exp-dir DIR [--dataset NAME] --set NAME WAV_DIR [REF_DIR] [--set ...]
                [--stages extract,diarize,score] [--force] [--collars 0 0.25] [--skip-overlap]
                [--oracle-vad] [--silero-root DIR] [--tag NAME] [extract.py / diarize.py options]

Options of ``extract.py`` and ``diarize.py`` are given directly and routed to
the script that has them (``--vad-silero-prob``, ``--onset``, ``--batch-size``
-> extract; ``--threshold``, ``--Fa``, ``--Fb``, ``--loopP`` -> diarize;
``--model``, ``--num-workers`` -> both).  Without ``--model`` / ``--weights``
the ``--default-model`` is used.  Paths that the pipeline sets (``--wav-dir``,
``--out-dir``, ``--xvec-dir``, ``--vad-silero``, ``--overwrite``) are not accepted.

VAD: Silero segments by default (``--vad-silero-prob --onset .. --offset ..
--merge-gap .. --min-dur ..`` to threshold its probabilities instead, e.g. with
the values from ``tune.py --target vad``); ``--oracle-vad`` takes the speech of
the reference; ``--vad-lab-dir DIR`` VBx ``.lab`` files.  Silero runs once per
set, for the files missing from its cache.

Layout, per set (``<root> = <exp-dir>/<dataset>``, or ``<exp-dir>`` without ``--dataset``):

    <root>/<set>/silero/*.npz                         Silero VAD cache (or <silero-root>/<set>/silero)
    <root>/<set>/xvec-<hash>/*.npz                    extract.py output, one dir per parameter set
    <root>/<set>/xvec-<hash>/.config.json             its parameters
    <root>/<set>/xvec-<hash>/diar-<hash>/rttm/        diarize.py output, one dir per parameter set
    <root>/<set>/xvec-<hash>/diar-<hash>/config.json  its parameters
    <root>/<set>/xvec-<hash>/diar-<hash>/scores.csv   per-file DER and VAD metrics (scores-skip-overlap.csv)
    <root>/results.tsv                                one line per (set, xvec, diar, scoring, tag)

A stage is skipped when it has completed with the same parameters.  Model
files (``--weights`` or the preset's, ``--plda``, ``--xvec-transform``) are
compared by md5 (directories by the md5 of their files); options that do not
change the result (batch size, workers, device) are ignored.  An interrupted
extraction resumes.  Scoring is cheap and always runs for sets with a reference.
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

VBX_DIR = Path(__file__).resolve().parent.parent
DIAR_DIR = VBX_DIR.parent
sys.path.insert(0, str(VBX_DIR))
sys.path.insert(1, str(DIAR_DIR))

import diarize  # noqa: E402
import extract  # noqa: E402
from models import PRESETS  # noqa: E402

from ecapa import silero_vad  # noqa: E402
from ecapa.score_der import COLUMNS, format_table, score_dirs  # noqa: E402

STAGES = ("extract", "diarize", "score")
# set by the pipeline itself
RESERVED = {"--wav-dir", "--out-dir", "--xvec-dir", "--vad-silero", "--vad-rttm-dir", "--overwrite", "--help", "-h"}
# do not change the result
EXTRACT_RUNTIME = {"out_dir", "overwrite", "batch_size", "num_workers", "device"}
DIARIZE_RUNTIME = {"xvec_dir", "out_dir", "num_workers"}
PROB_PARAMS = ("onset", "offset", "merge_gap", "min_dur")


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
    p.add_argument("--oracle-vad", action="store_true", help="speech of the reference instead of Silero")
    p.add_argument("--tag", default="", help="label of the results.tsv lines (e.g. a benchmark's VAD variant)")
    p.add_argument("--silero-root", type=Path, default=None, help="Silero caches at DIR/<set>/silero")
    p.add_argument("--silero-workers", type=int, default=4)
    p.add_argument("--default-model", choices=sorted(PRESETS), default=None)
    args, rest = p.parse_known_args()

    for spec in args.sets:
        if len(spec) not in (2, 3):
            p.error(f"--set takes NAME WAV_DIR [REF_DIR], got {spec}")
        if args.oracle_vad and len(spec) < 3:
            p.error(f"--oracle-vad: set {spec[0]} has no reference")
    args.stages = [s for s in args.stages.split(",") if s]
    if unknown := set(args.stages) - set(STAGES):
        p.error(f"unknown stages {sorted(unknown)}")

    routed = route(rest, {"extract": extract.build_parser(), "diarize": diarize.build_parser()}, p)
    if not any(opt.split("=")[0] in ("--model", "--weights") for opt in routed["extract"]):
        if args.default_model is None:
            p.error("give --model or --weights")
        routed["extract"] += ["--model", args.default_model]
        routed["diarize"] += ["--model", args.default_model]
    if "--vad-lab-dir" in routed["extract"] and args.oracle_vad:
        p.error("--oracle-vad and --vad-lab-dir are exclusive")
    return args, routed["extract"], routed["diarize"]


def route(
    argv: tp.List[str], parsers: tp.Mapping[str, argparse.ArgumentParser], main: argparse.ArgumentParser
) -> tp.Dict[str, tp.List[str]]:
    """Split ``argv`` by which parser knows each option (options of several
    parsers go to all of them); values follow their option."""
    routed: tp.Dict[str, tp.List[str]] = {name: [] for name in parsers}
    owners: tp.List[str] = []
    for token in argv:
        if token.startswith("--"):
            option = token.split("=", 1)[0]
            if option in RESERVED:
                main.error(f"{option} is set by the pipeline")
            owners = [name for name, parser in parsers.items() if option in parser._option_string_actions]
            if not owners:
                main.error(f"{option}: unknown option")
        elif not owners:
            main.error(f"unexpected argument {token}")
        for name in owners:
            routed[name].append(token)
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


def fingerprint(path: Path) -> str:
    """md5 of a file, or of the names and md5s of the files of a directory."""
    if path.is_file():
        return md5(path)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    return hashlib.md5(json.dumps([(str(p.relative_to(path)), md5(p)) for p in files]).encode()).hexdigest()


def parameters(ns: argparse.Namespace, runtime: tp.Set[str], hashed: tp.Set[str]) -> tp.Tuple[dict, dict]:
    """``(key, info)``: ``key`` decides whether results are reusable, ``info`` is
    for people.  Paths in ``hashed`` are keyed by content, others by path."""
    key, info = {}, {}
    for name, value in sorted(vars(ns).items()):
        if name in runtime:
            continue
        if isinstance(value, Path):
            info[name] = str(value)
            value = {"md5": fingerprint(value)} if name in hashed else str(value.resolve())
        key[name] = value
    return key, info


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


def run(script: Path, argv: tp.List[str]) -> None:
    cmd = [sys.executable, str(script), *argv]
    print("+", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True)


def vad_tag(ns: argparse.Namespace) -> str:
    if ns.vad_rttm_dir is not None:
        return "oracle"
    if ns.vad_lab_dir is not None:
        return f"lab:{ns.vad_lab_dir.name}"
    if ns.vad_silero_prob:
        return "silero-prob " + "/".join(f"{getattr(ns, n):.4g}" for n in PROB_PARAMS)
    return "silero"


def model_tag(ns: argparse.Namespace, key: dict) -> str:
    name = ns.model or Path(ns.weights).name
    return f"{name}@{key['weights']['md5'][:8]}"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
class Extraction(tp.NamedTuple):
    dir: Path
    argv: tp.List[str]
    key: dict
    info: dict
    model: str
    vad: str


def extraction(set_dir: Path, wav_dir: Path, vad_argv: tp.List[str], extract_args: tp.List[str]) -> Extraction:
    argv = ["--wav-dir", str(wav_dir), "--out-dir", "-", *vad_argv, *extract_args]
    ns = extract.parse_args(argv)
    runtime = set(EXTRACT_RUNTIME)
    if not ns.vad_silero_prob:
        runtime |= set(PROB_PARAMS)
    key, info = parameters(ns, runtime, {"weights"})
    key.pop("wav_dir")  # the set is given by the directory
    out_dir = set_dir / f"xvec-{digest(key)}"
    argv[3] = str(out_dir)
    return Extraction(out_dir, argv, key, info, model_tag(ns, key), vad_tag(ns))


def stage_extract(ex: Extraction, force: bool) -> None:
    state_path = ex.dir / ".config.json"
    state = load_state(state_path)
    same = state is not None and state.get("key") == ex.key and not force
    if same and state.get("done"):
        print(f"extract: up to date ({ex.dir.name}: {ex.model}, vad {ex.vad})")
        return
    if same:
        print(f"extract: resuming ({ex.dir})")
    save_state(state_path, {"key": ex.key, "info": ex.info, "done": False})
    run(VBX_DIR / "extract.py", ex.argv if same else [*ex.argv, "--overwrite"])
    save_state(state_path, {"key": ex.key, "info": ex.info, "done": True, "time": time.strftime("%F %T")})


def stage_silero(wav_dir: Path, out_dir: Path, ext: str, num_workers: int) -> None:
    todo = silero_vad.pending(wav_dir, out_dir, ext)
    if not todo:
        print(f"silero: up to date ({out_dir})")
        return
    print(f"silero: {len(todo)} files", flush=True)
    argv = ["--wav-dir", str(wav_dir), "--out-dir", str(out_dir), "--ext", ext, "--num-workers", str(num_workers)]
    run(DIAR_DIR / "ecapa" / "silero_vad.py", argv)


class Diarization(tp.NamedTuple):
    dir: Path
    argv: tp.List[str]
    state: dict


def non_default(parser: argparse.ArgumentParser, ns: argparse.Namespace, required: tp.List[str]) -> str:
    """Compact ``name=value`` list of the options that differ from the defaults."""
    defaults = vars(diarize.resolve(parser, parser.parse_args(required)))
    items = []
    for name, value in sorted(vars(ns).items()):
        if value != defaults.get(name) and name not in DIARIZE_RUNTIME | {"model"}:
            items.append(f"{name}={value.name if isinstance(value, Path) else value}")
    return " ".join(items) or "defaults"


def diarization(ex: Extraction, diarize_args: tp.List[str]) -> Diarization:
    argv = ["--xvec-dir", str(ex.dir), "--out-dir", "-", *diarize_args]
    parser = diarize.build_parser()
    ns = diarize.resolve(parser, parser.parse_args(argv))
    key, info = parameters(ns, DIARIZE_RUNTIME, {"plda", "xvec_transform"})
    key.pop("model")  # only the defaults, which are resolved in the key
    model = ["--model", ns.model] if ns.model else []
    state = {"key": key, "info": info, "params": non_default(parser, ns, [*argv[:4], *model])}
    out_dir = ex.dir / f"diar-{digest(key)}"
    argv[3] = str(out_dir / "rttm")
    return Diarization(out_dir, argv, state)


def stage_diarize(d: Diarization, force: bool) -> None:
    old = load_state(d.dir / "config.json")
    if old is not None and old.get("key") == d.state["key"] and old.get("done") and not force:
        print(f"diarize: up to date ({d.dir.name}: {d.state['params']})")
        return
    save_state(d.dir / "config.json", {**d.state, "done": False})
    run(VBX_DIR / "diarize.py", d.argv)
    save_state(d.dir / "config.json", {**d.state, "done": True, "time": time.strftime("%F %T")})


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
    """Upsert ``entries`` into the tsv by ``(set, xvec, diar, scoring, tag)``."""
    rows: tp.Dict[tp.Tuple[str, ...], tp.Dict[str, str]] = {}
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                rows[row["set"], row["xvec"], row["diar"], row.get("scoring", ""), row.get("tag", "")] = row
    for entry in entries:
        rows[entry["set"], entry["xvec"], entry["diar"], entry["scoring"], entry["tag"]] = entry
    fields: tp.List[str] = []
    for row in rows.values():
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t", restval="")
        writer.writeheader()
        writer.writerows(rows.values())


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
        print(f"=== {args.dataset or root.name}/{name}", flush=True)

        silero_dir = silero_root / name / "silero"
        if args.oracle_vad:
            vad_argv = ["--vad-rttm-dir", str(ref_dir)]
        elif "--vad-lab-dir" in extract_args:
            vad_argv = []
        else:
            vad_argv = ["--vad-silero", str(silero_dir)]
        ex = extraction(set_dir, Path(wav_dir), vad_argv, extract_args)
        d = diarization(ex, diarize_args)

        if "extract" in args.stages:
            if "--vad-silero" in vad_argv:
                stage_silero(Path(wav_dir), silero_dir, extract.parse_args(ex.argv).ext, args.silero_workers)
            stage_extract(ex, args.force)
        if "diarize" in args.stages:
            stage_diarize(d, args.force)
        if "score" not in args.stages or ref_dir is None:
            continue
        if not (d.dir / "rttm").is_dir():
            print(f"score: no {d.dir / 'rttm'}, run the diarize stage", file=sys.stderr)
            continue
        header, total = stage_score(ref_dir, d.dir, args.collars, args.skip_overlap)
        print(f"score: {d.dir / scores_name(args.skip_overlap)}")
        table_header = ["set", "xvec", "diar", *header]
        table.append([name, ex.dir.name, d.dir.name, *total])
        results.append(
            {
                "set": name,
                "xvec": ex.dir.name,
                "diar": d.dir.name,
                "model": ex.model,
                "vad": ex.vad,
                "diarize": d.state["params"],
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
        print(f"collar {c:g}{overlap}  ({results[0]['model']}; vad {results[0]['vad']}; {results[0]['diarize']})")
        rows = [[r[0], r[1], r[2], r[3], *[r[i] for i in idx]] for r in table]
        print(format_table(["set", "xvec", "diar", "files", *COLUMNS], rows))
        print()
    print(f"all runs: {root / 'results.tsv'}")


if __name__ == "__main__":
    main()
