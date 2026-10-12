"""Benchmark of the diarization systems on all datasets of ``datasets.tsv``.

    benchmark.py STAGE [STAGE ...] [--bench-dir DIR] [--datasets a,b] [--vad tuned|average]
                 [--ecapa-models a,b] [--vbx-models a,b] [--skip-failed]

Stages, in the order they are given:

    nemotron  Nemotron-3-Diarization (../nemotron/run.sh, Docker) on every set,
              RTTMs at threshold 0.5, scored
    silero    Silero VAD of every set, once: <bench>/silero/<Dataset>/<set>/silero
    tune      the Silero thresholds (onset, offset, merge_gap, min_dur) of every
              dataset by the VAD error on its tune set (``../vbx/tune.py --target
              vad``, all files: --tune-fraction 1): <bench>/vad/<Dataset>.json
    average   their mean over the datasets: <bench>/vad/average.json
    ecapa     every --ecapa-models on every set with the Silero VAD of --vad
    vbx       every --vbx-models preset on every set with the Silero VAD of --vad

``--vad tuned`` takes each dataset's own thresholds, ``--vad average`` the mean
ones for all.  ecapa and vbx run through ``<recipe>/scripts/run-<key>.sh``
(dataset preparation, default rates and caching as there) into
<bench>/<recipe>/<model>/, tagged with the VAD variant, and are scored twice:
with and without (--skip-overlap) the overlap regions.  The ecapa extraction
does not depend on the VAD, so the second variant only re-diarizes.

Logs go to <bench>/logs/; a failed run stops the benchmark unless
--skip-failed.  ``aggregate.py`` puts the results into tables.  Typical order:

    benchmark.py nemotron
    benchmark.py silero tune ecapa vbx --vad tuned
    benchmark.py average ecapa vbx --vad average
    aggregate.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import typing as tp
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
DIAR_DIR = TOOLS_DIR.parent
sys.path.insert(0, str(DIAR_DIR))
sys.path.insert(0, str(TOOLS_DIR))

import prepare_data  # noqa: E402

DATA_DIR = Path(os.environ.get("DIAR_DATA_DIR", "/media/raid/diarization/data"))
BENCH_DIR = Path("/media/raid/diarization/results/bench")
STAGES = ("nemotron", "silero", "tune", "average", "ecapa", "vbx")
VAD_PARAMS = ("onset", "offset", "merge_gap", "min_dur")

ECAPA_MODELS: tp.Dict[str, tp.List[str]] = {
    "ecapa-tdnn-003": [
        "--arch", "ECAPA_TDNN_VAD",
        "--ckpt", "/media/raid/diarization/exps/ecapa-tdnn-003/last.ckpt",
        "--batch-size", "32",
    ],
    "resnettf-teacher-vox4": [
        "--arch", "ResNetTF",
        "--config", "/media/data2/Projects/formanta-distill/egs/voxceleb-v2/data/cfg-models/rn100_v016_flr_vox4_v2.yaml",
        "--ckpt", "/media/data2/Projects/formanta-distill/egs/voxceleb-v2/data/ckpt/rn100_v016_flr_vox4_v2/model.pt",
        "--batch-size", "32",
    ],
}  # fmt: skip
VBX_MODELS = ("resnet101-8k", "resnet101-16k", "redimnet", "redimnet-tel", "redimnet-tel-small")


class Dataset(tp.NamedTuple):
    key: str
    name: str
    root: Path
    rate: int
    sets: str  # comma-separated, "*" = all prepared
    tune_set: str

    def prepared(self) -> Path:
        return self.root / prepare_data.PREPARED

    def set_names(self) -> tp.List[str]:
        if self.sets != "*":
            return self.sets.split(",")
        return sorted(d.name for d in self.prepared().iterdir() if d.is_dir() and d.name != "audio")

    def wav(self, set_name: str) -> Path:
        return self.prepared() / set_name / "wav"

    def ref(self, set_name: str) -> Path:
        return self.prepared() / set_name / "rttm"


def read_datasets() -> tp.Dict[str, Dataset]:
    out = {}
    for line in (TOOLS_DIR / "datasets.tsv").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            key, name, root, rate, sets, tune_set = line.split("\t")
            out[key] = Dataset(key, name, DATA_DIR / root, int(rate), sets, tune_set)
    return out


def parse_args() -> argparse.Namespace:
    datasets = read_datasets()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stages", nargs="+", choices=STAGES)
    p.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    p.add_argument("--datasets", default=",".join(datasets), help="keys of datasets.tsv")
    p.add_argument("--vad", choices=["tuned", "average"], default="tuned", help="ecapa, vbx: Silero thresholds")
    p.add_argument("--ecapa-models", default=",".join(ECAPA_MODELS))
    p.add_argument("--vbx-models", default=",".join(VBX_MODELS))
    p.add_argument("--tune-fraction", type=float, default=1.0)
    p.add_argument("--tune-trials", type=int, default=200)
    p.add_argument("--force", action="store_true", help="tune again; re-run up-to-date pipeline stages")
    p.add_argument("--skip-failed", action="store_true", help="go on with the next run after a failure")
    p.add_argument("--silero-workers", type=int, default=8)
    args = p.parse_args()
    keys = [k for k in args.datasets.split(",") if k]
    if unknown := set(keys) - set(datasets):
        p.error(f"unknown datasets {sorted(unknown)} ({','.join(datasets)})")
    args.datasets = [datasets[k] for k in keys]
    args.ecapa_models = [m for m in args.ecapa_models.split(",") if m]
    args.vbx_models = [m for m in args.vbx_models.split(",") if m]
    if unknown := set(args.ecapa_models) - set(ECAPA_MODELS):
        p.error(f"unknown ecapa models {sorted(unknown)} ({','.join(ECAPA_MODELS)})")
    if unknown := set(args.vbx_models) - set(VBX_MODELS):
        p.error(f"unknown vbx models {sorted(unknown)} ({','.join(VBX_MODELS)})")
    return args


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
class Failed(Exception):
    pass


def py(script: Path, *argv: str) -> tp.List[str]:
    return [sys.executable, str(script), *argv]


def run(cmd: tp.Sequence[str], log: Path, args: argparse.Namespace, what: str) -> bool:
    """Run ``cmd`` with its output appended to ``log``; True if it succeeded."""
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(f"{time.strftime('%T')} {what} ...", end=" ", flush=True)
    with open(log, "a") as f:
        f.write(f"\n### {time.strftime('%F %T')} {' '.join(cmd)}\n")
        f.flush()
        code = subprocess.run(list(cmd), stdout=f, stderr=subprocess.STDOUT).returncode
    minutes = (time.time() - start) / 60
    if code == 0:
        print(f"done ({minutes:.1f} min)", flush=True)
        return True
    print(f"FAILED (exit {code}, {minutes:.1f} min), see {log}", flush=True)
    if not args.skip_failed:
        raise Failed(what)
    return False


def prepare(ds: Dataset) -> None:
    if not prepare_data.ready(ds.root):
        subprocess.run(py(TOOLS_DIR / "prepare_data.py", ds.key, str(ds.root)), check=True)


def stage_nemotron(args: argparse.Namespace) -> None:
    from ecapa.score_der import score_dirs

    for ds in args.datasets:
        prepare(ds)
        root = args.bench_dir / "nemotron" / ds.name
        results = []
        for set_name in ds.set_names():
            set_dir = root / set_name
            log = args.bench_dir / "logs" / f"nemotron-{ds.name}-{set_name}.log"
            cmd = [str(DIAR_DIR / "nemotron" / "run.sh"), str(ds.wav(set_name)), str(set_dir / "raw")]
            if not run(cmd, log, args, f"nemotron {ds.name}/{set_name}"):
                continue
            to_rttm = py(DIAR_DIR / "nemotron" / "to_rttm.py", "--in-dir", str(set_dir / "raw"), "--out-dir",
                         str(set_dir / "rttm"))
            if not run(to_rttm, log, args, "  to rttm"):
                continue
            for skip_overlap in (False, True):
                header, rows = score_dirs(ds.ref(set_name), set_dir / "rttm", [0.0, 0.25], skip_overlap,
                                          ignore_missing=True)
                name = "scores-skip-overlap.csv" if skip_overlap else "scores.csv"
                with open(set_dir / name, "w", newline="") as f:
                    csv.writer(f).writerows([header] + rows)
                results.append({
                    "set": set_name, "diar": "nemotron", "model": "nvidia/Nemotron-3-Diarization",
                    "diarize": "threshold=0.5", "scoring": "skip-overlap" if skip_overlap else "",
                    "tag": "nemotron", "files": str(len(rows) - 1), **dict(zip(header[1:], rows[-1][1:])),
                    "time": time.strftime("%F %T"),
                })  # fmt: skip
        if results:
            write_results(root / "results.tsv", results)


def write_results(path: Path, rows: tp.List[tp.Dict[str, str]]) -> None:
    fields: tp.List[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t", restval="")
        writer.writeheader()
        writer.writerows(rows)


def stage_silero(args: argparse.Namespace) -> None:
    for ds in args.datasets:
        prepare(ds)
        for set_name in ds.set_names():
            out = silero_dir(args, ds, set_name)
            log = args.bench_dir / "logs" / f"silero-{ds.name}-{set_name}.log"
            cmd = py(DIAR_DIR / "ecapa" / "silero_vad.py", "--wav-dir", str(ds.wav(set_name)), "--out-dir", str(out),
                     "--num-workers", str(args.silero_workers))
            run(cmd, log, args, f"silero {ds.name}/{set_name}")


def silero_dir(args: argparse.Namespace, ds: Dataset, set_name: str) -> Path:
    return args.bench_dir / "silero" / ds.name / set_name / "silero"


def vad_json(args: argparse.Namespace, name: str) -> Path:
    return args.bench_dir / "vad" / f"{name}.json"


def stage_tune(args: argparse.Namespace) -> None:
    for ds in args.datasets:
        out = vad_json(args, ds.name)
        if out.exists() and not args.force:
            print(f"tune {ds.name}: done ({out})")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = py(DIAR_DIR / "vbx" / "tune.py", "--target", "vad", "--silero-dir", str(silero_dir(args, ds, ds.tune_set)),
                 "--ref-dir", str(ds.ref(ds.tune_set)), "--fraction", str(args.tune_fraction),
                 "--n-trials", str(args.tune_trials), "--out", str(out))
        log = args.bench_dir / "logs" / f"tune-{ds.name}.log"
        if run(cmd, log, args, f"tune {ds.name}/{ds.tune_set}"):
            best = json.loads(out.read_text())["best"]
            print(f"  VAD error {best['value']:.2f}: {fmt_params(best['params'])}")


def fmt_params(params: tp.Mapping[str, float]) -> str:
    return " ".join(f"{k}={params[k]:.4g}" for k in VAD_PARAMS)


def stage_average(args: argparse.Namespace) -> None:
    tuned = {}
    for ds in args.datasets:
        path = vad_json(args, ds.name)
        if not path.exists():
            raise SystemExit(f"average: no {path}, run the tune stage")
        tuned[ds.name] = json.loads(path.read_text())["best"]["params"]
    mean = {k: sum(p[k] for p in tuned.values()) / len(tuned) for k in VAD_PARAMS}
    out = vad_json(args, "average")
    out.write_text(json.dumps({"params": mean, "datasets": tuned}, indent=2) + "\n")
    print(f"average of {len(tuned)} datasets: {fmt_params(mean)}\n{out}")


def vad_options(args: argparse.Namespace, ds: Dataset) -> tp.List[str]:
    path = vad_json(args, "average" if args.vad == "average" else ds.name)
    if not path.exists():
        raise SystemExit(f"no {path}, run the {'average' if args.vad == 'average' else 'tune'} stage")
    data = json.loads(path.read_text())
    params = data["params"] if args.vad == "average" else data["best"]["params"]
    options = ["--vad-silero-prob"]
    for k in VAD_PARAMS:
        options += [f"--{k.replace('_', '-')}", f"{params[k]:.6g}"]
    return options


def stage_recipe(recipe: str, args: argparse.Namespace) -> None:
    models = args.ecapa_models if recipe == "ecapa" else args.vbx_models
    for model in models:
        model_args = ECAPA_MODELS[model] + ["--silero"] if recipe == "ecapa" else ["--model", model]
        for ds in args.datasets:
            script = DIAR_DIR / recipe / "scripts" / f"run-{ds.key}.sh"
            common = ["--exp-dir", str(args.bench_dir / recipe / model), "--silero-root", str(args.bench_dir / "silero"),
                      "--tag", args.vad, *model_args, *vad_options(args, ds)]
            log = args.bench_dir / "logs" / f"{recipe}-{model}-{ds.name}.log"
            what = f"{recipe} {model} {ds.name} ({args.vad} VAD)"
            if not run([str(script), *common, *(["--force"] if args.force else [])], log, args, what):
                continue
            run([str(script), *common, "--stages", "score", "--skip-overlap"], log, args, "  score without overlap")


def main() -> None:
    args = parse_args()
    args.bench_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "nemotron": stage_nemotron,
        "silero": stage_silero,
        "tune": stage_tune,
        "average": stage_average,
        "ecapa": lambda a: stage_recipe("ecapa", a),
        "vbx": lambda a: stage_recipe("vbx", a),
    }
    try:
        for stage in args.stages:
            print(f"=== {stage}", flush=True)
            stages[stage](args)
    except Failed as e:
        raise SystemExit(f"stopped: {e} failed (--skip-failed to go on)")


if __name__ == "__main__":
    main()
