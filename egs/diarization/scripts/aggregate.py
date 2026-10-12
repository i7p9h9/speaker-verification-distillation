"""Tables of the benchmark results (``benchmark.py``): systems x datasets.

    aggregate.py [--bench-dir DIR] [--metric DER] [--collar 0.25] [--out-dir DIR]

Reads every tagged line of ``<bench>/nemotron/<Dataset>/results.tsv`` and
``<bench>/{ecapa,vbx}/<model>/<Dataset>/results.tsv`` (the latest one per
system, VAD variant, set and scoring) and prints two tables of ``--metric`` at
``--collar``: with the overlap regions and without them (--skip-overlap
scoring).  A row is a system with its VAD variant (tuned: each dataset's own
Silero thresholds, average: the mean ones), a column a dataset (``Dataset/set``
for datasets with several sets); ``mean`` averages the columns, over the ones
the row has (their number in brackets when some are missing).  Also written to
``<out-dir>/summary.tsv`` and ``summary-skip-overlap.tsv`` (default: the bench dir).
"""

from __future__ import annotations

import argparse
import csv
import sys
import typing as tp
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from benchmark import BENCH_DIR, read_datasets  # noqa: E402

METRICS = ("DER", "MS", "FA", "SC", "VAD", "VAD_MS", "VAD_FA")
RECIPE_ORDER = {"nemotron": 0, "ecapa": 1, "vbx": 2}
VARIANT_ORDER = {"nemotron": 0, "tuned": 1, "average": 2}

Key = tp.Tuple[str, str, str, str, str]  # system, variant, dataset, set, scoring


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    p.add_argument("--metric", choices=METRICS, default="DER")
    p.add_argument("--collar", type=float, default=0.25)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def collect(bench: Path, column: str) -> tp.Dict[Key, float]:
    latest: tp.Dict[Key, tp.Tuple[str, float]] = {}
    for path in sorted(bench.rglob("results.tsv")):
        parts = path.relative_to(bench).parts
        if parts[0] == "nemotron" and len(parts) == 3:
            system, dataset = "nemotron", parts[1]
        elif parts[0] in ("ecapa", "vbx") and len(parts) == 4:
            system, dataset = f"{parts[0]}/{parts[1]}", parts[2]
        else:
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if not row.get("tag") or not row.get(column):
                    continue
                key = (system, row["tag"], dataset, row["set"], row.get("scoring", ""))
                stamp = row.get("time", "")
                if key not in latest or stamp >= latest[key][0]:
                    latest[key] = (stamp, float(row[column]))
    return {k: v for k, (_, v) in latest.items()}


def table(values: tp.Dict[Key, float], scoring: str, dataset_order: tp.List[str]) -> tp.List[tp.List[str]]:
    values = {k: v for k, v in values.items() if k[4] == scoring}
    sets: tp.Dict[str, tp.Set[str]] = {}
    for _, _, dataset, set_name, _ in values:
        sets.setdefault(dataset, set()).add(set_name)
    order = {name: i for i, name in enumerate(dataset_order)}
    columns = [
        (d, s)
        for d in sorted(sets, key=lambda d: (order.get(d, len(order)), d))
        for s in sorted(sets[d])
    ]
    names = [d if len(sets[d]) == 1 else f"{d}/{s}" for d, s in columns]
    rows = sorted(
        {(k[0], k[1]) for k in values},
        key=lambda r: (RECIPE_ORDER.get(r[0].split("/")[0], 9), r[0], VARIANT_ORDER.get(r[1], 9), r[1]),
    )
    out = [["system", "vad", *names, "mean"]]
    for system, variant in rows:
        cells = [values.get((system, variant, d, s, scoring)) for d, s in columns]
        have = [c for c in cells if c is not None]
        mean = f"{sum(have) / len(have):.2f}" if have else "-"
        if have and len(have) < len(cells):
            mean += f" ({len(have)})"
        vad = "-" if variant == "nemotron" else variant
        out.append([system, vad, *["-" if c is None else f"{c:.2f}" for c in cells], mean])
    return out


def format_table(rows: tp.List[tp.List[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for n, row in enumerate(rows):
        cells = [c.ljust(w) if i < 2 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths))]
        lines.append("  ".join(cells).rstrip())
        if n == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    column = f"{args.metric}[collar={args.collar:g}]"
    values = collect(args.bench_dir, column)
    if not values:
        raise SystemExit(f"no tagged results in {args.bench_dir}")
    dataset_order = [ds.name for ds in read_datasets().values()]
    out_dir = args.out_dir or args.bench_dir
    for scoring, title, name in (
        ("", "with overlap", "summary.tsv"),
        ("skip-overlap", "overlap skipped", "summary-skip-overlap.tsv"),
    ):
        rows = table(values, scoring, dataset_order)
        if len(rows) == 1:
            continue
        print(f"{args.metric}, collar {args.collar:g}, {title}\n")
        print(format_table(rows))
        print()
        with open(out_dir / name, "w", newline="") as f:
            csv.writer(f, delimiter="\t").writerows(rows)
        print(f"{out_dir / name}\n")


if __name__ == "__main__":
    main()
