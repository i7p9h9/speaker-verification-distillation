"""A self-contained HTML report of the benchmark tables (``aggregate.py``).

    report.py [--bench-dir DIR] [--out FILE] [--title TEXT]

Reads ``<bench>/summary.tsv`` and ``<bench>/summary-skip-overlap.tsv`` as they
are and writes one HTML file (default ``<bench>/report.html``) with the data,
the page (``report_template.html``) and d3 (``vendor/``) inside, so it opens
offline in any browser and can be sent as is.  The page: mean DER by system,
average vs tuned VAD, the cost of overlapped speech, the whole table and a
chart per dataset, switchable between the two scorings; light and dark theme.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import typing as tp
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR.parent))

import diar_config  # noqa: E402

TEMPLATE = TOOLS_DIR / "report_template.html"
D3 = TOOLS_DIR / "vendor" / "d3.v7.9.0.min.js"
TABLES = {"with_overlap": "summary.tsv", "skip_overlap": "summary-skip-overlap.tsv"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench-dir", type=Path, default=None, help="where the tables are; default: benchmark.dir of the config")
    p.add_argument("--out", type=Path, default=None, help="default: <bench>/report.html")
    p.add_argument("--title", default="Diarization benchmark")
    return p.parse_args()


def read_table(path: Path) -> tp.Dict[str, tp.Any]:
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        columns = next(reader)
        rows = [dict(zip(columns, row)) for row in reader if row]
    return {"columns": columns, "rows": rows}


def html_text(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    args = parse_args()
    bench = args.bench_dir or diar_config.get_path("benchmark.dir")
    out = args.out or bench / "report.html"
    missing = [bench / name for name in TABLES.values() if not (bench / name).exists()]
    if missing:
        raise SystemExit(f"no {', '.join(map(str, missing))}: run aggregate.py first")
    data = {"datasets": {key: read_table(bench / name) for key, name in TABLES.items()}}
    # "</" would end the <script> element that holds the JSON
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = (
        TEMPLATE.read_text()
        .replace("__TITLE__", html_text(args.title))
        .replace("__GENERATED__", time.strftime("%Y-%m-%d %H:%M"))
        .replace("__SOURCES__", html_text(", ".join(TABLES.values())))
        .replace("/*__D3__*/", D3.read_text())
        .replace("__DATA__", payload)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"{out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
