"""Machine-specific paths of the diarization recipes, from ``config.yaml`` next
to this file (or ``$DIAR_CONFIG``).  Relative paths there are relative to the
config file's directory.

    python egs/diarization/diar_config.py data_dir    # print a path (for the shell scripts)
"""

from __future__ import annotations

import functools
import os
import sys
import typing as tp
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def config_path() -> Path:
    return Path(os.environ.get("DIAR_CONFIG", HERE / "config.yaml")).expanduser().resolve()


@functools.lru_cache(maxsize=None)
def load() -> tp.Dict[str, tp.Any]:
    path = config_path()
    if not path.is_file():
        raise SystemExit(f"no config {path} (see {HERE / 'config.yaml'})")
    return yaml.safe_load(path.read_text()) or {}


def get(dotted: str) -> tp.Any:
    """``get("benchmark.dir")``: a value of the config."""
    value: tp.Any = load()
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise SystemExit(f"{config_path()}: no '{dotted}'")
        value = value[part]
    return value


def resolve(value: str | os.PathLike) -> Path:
    """A path of the config, absolute (symlinks are kept)."""
    return Path(os.path.abspath(config_path().parent / Path(value).expanduser()))


def get_path(dotted: str) -> Path:
    return resolve(get(dotted))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(get_path(sys.argv[1]))
