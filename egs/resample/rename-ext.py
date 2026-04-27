import typing as tp
import argparse
from pathlib import Path


def find_wav_files(root: Path) -> tp.List[Path]:
    """Recursively find all .wav files under root directory."""
    return list(root.rglob("*.wav"))


def rename_wav_to_flac(
    root: Path,
    dry_run: bool = False,
    verbose: bool = True,
) -> tp.Tuple[int, int]:
    """
    Rename all *.wav files to *.flac recursively under root.

    Returns (success_count, error_count).
    """
    wav_files = find_wav_files(root)

    if not wav_files:
        print(f"No .wav files found under '{root}'")
        return 0, 0

    print(f"Found {len(wav_files)} .wav file(s) under '{root}'")
    if dry_run:
        print("[DRY RUN] No files will be renamed.\n")

    success, errors = 0, 0

    for wav_path in sorted(wav_files):
        flac_path = wav_path.with_suffix(".flac")

        if flac_path.exists():
            print(f"  [SKIP]  {wav_path}  →  target already exists: {flac_path}")
            errors += 1
            continue

        if verbose or dry_run:
            print(f"  {'[DRY]' if dry_run else '[OK] '} {wav_path}  →  {flac_path}")

        if not dry_run:
            try:
                wav_path.rename(flac_path)
                success += 1
            except OSError as exc:
                print(f"  [ERR]  {wav_path}: {exc}")
                errors += 1
        else:
            success += 1

    print(f"\nDone. Renamed: {success}, Skipped/Errors: {errors}")
    return success, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively rename *.wav → *.flac (undo accidental mass-rename)."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to search (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without touching any files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors and summary",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Error: '{root}' is not a directory.")
        raise SystemExit(1)

    rename_wav_to_flac(root, dry_run=args.dry_run, verbose=not args.quiet)


if __name__ == "__main__":
    main()
