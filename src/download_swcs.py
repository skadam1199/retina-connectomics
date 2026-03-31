#!/usr/bin/env python3
"""Download/export SWC files for a list of segment IDs.

For each segment ID, this script ensures an SWC exists in a flatone output tree,
then copies it into a target folder with filename `<segment_id>.swc`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _parse_segment_id(raw: str) -> int:
    value = raw.strip()
    if not value.isdigit():
        raise argparse.ArgumentTypeError(f"Non-numeric segment ID: {raw!r}")
    return int(value)


def load_segment_ids(ids_file: Path | None, cli_ids: list[int]) -> list[int]:
    ids: list[int] = []
    if ids_file is not None:
        for lineno, raw in enumerate(ids_file.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not line.isdigit():
                raise ValueError(f"{ids_file}:{lineno}: non-numeric segment ID: {line!r}")
            ids.append(int(line))

    ids.extend(cli_ids)
    if not ids:
        raise ValueError("No segment IDs provided. Use --segment-ids-file and/or SEGMENT_ID args.")

    seen: set[int] = set()
    unique: list[int] = []
    for seg_id in ids:
        if seg_id not in seen:
            seen.add(seg_id)
            unique.append(seg_id)
    return unique


def run_flatone_for_segment(
    seg_id: int,
    flatone_output_dir: Path,
    python_bin: str,
    flatone_cli: Path,
    quiet_flatone: bool,
) -> int:
    cmd = [python_bin, str(flatone_cli), str(seg_id), "--output-dir", str(flatone_output_dir)]
    if quiet_flatone:
        cmd.append("--no-verbose")
    result = subprocess.run(cmd, check=False)
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ensure SWCs exist for given segment IDs and export as "
            "<segment_id>.swc in a single output folder."
        )
    )
    parser.add_argument(
        "--segment-ids-file",
        type=Path,
        default=None,
        help="Optional text file with one numeric segment ID per line",
    )
    parser.add_argument(
        "segment_ids",
        nargs="*",
        type=_parse_segment_id,
        help="Optional segment IDs (can be combined with --segment-ids-file)",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("outputs/swc_exports"),
        help="Folder where final files are written as <segment_id>.swc",
    )
    parser.add_argument(
        "--flatone-output-dir",
        type=Path,
        default=Path("outputs/all_cells"),
        help="Flatone root folder containing per-segment subdirectories",
    )
    parser.add_argument(
        "--warped",
        action="store_true",
        help="Export warped SWCs (skeleton_warped.swc) instead of raw skeleton.swc",
    )
    parser.add_argument(
        "--run-flatone-missing",
        action="store_true",
        help="Run flatone only when the required SWC is missing",
    )
    parser.add_argument(
        "--python-bin",
        default=".venv/bin/python",
        help="Python executable used to run flatone (default: %(default)s)",
    )
    parser.add_argument(
        "--flatone-cli",
        type=Path,
        default=Path("flatone/flatone/cli.py"),
        help="Path to flatone CLI script (default: %(default)s)",
    )
    parser.add_argument(
        "--quiet-flatone",
        action="store_true",
        help="Pass --no-verbose when running flatone for missing IDs",
    )
    parser.add_argument(
        "--overwrite-export",
        action="store_true",
        help="Overwrite existing exported files in --export-dir",
    )
    args = parser.parse_args()

    if args.segment_ids_file and not args.segment_ids_file.exists():
        print(f"ERROR: segment ID file not found: {args.segment_ids_file}")
        return 2
    if args.run_flatone_missing:
        if not Path(args.python_bin).exists():
            print(f"ERROR: python executable not found: {args.python_bin}")
            return 2
        if not args.flatone_cli.exists():
            print(f"ERROR: flatone CLI not found: {args.flatone_cli}")
            return 2

    try:
        seg_ids = load_segment_ids(args.segment_ids_file, args.segment_ids)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    args.export_dir.mkdir(parents=True, exist_ok=True)
    source_name = "skeleton_warped.swc" if args.warped else "skeleton.swc"

    exported = 0
    skipped = 0
    flatone_processed: list[int] = []
    failed: list[int] = []

    for seg_id in seg_ids:
        seg_dir = args.flatone_output_dir / str(seg_id)
        source = seg_dir / source_name

        if not source.exists() and args.run_flatone_missing:
            print(f"[flatone] Missing {source_name} for {seg_id}; running flatone...")
            rc = run_flatone_for_segment(
                seg_id=seg_id,
                flatone_output_dir=args.flatone_output_dir,
                python_bin=args.python_bin,
                flatone_cli=args.flatone_cli,
                quiet_flatone=args.quiet_flatone,
            )
            if rc != 0:
                print(f"[flatone] FAILED for {seg_id} (exit={rc})")
                failed.append(seg_id)
                continue
            flatone_processed.append(seg_id)

        if not source.exists():
            print(f"MISSING: {source}")
            failed.append(seg_id)
            continue

        target = args.export_dir / f"{seg_id}.swc"
        if target.exists() and not args.overwrite_export:
            skipped += 1
            print(f"SKIP: exists {target}")
            continue

        shutil.copy2(source, target)
        exported += 1
        print(f"OK: {source} -> {target}")

    print("")
    print(f"Requested IDs: {len(seg_ids)}")
    print(f"Exported files: {exported}")
    print(f"Skipped existing: {skipped}")
    print(f"Flatone processed this run: {len(flatone_processed)}")
    print(f"Failed IDs: {len(failed)}")
    if failed:
        print(f"Failed segment IDs: {', '.join(str(x) for x in failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
