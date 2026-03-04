#!/usr/bin/env python3
"""Validate whether a segment ID appears in a flat annotations CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_ids(csv_path: Path) -> set[str]:
    """Load unique IDs from segment_1 and segment_2 columns."""
    ids: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ("segment_1", "segment_2"):
                value = (row.get(key) or "").strip()
                if value:
                    ids.add(value)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether one or more segment IDs exist in annotations_flat.csv."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/Users/supriyanagnathkadam/Downloads/annotations_flat.csv"),
        help="Path to flat annotations CSV (default: %(default)s)",
    )
    parser.add_argument(
        "segment_ids",
        nargs="+",
        help="One or more segment IDs to validate",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}")
        return 2

    ids = load_ids(args.csv)
    print(f"Loaded {len(ids)} unique IDs from {args.csv}")

    missing = 0
    for seg_id in args.segment_ids:
        if seg_id in ids:
            print(f"VALID   {seg_id}")
        else:
            print(f"INVALID {seg_id}")
            missing += 1

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
