#!/usr/bin/env python3
"""Validate CAVE token access for the EyeWire II datastack."""

from __future__ import annotations

import argparse
from typing import Iterable


def _fingerprint(token: str) -> str:
    if len(token) <= 8:
        return "*" * len(token)
    return f"{token[:4]}...{token[-4:]}"


def _pick_table(tables: Iterable[str], preferred: str | None) -> str | None:
    table_list = list(tables)
    if not table_list:
        return None
    if preferred and preferred in table_list:
        return preferred
    return table_list[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether your stored CAVE token can access EyeWire II "
            "(stroeh_mouse_retina)."
        )
    )
    parser.add_argument(
        "--datastack",
        default="stroeh_mouse_retina",
        help="Datastack name (default: %(default)s)",
    )
    parser.add_argument(
        "--table",
        default=None,
        help="Optional materialization table to query (default: first table)",
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Only verify datastack info and table listing",
    )
    args = parser.parse_args()

    try:
        from caveclient import CAVEclient
    except Exception as exc:
        print(f"ERROR: failed to import caveclient: {type(exc).__name__}: {exc}")
        return 2

    try:
        client = CAVEclient(args.datastack)
    except Exception as exc:
        print(
            f"ERROR: failed to initialize CAVEclient for '{args.datastack}': "
            f"{type(exc).__name__}: {exc}"
        )
        return 3

    token = getattr(client.auth, "token", None)
    if not token:
        print("FAIL: no token found in local CloudVolume/CAVE auth config.")
        print("Run `flatone add-token <TOKEN>` first.")
        return 4

    print(f"Token found: {_fingerprint(token)}")

    try:
        info = client.info.get_datastack_info()
        print(f"Datastack info: OK ({args.datastack})")
        print(f"Segmentation source: {info.get('segmentation_source')}")
    except Exception as exc:
        print(f"FAIL: datastack info call failed: {type(exc).__name__}: {exc}")
        return 5

    try:
        tables = client.materialize.get_tables()
        print(f"Materialization tables: OK ({len(tables)} tables)")
    except Exception as exc:
        print(f"FAIL: materialize.get_tables failed: {type(exc).__name__}: {exc}")
        return 6

    if args.skip_query:
        print("SUCCESS: token works for authenticated dataset access.")
        return 0

    table = _pick_table(tables, args.table)
    if table is None:
        print("FAIL: no materialization tables available to test query.")
        return 7

    try:
        df = client.materialize.query_table(table, limit=1)
        print(f"Query test: OK (table='{table}', rows={len(df)})")
    except Exception as exc:
        print(
            f"FAIL: materialize.query_table failed for '{table}': "
            f"{type(exc).__name__}: {exc}"
        )
        return 8

    print("SUCCESS: token works for EyeWire II API access.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
