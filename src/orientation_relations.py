#!/usr/bin/env python3
"""Compute orientation relationships between cells from segment IDs.

Workflow:
1) Read segment IDs from a text file (one numeric ID per line).
2) Optionally run flatone for IDs that do not yet have skeleton outputs.
3) Load skeleton node coordinates from NPZ files.
4) Compute per-cell principal orientation and pairwise orientation angles.
5) Write CSV/JSON outputs for reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CellOrientation:
    segment_id: int
    skeleton_path: Path
    n_nodes: int
    axis3d_x: float
    axis3d_y: float
    axis3d_z: float
    axis_xy_x: float
    axis_xy_y: float
    azimuth_deg: float
    elevation_deg: float
    elongation_3d: float
    elongation_xy: float


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm


def _canonical_axis(v: np.ndarray) -> np.ndarray:
    """Fix axis sign so orientation vectors are stable across runs.

    Eigenvectors are sign-ambiguous (+v and -v are equivalent), so we enforce
    a deterministic sign: first non-negligible component is positive.
    """
    v = _normalize(v)
    for comp in v:
        if abs(float(comp)) > 1e-12:
            return v if comp > 0 else -v
    return v


def _principal_axis(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (axis, eigenvalues_desc) from PCA of centered 3D points."""
    centered = nodes - nodes.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    axis = eigvecs[:, order[0]]
    return _canonical_axis(axis), eigvals


def _principal_axis_xy(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (axis, eigenvalues_desc) from PCA of centered XY points."""
    xy = nodes[:, :2]
    centered = xy - xy.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    axis = eigvecs[:, order[0]]
    return _canonical_axis(axis), eigvals


def _angle_undirected_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two orientation axes in [0, 90] degrees."""
    a = _normalize(v1)
    b = _normalize(v2)
    cosv = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def _relation_label(angle_deg: float) -> str:
    if angle_deg < 15.0:
        return "aligned"
    if angle_deg < 45.0:
        return "oblique"
    return "near-orthogonal"


def load_segment_ids(path: Path) -> list[int]:
    ids: list[int] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.isdigit():
            raise ValueError(f"{path}:{lineno}: non-numeric segment ID: {line!r}")
        ids.append(int(line))
    # Preserve original order but de-duplicate.
    seen: set[int] = set()
    unique: list[int] = []
    for seg_id in ids:
        if seg_id not in seen:
            seen.add(seg_id)
            unique.append(seg_id)
    return unique


def find_skeleton_npz(seg_id: int, output_root: Path, kind: str) -> Path | None:
    seg_dir = output_root / str(seg_id)
    warped = seg_dir / "skeleton_warped.npz"
    raw = seg_dir / "skeleton.npz"
    if kind == "warped":
        return warped if warped.exists() else None
    if kind == "raw":
        return raw if raw.exists() else None
    # auto
    if warped.exists():
        return warped
    if raw.exists():
        return raw
    return None


def maybe_run_flatone(
    seg_ids: list[int],
    output_root: Path,
    skeleton_kind: str,
    python_bin: str,
    flatone_cli: Path,
    run_missing: bool,
    warp_mesh: bool,
    verbose_flatone: bool,
) -> tuple[list[int], list[int]]:
    """Run flatone for missing IDs if requested. Returns (processed, failed)."""
    processed: list[int] = []
    failed: list[int] = []
    if not run_missing:
        return processed, failed

    for seg_id in seg_ids:
        if find_skeleton_npz(seg_id, output_root, skeleton_kind) is not None:
            continue
        cmd = [python_bin, str(flatone_cli), str(seg_id), "--output-dir", str(output_root)]
        if warp_mesh:
            cmd.append("--warp-mesh")
        if not verbose_flatone:
            cmd.append("--no-verbose")
        print(f"[flatone] Running for {seg_id} ...")
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            processed.append(seg_id)
        else:
            failed.append(seg_id)
            print(f"[flatone] FAILED for {seg_id} (exit={result.returncode})")
    return processed, failed


def compute_cell_orientation(seg_id: int, skeleton_path: Path) -> CellOrientation:
    with np.load(skeleton_path, allow_pickle=True) as z:
        if "nodes" not in z:
            raise ValueError(f"{skeleton_path} does not contain 'nodes'")
        nodes = np.asarray(z["nodes"], dtype=float)

    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(f"{skeleton_path}: unexpected nodes shape {nodes.shape}")
    if nodes.shape[0] < 3:
        raise ValueError(f"{skeleton_path}: not enough nodes ({nodes.shape[0]})")

    axis3d, eig3 = _principal_axis(nodes)
    axis_xy, eig2 = _principal_axis_xy(nodes)

    azimuth = math.degrees(math.atan2(float(axis3d[1]), float(axis3d[0]))) % 180.0
    elevation = math.degrees(
        math.atan2(float(axis3d[2]), math.hypot(float(axis3d[0]), float(axis3d[1])))
    )

    # Avoid divide-by-zero for nearly isotropic objects.
    elong3 = float(eig3[0] / max(float(eig3[1]), 1e-12))
    elong2 = float(eig2[0] / max(float(eig2[1]), 1e-12))

    return CellOrientation(
        segment_id=seg_id,
        skeleton_path=skeleton_path,
        n_nodes=int(nodes.shape[0]),
        axis3d_x=float(axis3d[0]),
        axis3d_y=float(axis3d[1]),
        axis3d_z=float(axis3d[2]),
        axis_xy_x=float(axis_xy[0]),
        axis_xy_y=float(axis_xy[1]),
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        elongation_3d=elong3,
        elongation_xy=elong2,
    )


def write_per_cell_csv(rows: list[CellOrientation], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "segment_id",
                "skeleton_path",
                "n_nodes",
                "axis3d_x",
                "axis3d_y",
                "axis3d_z",
                "axis_xy_x",
                "axis_xy_y",
                "azimuth_deg",
                "elevation_deg",
                "elongation_3d",
                "elongation_xy",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.segment_id,
                    str(r.skeleton_path),
                    r.n_nodes,
                    r.axis3d_x,
                    r.axis3d_y,
                    r.axis3d_z,
                    r.axis_xy_x,
                    r.axis_xy_y,
                    r.azimuth_deg,
                    r.elevation_deg,
                    r.elongation_3d,
                    r.elongation_xy,
                ]
            )


def write_pairwise_csv(rows: list[CellOrientation], out_path: Path) -> dict[str, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"aligned": 0, "oblique": 0, "near-orthogonal": 0}
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "segment_id_a",
                "segment_id_b",
                "angle_3d_deg",
                "angle_xy_deg",
                "relation_3d",
                "relation_xy",
            ]
        )
        for i in range(len(rows)):
            a = rows[i]
            va3 = np.array([a.axis3d_x, a.axis3d_y, a.axis3d_z], dtype=float)
            va2 = np.array([a.axis_xy_x, a.axis_xy_y], dtype=float)
            for j in range(i + 1, len(rows)):
                b = rows[j]
                vb3 = np.array([b.axis3d_x, b.axis3d_y, b.axis3d_z], dtype=float)
                vb2 = np.array([b.axis_xy_x, b.axis_xy_y], dtype=float)
                ang3 = _angle_undirected_deg(va3, vb3)
                ang2 = _angle_undirected_deg(va2, vb2)
                rel3 = _relation_label(ang3)
                rel2 = _relation_label(ang2)
                counts[rel3] += 1
                writer.writerow([a.segment_id, b.segment_id, ang3, ang2, rel3, rel2])
    return counts


def write_angle_matrix_csv(rows: list[CellOrientation], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = [r.segment_id for r in rows]
    axes = [np.array([r.axis3d_x, r.axis3d_y, r.axis3d_z], dtype=float) for r in rows]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_id", *ids])
        for i, sid in enumerate(ids):
            vals = []
            for j in range(len(ids)):
                if i == j:
                    vals.append(0.0)
                else:
                    vals.append(_angle_undirected_deg(axes[i], axes[j]))
            writer.writerow([sid, *vals])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute orientation relationships across cells listed in a segment-ID file."
        )
    )
    p.add_argument(
        "--segment-ids",
        type=Path,
        required=True,
        help="Text file with one numeric segment ID per line",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/test1"),
        help="Flatone output root containing one folder per segment ID",
    )
    p.add_argument(
        "--skeleton-kind",
        choices=["auto", "warped", "raw"],
        default="auto",
        help="Which skeleton NPZ to use per segment (default: auto=prefer warped)",
    )
    p.add_argument(
        "--run-flatone-missing",
        action="store_true",
        help="Run flatone for IDs that do not yet have a skeleton NPZ",
    )
    p.add_argument(
        "--python-bin",
        default=".venv/bin/python",
        help="Python executable for running flatone (default: %(default)s)",
    )
    p.add_argument(
        "--flatone-cli",
        type=Path,
        default=Path("flatone/flatone/cli.py"),
        help="Path to flatone CLI script (default: %(default)s)",
    )
    p.add_argument(
        "--warp-mesh",
        action="store_true",
        help="When running missing flatone jobs, also warp mesh (slower)",
    )
    p.add_argument(
        "--quiet-flatone",
        action="store_true",
        help="When running missing flatone jobs, pass --no-verbose",
    )
    p.add_argument(
        "--out-per-cell-csv",
        type=Path,
        default=Path("outputs/orientation/per_cell_orientation.csv"),
        help="Output CSV for per-cell orientations",
    )
    p.add_argument(
        "--out-pairwise-csv",
        type=Path,
        default=Path("outputs/orientation/pairwise_orientation_angles.csv"),
        help="Output CSV for pairwise orientation relationships",
    )
    p.add_argument(
        "--out-matrix-csv",
        type=Path,
        default=Path("outputs/orientation/orientation_angle_matrix.csv"),
        help="Output CSV for NxN 3D orientation angle matrix",
    )
    p.add_argument(
        "--out-summary-json",
        type=Path,
        default=Path("outputs/orientation/orientation_summary.json"),
        help="Output JSON with run summary and missing/failed IDs",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    if not args.segment_ids.exists():
        print(f"ERROR: segment ID file not found: {args.segment_ids}")
        return 2

    seg_ids = load_segment_ids(args.segment_ids)
    if not seg_ids:
        print(f"ERROR: no IDs found in {args.segment_ids}")
        return 2

    processed, failed = maybe_run_flatone(
        seg_ids=seg_ids,
        output_root=args.output_dir,
        skeleton_kind=args.skeleton_kind,
        python_bin=args.python_bin,
        flatone_cli=args.flatone_cli,
        run_missing=args.run_flatone_missing,
        warp_mesh=args.warp_mesh,
        verbose_flatone=not args.quiet_flatone,
    )

    per_cell: list[CellOrientation] = []
    missing: list[int] = []
    errored: list[dict[str, str | int]] = []

    for seg_id in seg_ids:
        skel = find_skeleton_npz(seg_id, args.output_dir, args.skeleton_kind)
        if skel is None:
            missing.append(seg_id)
            continue
        try:
            per_cell.append(compute_cell_orientation(seg_id, skel))
        except Exception as exc:
            errored.append(
                {
                    "segment_id": seg_id,
                    "skeleton_path": str(skel),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if len(per_cell) < 2:
        print("ERROR: need at least 2 valid cells to compute pairwise relationships.")
        print(f"Loaded valid cells: {len(per_cell)}")
        return 1

    per_cell.sort(key=lambda x: x.segment_id)
    write_per_cell_csv(per_cell, args.out_per_cell_csv)
    relation_counts = write_pairwise_csv(per_cell, args.out_pairwise_csv)
    write_angle_matrix_csv(per_cell, args.out_matrix_csv)

    summary = {
        "segment_id_file": str(args.segment_ids),
        "requested_ids": len(seg_ids),
        "cells_with_orientations": len(per_cell),
        "missing_skeleton_ids": missing,
        "flatone_processed_ids": processed,
        "flatone_failed_ids": failed,
        "errored_ids": errored,
        "relation_counts_3d": relation_counts,
        "outputs": {
            "per_cell_csv": str(args.out_per_cell_csv),
            "pairwise_csv": str(args.out_pairwise_csv),
            "matrix_csv": str(args.out_matrix_csv),
        },
    }
    args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Requested IDs: {len(seg_ids)}")
    print(f"Cells with orientation computed: {len(per_cell)}")
    print(f"Missing skeletons: {len(missing)}")
    print(f"Flatone processed this run: {len(processed)}")
    print(f"Flatone failures this run: {len(failed)}")
    print(f"Cells with parse/compute errors: {len(errored)}")
    print(f"Wrote per-cell CSV: {args.out_per_cell_csv}")
    print(f"Wrote pairwise CSV: {args.out_pairwise_csv}")
    print(f"Wrote angle matrix CSV: {args.out_matrix_csv}")
    print(f"Wrote summary JSON: {args.out_summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
