#!/usr/bin/env python3
"""Compute XY convex hull of a post-synaptic cell and identify pre-SAC segments within it.

Workflow
--------
1. Load the post cell's skeleton NPZ → compute 2D convex hull in the XY plane.
2. Parse the annotations CSV to find all pre-SAC partners of the post cell.
   A row qualifies when:
     - the post cell's segment ID appears in --post-col  (default: segment_2)
     - the description starts with "<sac-label>->"       (default: "SAC->")
   The segment in --pre-col (default: segment_1) is treated as the pre-SAC ID.
3. For each pre-SAC segment, obtain a representative XY position:
     - Priority 1: centroid of its skeleton NPZ nodes (warped → raw).
     - Priority 2: mean XY of the synapse annotation points in the CSV.
     - Otherwise:  no position — segment is listed but untestable.
4. Test containment: which pre-SAC centroids lie inside the post cell's hull.
5. Write results to CSV and/or JSON.  Optionally save a plot.

Column convention
-----------------
The default assumes segment_1 = pre-synaptic, segment_2 = post-synaptic.
If your CSV uses the opposite convention pass --pre-col segment_2 --post-col segment_1.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Skeleton helpers (shared with orientation_relations.py conventions)
# ---------------------------------------------------------------------------

def _load_nodes(npz_path: Path) -> np.ndarray:
    """Return (N, 3) float array of skeleton node positions from a NPZ file."""
    with np.load(npz_path, allow_pickle=True) as z:
        if "nodes" not in z:
            raise ValueError(f"{npz_path}: missing 'nodes' key")
        nodes = np.asarray(z["nodes"], dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(f"{npz_path}: unexpected nodes shape {nodes.shape}")
    return nodes


def _find_npz(seg_id: int, output_root: Path, kind: str) -> Path | None:
    """Locate the best skeleton NPZ for a segment (warped preferred in 'auto' mode)."""
    seg_dir = output_root / str(seg_id)
    warped = seg_dir / "skeleton_warped.npz"
    raw = seg_dir / "skeleton.npz"
    if kind == "warped":
        return warped if warped.exists() else None
    if kind == "raw":
        return raw if raw.exists() else None
    return warped if warped.exists() else (raw if raw.exists() else None)


# ---------------------------------------------------------------------------
# Convex hull
# ---------------------------------------------------------------------------

def build_hull(nodes: np.ndarray) -> "ConvexHull":
    """
    Build a scipy 2D ConvexHull from the XY projection of skeleton nodes.

    In 2D scipy convention:
      hull.volume  = area      of the convex hull polygon
      hull.area    = perimeter of the convex hull polygon
    """
    if not _HAS_SCIPY:
        raise ImportError(
            "scipy is required for convex hull computation.\n"
            "Install with:  pip install scipy"
        )
    xy = np.unique(nodes[:, :2], axis=0)
    if len(xy) < 3:
        raise ValueError(
            f"Only {len(xy)} unique XY points — need at least 3 to form a hull."
        )
    return ConvexHull(xy)


def point_in_hull(xy: np.ndarray, hull: "ConvexHull") -> bool:
    """
    Test whether 2D point *xy* lies inside (or on the boundary of) the hull.

    Uses the half-space representation: a point is inside iff it satisfies
    all inequalities  A @ p + b <= 0  (with a small numerical tolerance).
    """
    # hull.equations row = [a, b, c]  →  a*x + b*y + c <= 0 for interior
    return bool(np.all(hull.equations @ np.append(xy, 1.0) <= 1e-10))


# ---------------------------------------------------------------------------
# Annotation CSV parsing
# ---------------------------------------------------------------------------

def _parse_annotations(
    csv_path: Path,
    post_seg_id: int,
    sac_label: str,
    pre_col: str,
    post_col: str,
) -> dict[int, list[tuple[float, float, float]]]:
    """
    Scan the annotations CSV and return pre-SAC partners of *post_seg_id*.

    Returns
    -------
    dict mapping sac_seg_id → list of (x, y, z) annotation point coordinates.
    Coordinates are available when the CSV has x / y / z columns (synapse location);
    otherwise the list is empty.  The list is used as a fallback position when
    no skeleton NPZ is available for the SAC.
    """
    post_str = str(post_seg_id)
    sac_prefix = f"{sac_label}->"
    partners: dict[int, list[tuple[float, float, float]]] = {}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        has_coords = {"x", "y", "z"}.issubset(fieldnames)

        for row in reader:
            desc = (row.get("description") or "").strip()
            pre = (row.get(pre_col) or "").strip()
            post = (row.get(post_col) or "").strip()

            if post != post_str:
                continue
            if not desc.startswith(sac_prefix):
                continue
            if not pre or not pre.isdigit():
                continue

            sac_id = int(pre)
            if has_coords:
                try:
                    pt: tuple[float, float, float] = (
                        float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                    )
                    partners.setdefault(sac_id, []).append(pt)
                    continue
                except (ValueError, TypeError):
                    pass
            partners.setdefault(sac_id, [])

    return partners


# ---------------------------------------------------------------------------
# Representative position for a SAC segment
# ---------------------------------------------------------------------------

def _representative_xy(
    sac_id: int,
    output_root: Path | None,
    skeleton_kind: str,
    annotation_pts: list[tuple[float, float, float]],
) -> tuple[np.ndarray | None, str]:
    """
    Return (xy_position, source_label) for a SAC segment.

    Priority
    --------
    1. Centroid of skeleton nodes (warped preferred, then raw).
    2. Mean XY of synapse annotation points from the CSV.
    3. None — no position available.
    """
    if output_root is not None:
        npz = _find_npz(sac_id, output_root, skeleton_kind)
        if npz is not None:
            try:
                nodes = _load_nodes(npz)
                return nodes[:, :2].mean(axis=0), "skeleton_centroid"
            except Exception:
                pass

    if annotation_pts:
        pts = np.array([[x, y] for x, y, _ in annotation_pts], dtype=float)
        return pts.mean(axis=0), "annotation_mean"

    return None, "none"


# ---------------------------------------------------------------------------
# Optional plot
# ---------------------------------------------------------------------------

def _save_plot(
    post_seg_id: int,
    hull: "ConvexHull",
    results: list[dict],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw convex hull polygon
    verts = hull.points[hull.vertices]
    poly = Polygon(verts, closed=True, facecolor="steelblue", alpha=0.15, edgecolor="steelblue", linewidth=1.5)
    ax.add_patch(poly)

    # Plot SAC positions
    for r in results:
        if r["x"] is None:
            continue
        color = "green" if r["status"] == "inside" else "red"
        marker = "o" if r["status"] == "inside" else "x"
        ax.scatter(r["x"], r["y"], c=color, marker=marker, s=60, zorder=3)
        ax.annotate(
            str(r["sac_seg_id"]),
            (r["x"], r["y"]),
            fontsize=6,
            ha="left",
            va="bottom",
            color=color,
        )

    # Legends via proxy artists
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8, label="inside hull"),
        Line2D([0], [0], marker="x", color="red", markersize=8, label="outside hull"),
        Polygon([[0, 0]], facecolor="steelblue", alpha=0.3, label="post cell hull"),
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8)

    ax.set_aspect("equal")
    ax.set_title(f"Post cell {post_seg_id} — pre-SAC segments", fontsize=11)
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.autoscale_view()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute XY convex hull of a post-synaptic cell and identify "
            "pre-SAC segments whose centroid falls within the hull."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # required
    p.add_argument("--post-seg-id", type=int, required=True,
                   help="Segment ID of the post-synaptic cell")
    p.add_argument("--csv", type=Path, required=True,
                   help="Neuroglancer annotations CSV (annotations_flat.csv)")

    # skeleton
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Flatone output root (per-segment subdirectories with NPZ files)")
    p.add_argument("--skeleton-kind", choices=["auto", "warped", "raw"], default="auto",
                   help="Which skeleton NPZ to use: auto (default, prefers warped), warped, raw")

    # annotation convention
    p.add_argument("--sac-label", default="SAC",
                   help="Cell-type label for SAC in description strings (default: SAC)")
    p.add_argument("--pre-col", default="segment_1",
                   help="CSV column holding the pre-synaptic segment ID (default: segment_1)")
    p.add_argument("--post-col", default="segment_2",
                   help="CSV column holding the post-synaptic segment ID (default: segment_2)")

    # outputs
    p.add_argument("--out-csv", type=Path, default=None,
                   help="Write per-SAC results table to this CSV")
    p.add_argument("--out-json", type=Path, default=None,
                   help="Write full results + summary to this JSON")
    p.add_argument("--out-plot", type=Path, default=None,
                   help="Save XY hull + SAC positions plot to this PNG")

    return p


def main() -> int:
    args = _build_parser().parse_args()

    if not _HAS_SCIPY:
        print("ERROR: scipy is required.  Install with: pip install scipy")
        return 2

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}")
        return 2

    if args.output_dir is not None and not args.output_dir.exists():
        print(f"ERROR: --output-dir not found: {args.output_dir}")
        return 2

    # ------------------------------------------------------------------
    # 1. Load post cell skeleton and build XY convex hull
    # ------------------------------------------------------------------
    post_npz: Path | None = None
    if args.output_dir is not None:
        post_npz = _find_npz(args.post_seg_id, args.output_dir, args.skeleton_kind)

    if post_npz is None:
        print(f"ERROR: no skeleton NPZ found for post cell {args.post_seg_id}")
        if args.output_dir:
            print(f"  Searched: {args.output_dir / str(args.post_seg_id)}/")
        print("  Run  flatone <SEGMENT_ID>  first to generate skeleton files.")
        return 1

    try:
        post_nodes = _load_nodes(post_npz)
    except Exception as exc:
        print(f"ERROR loading post cell skeleton: {exc}")
        return 1

    try:
        hull = build_hull(post_nodes)
    except ValueError as exc:
        print(f"ERROR building hull: {exc}")
        return 1

    hull_area = float(hull.volume)   # scipy 2D: .volume = polygon area
    hull_perim = float(hull.area)    # scipy 2D: .area   = perimeter

    print(f"Post cell {args.post_seg_id}")
    print(f"  Skeleton  : {post_npz}")
    print(f"  Nodes     : {len(post_nodes)}")
    print(f"  Hull verts: {len(hull.vertices)}")
    print(f"  Hull area : {hull_area:.2f} (XY units²)")
    print(f"  Hull perim: {hull_perim:.2f} (XY units)")

    # ------------------------------------------------------------------
    # 2. Find pre-SAC partners from annotations CSV
    # ------------------------------------------------------------------
    print(
        f"\nScanning {args.csv.name} for pre-{args.sac_label} partners "
        f"(post_col={args.post_col}, pre_col={args.pre_col}) …"
    )
    sac_partners = _parse_annotations(
        args.csv,
        args.post_seg_id,
        args.sac_label,
        args.pre_col,
        args.post_col,
    )
    print(f"  Found {len(sac_partners)} unique pre-{args.sac_label} segment(s)")

    if not sac_partners:
        print(
            f"\nNo pre-{args.sac_label} partners found.\n"
            f"  • Check that --pre-col / --post-col match your CSV columns.\n"
            f"  • Check that descriptions use the '{args.sac_label}->' prefix.\n"
            f"  • Confirm that {args.post_seg_id} appears in the '{args.post_col}' column."
        )
        return 1

    # ------------------------------------------------------------------
    # 3. Get representative XY position for each SAC and test containment
    # ------------------------------------------------------------------
    results: list[dict] = []
    n_inside = 0
    n_outside = 0
    n_no_pos = 0

    for sac_id, ann_pts in sac_partners.items():
        xy, source = _representative_xy(
            sac_id, args.output_dir, args.skeleton_kind, ann_pts
        )

        if xy is None:
            status = "no_position"
            in_hull_val: bool | None = None
            n_no_pos += 1
        else:
            in_hull_val = point_in_hull(xy, hull)
            if in_hull_val:
                status = "inside"
                n_inside += 1
            else:
                status = "outside"
                n_outside += 1

        results.append({
            "sac_seg_id": sac_id,
            "n_synapses_onto_post": len(ann_pts),
            "position_source": source,
            "x": float(xy[0]) if xy is not None else None,
            "y": float(xy[1]) if xy is not None else None,
            "in_hull": in_hull_val,
            "status": status,
        })

    # Sort: inside first, then outside, then no_position; within group by ID
    _order = {"inside": 0, "outside": 1, "no_position": 2}
    results.sort(key=lambda r: (_order[r["status"]], r["sac_seg_id"]))

    # ------------------------------------------------------------------
    # 4. Print summary
    # ------------------------------------------------------------------
    print(f"\n{'─' * 55}")
    print(f"  Pre-{args.sac_label} partners found : {len(results)}")
    print(f"  Inside hull                : {n_inside}")
    print(f"  Outside hull               : {n_outside}")
    print(f"  No position (untestable)   : {n_no_pos}")
    print(f"{'─' * 55}")

    if n_inside:
        print(f"\nPre-{args.sac_label} segments INSIDE hull:")
        print(f"  {'segment_id':<22}  {'synapses':>8}  {'pos_source':<20}  {'x':>10}  {'y':>10}")
        for r in results:
            if r["status"] != "inside":
                continue
            print(
                f"  {r['sac_seg_id']:<22}  {r['n_synapses_onto_post']:>8}  "
                f"{r['position_source']:<20}  {r['x']:>10.2f}  {r['y']:>10.2f}"
            )

    if n_no_pos:
        print(
            f"\nNote: {n_no_pos} SAC segment(s) could not be tested (no skeleton NPZ "
            f"and no annotation coordinates). Run flatone for those IDs and rerun with --output-dir."
        )

    # ------------------------------------------------------------------
    # 5. Write outputs
    # ------------------------------------------------------------------
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWrote CSV : {args.out_csv}")

    summary = {
        "post_seg_id": args.post_seg_id,
        "post_skeleton": str(post_npz),
        "post_n_nodes": int(post_nodes.shape[0]),
        "hull_n_vertices": int(len(hull.vertices)),
        "hull_area_xy": hull_area,
        "hull_perimeter_xy": hull_perim,
        "sac_label": args.sac_label,
        "pre_col": args.pre_col,
        "post_col": args.post_col,
        "total_pre_sac": len(results),
        "n_inside": n_inside,
        "n_outside": n_outside,
        "n_no_position": n_no_pos,
        "inside_seg_ids": [r["sac_seg_id"] for r in results if r["status"] == "inside"],
        "results": results,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {args.out_json}")

    if args.out_plot:
        _save_plot(args.post_seg_id, hull, results, args.out_plot)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
