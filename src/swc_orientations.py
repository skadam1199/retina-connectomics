#!/usr/bin/env python3
"""Compute orientation relationships between cells from SWC files.

This script reads SWC files directly, fits a line to the skeleton points using PCA,
and computes pairwise orientation angles between cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def load_swc_points(swc_path: Path) -> np.ndarray:
    """Load x,y,z points from SWC file."""
    points = []
    with open(swc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    points.append([x, y, z])
                except ValueError:
                    continue
    return np.array(points)


def principal_axis(points: np.ndarray) -> np.ndarray:
    """Compute principal axis using PCA."""
    if len(points) < 2:
        return np.array([1.0, 0.0, 0.0])  # default axis

    centered = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axis = eigvecs[:, order[0]]

    # Canonical sign
    for comp in axis:
        if abs(comp) > 1e-12:
            return axis if comp > 0 else -axis
    return axis


def angle_between_axes(axis1: np.ndarray, axis2: np.ndarray) -> float:
    """Compute undirected angle between two axes in degrees."""
    a = axis1 / np.linalg.norm(axis1)
    b = axis2 / np.linalg.norm(axis2)
    cosv = abs(np.dot(a, b))
    return np.degrees(np.arccos(np.clip(cosv, 0.0, 1.0)))


def classify_relation(angle_deg: float) -> str:
    if angle_deg < 15.0:
        return "aligned"
    elif angle_deg < 45.0:
        return "oblique"
    else:
        return "near-orthogonal"


def main():
    parser = argparse.ArgumentParser(description="Compute orientations from SWC files")
    parser.add_argument("segment_ids_file", type=Path, help="File with segment IDs")
    parser.add_argument("swc_dir", type=Path, help="Directory containing SWC files")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    args = parser.parse_args()

    # Load segment IDs
    with open(args.segment_ids_file, 'r') as f:
        segment_ids = [int(line.strip()) for line in f if line.strip() and not line.startswith('#')]

    print(f"Processing {len(segment_ids)} segment IDs")

    # Load orientations
    orientations = {}
    for seg_id in segment_ids:
        swc_path = args.swc_dir / f"{seg_id}.swc"
        if not swc_path.exists():
            print(f"Warning: SWC not found for {seg_id}")
            continue

        points = load_swc_points(swc_path)
        if len(points) == 0:
            print(f"Warning: No points in SWC for {seg_id}")
            continue

        axis = principal_axis(points)
        orientations[seg_id] = {
            'axis': axis,
            'n_points': len(points)
        }
        print(f"Processed {seg_id}: {len(points)} points")

    # Compute pairwise angles
    pairs = []
    for i, id1 in enumerate(segment_ids):
        if id1 not in orientations:
            continue
        for id2 in segment_ids[i+1:]:
            if id2 not in orientations:
                continue
            angle = angle_between_axes(orientations[id1]['axis'], orientations[id2]['axis'])
            relation = classify_relation(angle)
            pairs.append({
                'id_a': id1,
                'id_b': id2,
                'angle_deg': angle,
                'relation': relation
            })

    # Create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write per-cell CSV
    with open(args.output_dir / "per_cell.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['segment_id', 'n_points', 'axis_x', 'axis_y', 'axis_z'])
        for seg_id, data in orientations.items():
            writer.writerow([seg_id, data['n_points'], *data['axis']])

    # Write pairwise CSV
    with open(args.output_dir / "pairwise.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['segment_id_a', 'segment_id_b', 'angle_deg', 'relation'])
        for pair in pairs:
            writer.writerow([pair['id_a'], pair['id_b'], pair['angle_deg'], pair['relation']])

    # Write summary JSON
    summary = {
        'total_ids': len(segment_ids),
        'processed_ids': len(orientations),
        'pairs_computed': len(pairs),
        'outputs': {
            'per_cell_csv': str(args.output_dir / "per_cell.csv"),
            'pairwise_csv': str(args.output_dir / "pairwise.csv")
        }
    }
    with open(args.output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()