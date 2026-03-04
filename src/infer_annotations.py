#!/usr/bin/env python3
"""Compute lightweight inferences from local Neuroglancer annotation exports."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-]+")


@dataclass
class AnnotationStats:
    rows_total: int
    rows_with_description: int
    rows_with_two_segments: int
    rows_with_single_segment: int
    rows_with_no_segment: int
    unique_descriptions: int
    unique_segments: int


def _as_clean(value: str | None) -> str:
    return (value or "").strip()


def parse_transition(description: str) -> tuple[str, str] | None:
    """
    Parse labels like `BK->CB` or `BK->GC (asymm)` to (source, target).
    """
    if "->" not in description:
        return None
    left, right = description.split("->", maxsplit=1)
    m_left = TOKEN_RE.search(left.strip())
    m_right = TOKEN_RE.search(right.strip())
    if not m_left or not m_right:
        return None
    return m_left.group(0), m_right.group(0)


def build_transition_matrix(
    transition_pair_counts: Counter[tuple[str, str]],
) -> dict[str, Any]:
    classes = sorted(
        {
            cls
            for source, target in transition_pair_counts
            for cls in (source, target)
        }
    )
    by_source: dict[str, dict[str, int]] = {
        source: {target: 0 for target in classes} for source in classes
    }
    for (source, target), count in transition_pair_counts.items():
        by_source[source][target] = count

    row_totals = {source: sum(by_source[source].values()) for source in classes}
    col_totals = {
        target: sum(by_source[source][target] for source in classes) for target in classes
    }

    return {
        "classes": classes,
        "by_source": by_source,
        "row_totals": row_totals,
        "column_totals": col_totals,
    }


def write_transition_matrix_csv(matrix: dict[str, Any], out_path: Path) -> None:
    classes: list[str] = matrix["classes"]
    by_source: dict[str, dict[str, int]] = matrix["by_source"]
    row_totals: dict[str, int] = matrix["row_totals"]
    col_totals: dict[str, int] = matrix["column_totals"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_class", *classes, "row_total"])
        for source in classes:
            writer.writerow(
                [source, *(by_source[source][target] for target in classes), row_totals[source]]
            )
        writer.writerow(
            ["_column_total", *(col_totals[target] for target in classes), sum(col_totals.values())]
        )


def parse_csv(csv_path: Path) -> dict[str, Any]:
    description_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    transition_pair_counts: Counter[tuple[str, str]] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    segment_mentions: Counter[str] = Counter()
    segment_partners: dict[str, set[str]] = defaultdict(set)

    rows_total = 0
    rows_with_description = 0
    rows_with_two_segments = 0
    rows_with_single_segment = 0
    rows_with_no_segment = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1

            description = _as_clean(row.get("description"))
            if description:
                rows_with_description += 1
                description_counts[description] += 1
                transition = parse_transition(description)
                if transition:
                    source, target = transition
                    transition_counts[f"{source}->{target}"] += 1
                    transition_pair_counts[(source, target)] += 1

            seg1 = _as_clean(row.get("segment_1"))
            seg2 = _as_clean(row.get("segment_2"))

            if seg1 and seg2:
                rows_with_two_segments += 1
                pair = tuple(sorted((seg1, seg2), key=int))
                pair_counts[pair] += 1
                segment_mentions[seg1] += 1
                segment_mentions[seg2] += 1
                segment_partners[seg1].add(seg2)
                segment_partners[seg2].add(seg1)
            elif seg1 or seg2:
                rows_with_single_segment += 1
                segment_mentions[seg1 or seg2] += 1
            else:
                rows_with_no_segment += 1

    stats = AnnotationStats(
        rows_total=rows_total,
        rows_with_description=rows_with_description,
        rows_with_two_segments=rows_with_two_segments,
        rows_with_single_segment=rows_with_single_segment,
        rows_with_no_segment=rows_with_no_segment,
        unique_descriptions=len(description_counts),
        unique_segments=len(segment_mentions),
    )

    top_hubs_by_mentions = segment_mentions.most_common()
    top_hubs_by_partners = sorted(
        (
            {"segment_id": sid, "unique_partners": len(partners)}
            for sid, partners in segment_partners.items()
        ),
        key=lambda x: x["unique_partners"],
        reverse=True,
    )

    repeated_bidirectional_labels = []
    for label, count in transition_counts.items():
        left, right = label.split("->", maxsplit=1)
        reverse = f"{right}->{left}"
        if reverse in transition_counts and label < reverse:
            repeated_bidirectional_labels.append(
                {
                    "forward": label,
                    "forward_count": count,
                    "reverse": reverse,
                    "reverse_count": transition_counts[reverse],
                }
            )
    repeated_bidirectional_labels.sort(
        key=lambda x: x["forward_count"] + x["reverse_count"], reverse=True
    )
    transition_matrix = build_transition_matrix(transition_pair_counts)

    return {
        "annotation_stats": stats.__dict__,
        "top_descriptions": [
            {"description": desc, "count": cnt}
            for desc, cnt in description_counts.most_common()
        ],
        "top_transitions": [
            {"transition": tr, "count": cnt}
            for tr, cnt in transition_counts.most_common()
        ],
        "top_pairs": [
            {"segment_a": a, "segment_b": b, "count": cnt}
            for (a, b), cnt in pair_counts.most_common()
        ],
        "top_hubs_by_mentions": [
            {"segment_id": sid, "mentions": cnt}
            for sid, cnt in top_hubs_by_mentions
        ],
        "top_hubs_by_unique_partners": top_hubs_by_partners,
        "repeated_bidirectional_labels": repeated_bidirectional_labels,
        "transition_matrix": transition_matrix,
    }


def parse_state(state_path: Path) -> dict[str, Any]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    layers = state.get("layers", [])

    by_type = Counter(layer.get("type", "unknown") for layer in layers)
    layer_summaries = []
    segmentation_summary = None
    annotation_summary = None

    for layer in layers:
        layer_summaries.append(
            {
                "name": layer.get("name"),
                "type": layer.get("type"),
                "keys": sorted(layer.keys()),
            }
        )

        if layer.get("type") == "segmentation" and segmentation_summary is None:
            segments = [str(x) for x in layer.get("segments", [])]
            negated = sum(1 for s in segments if s.startswith("!"))
            segmentation_summary = {
                "layer_name": layer.get("name"),
                "segments_total": len(segments),
                "segments_included": len(segments) - negated,
                "segments_excluded": negated,
                "segment_query": layer.get("segmentQuery", ""),
            }

        if layer.get("type") == "annotation" and annotation_summary is None:
            annotation_summary = {
                "layer_name": layer.get("name"),
                "annotations_count": len(layer.get("annotations", [])),
                "linked_segmentation_layer": layer.get("linkedSegmentationLayer"),
            }

    return {
        "layer_count": len(layers),
        "layers_by_type": dict(by_type),
        "layers": layer_summaries,
        "segmentation": segmentation_summary,
        "annotation": annotation_summary,
    }


def render_markdown(summary: dict[str, Any], top_n: int) -> str:
    ann = summary["annotations"]["annotation_stats"]
    state = summary["state"]
    top_desc = summary["annotations"]["top_descriptions"][:top_n]
    top_pairs = summary["annotations"]["top_pairs"][:top_n]
    top_hubs = summary["annotations"]["top_hubs_by_mentions"][:top_n]
    top_transitions = summary["annotations"]["top_transitions"][:top_n]

    lines = []
    lines.append("# Local Inference Report")
    lines.append("")
    lines.append("## Annotation Overview")
    lines.append(f"- Rows total: {ann['rows_total']}")
    lines.append(f"- Rows with description: {ann['rows_with_description']}")
    lines.append(f"- Rows with two segments: {ann['rows_with_two_segments']}")
    lines.append(f"- Rows with single segment: {ann['rows_with_single_segment']}")
    lines.append(f"- Rows with no segment: {ann['rows_with_no_segment']}")
    lines.append(f"- Unique descriptions: {ann['unique_descriptions']}")
    lines.append(f"- Unique segments: {ann['unique_segments']}")
    lines.append("")
    lines.append("## State Overview")
    lines.append(f"- Layer count: {state['layer_count']}")
    lines.append(f"- Layers by type: {state['layers_by_type']}")
    seg = state.get("segmentation") or {}
    if seg:
        lines.append(
            f"- Segmentation layer '{seg['layer_name']}': "
            f"{seg['segments_included']} included, "
            f"{seg['segments_excluded']} excluded, "
            f"{seg['segments_total']} total"
        )
    ann_layer = state.get("annotation") or {}
    if ann_layer:
        lines.append(
            f"- Annotation layer '{ann_layer['layer_name']}': "
            f"{ann_layer['annotations_count']} annotations"
        )
    lines.append("")
    lines.append("## Top Descriptions")
    for item in top_desc:
        lines.append(f"- {item['description']}: {item['count']}")
    lines.append("")
    lines.append("## Top Transitions")
    for item in top_transitions:
        lines.append(f"- {item['transition']}: {item['count']}")
    lines.append("")
    lines.append("## Top Segment Hubs (By Mentions)")
    for item in top_hubs:
        lines.append(f"- {item['segment_id']}: {item['mentions']}")
    lines.append("")
    lines.append("## Top Repeated Segment Pairs")
    for item in top_pairs:
        lines.append(
            f"- ({item['segment_a']}, {item['segment_b']}): {item['count']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Infer summary patterns from local annotations CSV and state JSON."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/Users/supriyanagnathkadam/Downloads/annotations_flat.csv"),
        help="Path to annotations_flat.csv",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("/Users/supriyanagnathkadam/Downloads/state.json"),
        help="Path to Neuroglancer state.json",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="How many rows to keep in top lists",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write full JSON summary",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional path to write Markdown report",
    )
    parser.add_argument(
        "--out-transition-matrix-csv",
        type=Path,
        default=None,
        help="Optional path to write source->target class interaction matrix CSV",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}")
        return 2
    if not args.state.exists():
        print(f"ERROR: state JSON not found: {args.state}")
        return 2
    if args.top_n <= 0:
        print("ERROR: --top-n must be > 0")
        return 2

    summary = {
        "inputs": {"csv": str(args.csv), "state": str(args.state)},
        "annotations": parse_csv(args.csv),
        "state": parse_state(args.state),
    }

    md = render_markdown(summary, top_n=args.top_n)
    print(md)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to: {args.out_json}")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
        print(f"Wrote Markdown report to: {args.out_md}")
    if args.out_transition_matrix_csv:
        write_transition_matrix_csv(
            summary["annotations"]["transition_matrix"],
            args.out_transition_matrix_csv,
        )
        print(f"Wrote transition matrix CSV to: {args.out_transition_matrix_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
