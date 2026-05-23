"""
Analyze the per-class distribution of the YOLO staging conversions.

Walks both ``open_images`` and ``uec_food`` staging dirs, counts bboxes per
target YOLO class (cross-dataset), computes share + size statistics
per class, and writes a JSON report named ``class_distribution_<timestamp>.json`` 
under ``reports/``.

Run with::
    python -m src.data.validation.analyze_class_distribution
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

from src.data.common.convert_to_yolo import load_classes_config
from src.data.common.dataset_io import (
    list_staging_images,
    parse_yolo_label_file,
)
from src.utils.paths import (
    get_open_images_staging_dir,
    get_reports_dir,
    get_uec_staging_dir,
)


# ---------------------------------------------------------------------------
# Walking the staging dirs
# ---------------------------------------------------------------------------
def _label_path_for(staging_dir: Path, image_path: Path) -> Path:
    return staging_dir / "labels" / f"{image_path.stem}.txt"


def collect_class_stats(
    staging_dirs: list[tuple[str, Path]],
) -> tuple[dict[int, int], dict[int, list[float]], dict[int, list[float]], int]:
    """Walk every staging dir and aggregate per-class counts and bbox sizes.

    Returns ``(boxes_per_class, widths_per_class, heights_per_class,
    invalid_lines)``. ``invalid_lines`` counts lines silently dropped by
    ``parse_yolo_label_file`` so the report can flag malformed staging.
    """
    boxes_per_class: dict[int, int] = {}
    widths_per_class: dict[int, list[float]] = {}
    heights_per_class: dict[int, list[float]] = {}
    invalid_lines = 0

    for dataset, staging_dir in staging_dirs:
        if not staging_dir.exists():
            print(f"  WARNING: staging dir for '{dataset}' not found: {staging_dir}")
            continue

        for image_path in list_staging_images(staging_dir):
            label_path = _label_path_for(staging_dir, image_path)
            if not label_path.exists():
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                raw_lines = [line for line in f if line.strip()]

            parsed = parse_yolo_label_file(label_path)
            invalid_lines += len(raw_lines) - len(parsed)

            for class_id, bbox in parsed:
                _, _, w, h = bbox
                boxes_per_class[class_id] = boxes_per_class.get(class_id, 0) + 1
                widths_per_class.setdefault(class_id, []).append(w)
                heights_per_class.setdefault(class_id, []).append(h)

    return boxes_per_class, widths_per_class, heights_per_class, invalid_lines


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _bbox_size_stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": float(min(values)),
        "p50": float(statistics.median(values)),
        "max": float(max(values)),
    }


def build_report(
    classes: list[dict],
    staging_dirs: list[tuple[str, Path]],
    boxes_per_class: dict[int, int],
    widths_per_class: dict[int, list[float]],
    heights_per_class: dict[int, list[float]],
    invalid_lines: int,
) -> dict:
    """Build the class-distribution report dict ready to serialize as JSON."""
    boxes_total = sum(boxes_per_class.values())
    classes_present_ids: set[int] = set()
    classes_missing: list[str] = []
    per_class: dict[str, dict] = {}

    for entry in sorted(classes, key=lambda c: int(c["id"])):
        class_id = int(entry["id"])
        class_name = entry["name"]
        boxes = boxes_per_class.get(class_id, 0)

        if boxes > 0:
            classes_present_ids.add(class_id)
        else:
            classes_missing.append(class_name)

        share = (boxes / boxes_total) if boxes_total > 0 else 0.0

        widths = widths_per_class.get(class_id, [])
        heights = heights_per_class.get(class_id, [])
        w_stats = _bbox_size_stats(widths)
        h_stats = _bbox_size_stats(heights)

        per_class[class_name] = {
            "id": class_id,
            "boxes": boxes,
            "share": round(share, 4),
            "bbox_size_stats": (
                {
                    "w_min": w_stats["min"] if w_stats else None,
                    "w_p50": w_stats["p50"] if w_stats else None,
                    "w_max": w_stats["max"] if w_stats else None,
                    "h_min": h_stats["min"] if h_stats else None,
                    "h_p50": h_stats["p50"] if h_stats else None,
                    "h_max": h_stats["max"] if h_stats else None,
                }
                if w_stats and h_stats
                else None
            ),
        }

    imbalance = _build_imbalance_block(per_class, boxes_total)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "staging_dirs": {dataset: str(path) for dataset, path in staging_dirs},
        "per_class": per_class,
        "totals": {
            "boxes_total": boxes_total,
            "classes_present": len(classes_present_ids),
            "classes_missing": classes_missing,
        },
        "imbalance": imbalance,
        "invalid_lines": invalid_lines,
    }


def _build_imbalance_block(per_class: dict[str, dict], boxes_total: int) -> dict:
    populated = {name: data for name, data in per_class.items() if data["boxes"] > 0}
    if not populated or boxes_total == 0:
        return {
            "max_class": None,
            "min_class": None,
            "max_min_ratio": None,
            "dominant_class_share": None,
        }

    max_name, max_data = max(populated.items(), key=lambda kv: kv[1]["boxes"])
    min_name, min_data = min(populated.items(), key=lambda kv: kv[1]["boxes"])

    return {
        "max_class": max_name,
        "min_class": min_name,
        "max_min_ratio": round(max_data["boxes"] / min_data["boxes"], 2),
        "dominant_class_share": round(max_data["boxes"] / boxes_total, 4),
    }


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
def _format_stat(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "  -  "


def print_report(report: dict, classes: list[dict]) -> None:
    print("\nBoxes per class")
    print("-" * 97)
    header = (
        f"{'class':<14}{'id':>4}{'boxes':>10}{'share':>10}    "
        f"{'w_min':>7}{'w_p50':>8}{'w_max':>8}    "
        f"{'h_min':>7}{'h_p50':>8}{'h_max':>8}"
    )
    print(header)

    for entry in sorted(classes, key=lambda c: int(c["id"])):
        class_name = entry["name"]
        data = report["per_class"][class_name]
        size = data["bbox_size_stats"] or {}
        share_pct = f"{data['share'] * 100:.2f}%"
        print(
            f"{class_name:<14}{data['id']:>4}{data['boxes']:>10}{share_pct:>10}    "
            f"{_format_stat(size.get('w_min')):>7}{_format_stat(size.get('w_p50')):>8}"
            f"{_format_stat(size.get('w_max')):>8}    "
            f"{_format_stat(size.get('h_min')):>7}{_format_stat(size.get('h_p50')):>8}"
            f"{_format_stat(size.get('h_max')):>8}"
        )

    totals = report["totals"]
    classes_total = len(classes)
    print("\nTotals")
    print(f"  boxes_total:           {totals['boxes_total']}")
    print(f"  classes_present:       {totals['classes_present']} / {classes_total}")
    print(f"  classes_missing:       {totals['classes_missing']}")

    imbalance = report["imbalance"]
    print("\nImbalance")
    if imbalance["max_class"] is None:
        print("  (no boxes found)")
    else:
        max_boxes = report["per_class"][imbalance["max_class"]]["boxes"]
        min_boxes = report["per_class"][imbalance["min_class"]]["boxes"]
        share_pct = f"{imbalance['dominant_class_share'] * 100:.2f}%"
        print(f"  max_class:             {imbalance['max_class']} ({max_boxes} boxes)")
        print(f"  min_class:             {imbalance['min_class']} ({min_boxes} boxes)")
        print(f"  max_min_ratio:         {imbalance['max_min_ratio']}")
        print(f"  dominant_class_share:  {share_pct}")

    if report["invalid_lines"]:
        print(f"\nNote: skipped {report['invalid_lines']} malformed label line(s).")


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
def _timestamped_report_path(now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return get_reports_dir() / f"class_distribution_{timestamp}.json"


def write_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("YOLO staging class distribution")
    print("=" * 60)

    classes = load_classes_config()
    staging_dirs: list[tuple[str, Path]] = [
        ("open_images", get_open_images_staging_dir()),
        ("uec_food", get_uec_staging_dir()),
    ]

    print("staging dirs:")
    for dataset, path in staging_dirs:
        print(f"  {dataset:<13} {path}")

    boxes_per_class, widths_per_class, heights_per_class, invalid_lines = (
        collect_class_stats(staging_dirs)
    )

    if sum(boxes_per_class.values()) == 0:
        print(
            "\nNo bboxes found in staging. Run the per-source converters first:\n"
            "  python -m src.data.conversion.convert_uec_food_to_yolo\n"
            "  python -m src.data.conversion.convert_open_images_to_yolo"
        )
        return

    report = build_report(
        classes=classes,
        staging_dirs=staging_dirs,
        boxes_per_class=boxes_per_class,
        widths_per_class=widths_per_class,
        heights_per_class=heights_per_class,
        invalid_lines=invalid_lines,
    )
    print_report(report, classes)

    output_path = _timestamped_report_path()
    write_report(report, output_path)
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
