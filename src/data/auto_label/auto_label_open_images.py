"""
Phase-6 enrichment pass for notebook 0.5: run a pretrained COCO-80 YOLO over the
raw Open Images COCO export (v1) and write an enriched COPY (v2) with the
detector's boxes injected as new annotations.

The raw v1 export
(``datasets/raw_datasets/open_images_subset/labels.json``) is the read-only
source data and is NEVER modified by this script: the enriched output goes to a
separate ``open_images_subset_v2/labels.json`` and the run refuses to start if
the resolved output path collides with the source path. The original image
files stay where they are; only a new ``labels.json`` is produced (v2 reuses the
v1 ``data/`` images by their original ``file_name``). Notebook 01 later consumes
v2 instead of v1 to gain the auto-labeled ``Food`` (and reinforced cutlery)
boxes that the 10-class v1 export lacks.

For each selected image the pass:

- gathers the v1 boxes that already map to one of our 7 target classes (so a
  detection on top of an existing box is not injected twice);
- runs the detector, processes detections by confidence descending, maps each
  detector name to a target id (dropping unmapped names), clips to the image,
  drops degenerate boxes, and drops same-class duplicates via IoU;
- injects each surviving box as a new COCO annotation carrying the
  downstream-recognized category name (e.g. ``"Food"``) plus a ``score`` and a
  ``source="auto_label"`` tag.

Detection runs in chunks of ``--chunk-size`` images per ``model.predict()`` call.
Ultralytics stacks a list source into ONE batch tensor, so the chunk size is the
effective inference batch size: passing the full ~11k path list makes it decode
every image up front and try to allocate a ~50 GiB tensor (RuntimeError, and the
progress bar appears hung at 0 until then). Small chunks bound memory and start
emitting results — and advancing the bar — almost immediately.

It always writes a JSON report (counts only) and prints a human summary, so a
``--dry-run`` answers "how many boxes would this add and where" without writing
the v2 document.

Usage (local, against the already-extracted raw dir)::

    python -m src.data.auto_label.auto_label_open_images --limit 50 --dry-run
    python -m src.data.auto_label.auto_label_open_images --limit 50
    python -m src.data.auto_label.auto_label_open_images            # all images

Downloaded YOLO weights are kept under ``models/`` (gitignored), never the repo
root.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.data.auto_label.coco_io import (
    add_annotation,
    ensure_category,
    load_coco,
    next_annotation_id,
    save_coco,
)
from src.data.auto_label.coco_target_mapping import (
    TARGET_TO_COCO_CATEGORY,
    is_duplicate,
    load_coco_pretrained_mapping,
    map_detection_name,
    xywh_to_xyxy,
)
from src.data.auto_label.prepare_open_images_input import find_labels_json
from src.data.auto_label.preview_detections import resolve_weights
from src.data.common.convert_to_yolo import load_label_mapping
from src.utils.labels import load_class_names
from src.utils.paths import (
    get_open_images_dataset_original_dir,
    get_raw_datasets_dir,
    get_reports_dir,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def default_out_path() -> Path:
    """Default v2 labels path: a sibling ``open_images_subset_v2`` dir."""
    return get_raw_datasets_dir() / "open_images_subset_v2" / "labels.json"


def default_report_path() -> Path:
    """Default report path under ``reports/``."""
    return get_reports_dir() / "auto_label_report.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for the phase-6 enrichment pass."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.auto_label.auto_label_open_images",
        description=(
            "Enrich the raw Open Images COCO export (v1) with pretrained-detector "
            "boxes and write an enriched COPY (v2). v1 is never modified."
        ),
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=get_open_images_dataset_original_dir(),
        help="Root of the extracted v1 Open Images COCO export (labels.json + data/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out_path(),
        help="Destination for the enriched v2 labels.json (must differ from v1).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8x.pt",
        help="Pretrained YOLO weights name or path (kept under models/).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Detector confidence threshold.",
    )
    parser.add_argument(
        "--iou-dedup",
        type=float,
        default=0.5,
        help="IoU at/above which a same-class detection duplicates an existing box.",
    )
    parser.add_argument(
        "--predict-iou",
        type=float,
        default=0.5,
        help="IoU passed to the detector's own NMS during prediction.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Inference device (empty string = ultralytics auto-select).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only a random sample of this many images (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the sample selection (only used with --limit).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help=(
            "Number of images per model.predict() call. Ultralytics stacks a list "
            "source into ONE batch tensor, so the chunk size IS the inference batch "
            "size: the full 11k list would need a ~50 GiB tensor (hard crash), and "
            "large chunks can OOM smaller GPUs (e.g. a T4). 32 is safe and fast. "
            "Must be >= 1."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report counts only; do not write the v2 labels.json.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=default_report_path(),
        help="Where to write the JSON run report.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def select_images(images: list[dict], limit: int | None, seed: int) -> list[dict]:
    """Return all images, or a seeded random sample of ``limit`` of them."""
    if limit is None or limit >= len(images):
        return list(images)
    rng = random.Random(seed)
    return rng.sample(images, limit)


def build_existing_target_boxes(
    annotations_by_image: dict[int, list[dict]],
    image_id: int,
    cat_id_to_name: dict[int, str],
    open_images_mapping: dict[str, int],
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """
    Collect the v1 boxes for one image that already map to a target class.

    Each v1 annotation's category NAME is mapped to a target id via the
    ``open_images`` section of ``label_mapping.yaml``; names that do not map
    (which should not occur in v1) are skipped defensively. The returned list
    seeds the per-image de-duplication so detections on top of existing boxes
    are not injected twice.
    """
    existing: list[tuple[int, tuple[float, float, float, float]]] = []
    for ann in annotations_by_image.get(image_id, []):
        name = cat_id_to_name.get(int(ann["category_id"]))
        if name is None:
            continue
        target_id = open_images_mapping.get(name)
        if target_id is None:
            continue
        existing.append((int(target_id), xywh_to_xyxy(ann["bbox"])))
    return existing


def class_key(target_id: int, target_names: dict[int, str]) -> str:
    """Render a ``"<id> <name>"`` key for the per-class report counters."""
    return f"{target_id} {target_names.get(target_id, f'id={target_id}')}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    """Run the enrichment pass and write the v2 doc (unless dry-run) + report."""
    # Import here so the module stays importable without torch/ultralytics.
    from ultralytics import YOLO
    from tqdm import tqdm

    args = parse_args(argv)

    if args.chunk_size < 1:
        raise ValueError(f"--chunk-size must be >= 1, got {args.chunk_size}.")

    src_dir: Path = args.src_dir
    json_path = find_labels_json(src_dir)
    out_path: Path = args.out

    # DEFENSIVE: never overwrite the read-only v1 source.
    if out_path.resolve() == json_path.resolve():
        raise ValueError(
            f"Refusing to run: --out ({out_path}) resolves to the v1 source "
            f"labels.json ({json_path}). The v1 export is read-only; point "
            "--out at a separate path (default: open_images_subset_v2/labels.json)."
        )

    coco = load_coco(json_path)
    target_names = load_class_names()

    open_images_mapping_raw = (load_label_mapping() or {}).get("open_images", {})
    open_images_mapping = {str(name): int(tid) for name, tid in open_images_mapping_raw.items()}

    detector_mapping = load_coco_pretrained_mapping()

    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in coco.get("categories", [])}
    annotations_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    images = coco.get("images", [])
    selected = select_images(images, args.limit, args.seed)

    data_dir = src_dir / "data"

    # Filter out images whose file is missing BEFORE predicting so the detector
    # results stay aligned with the entries we iterate over.
    present: list[dict] = []
    missing_image_files = 0
    for image in selected:
        if (data_dir / image["file_name"]).exists():
            present.append(image)
        else:
            missing_image_files += 1
            print(f"  WARN: image file missing, skipping: {data_dir / image['file_name']}")

    weights_path = resolve_weights(args.weights)
    print(f"Loading weights: {weights_path}")
    model = YOLO(str(weights_path))

    image_paths = [str(data_dir / image["file_name"]) for image in present]

    # Per-image accepted boxes: image_id -> [(category_name, xyxy, conf), ...].
    accepted_by_image: dict[int, list[tuple[str, tuple[float, float, float, float], float]]] = {}

    detections_total = 0
    added_total = 0
    added_per_class: Counter[str] = Counter()
    skipped_duplicates_per_class: Counter[str] = Counter()
    dropped_unmapped: Counter[str] = Counter()
    skipped_degenerate = 0
    images_with_additions = 0

    def process_image(image: dict, result) -> None:
        """Process one image's detections, updating the shared counters/accumulators."""
        nonlocal detections_total, added_total, skipped_degenerate, images_with_additions
        image_id = int(image["id"])
        width = int(image["width"])
        height = int(image["height"])

        existing = build_existing_target_boxes(
            annotations_by_image, image_id, cat_id_to_name, open_images_mapping
        )

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return

        detections = sorted(
            zip(
                boxes.xyxy.tolist(),
                boxes.conf.tolist(),
                boxes.cls.tolist(),
            ),
            key=lambda d: d[1],
            reverse=True,
        )

        added_this_image: list[tuple[str, tuple[float, float, float, float], float]] = []
        for xyxy, conf, cls in detections:
            detections_total += 1
            name = model.names[int(cls)]
            target_id = map_detection_name(name, detector_mapping)
            if target_id is None:
                dropped_unmapped[name] += 1
                continue

            x1 = min(max(xyxy[0], 0.0), float(width))
            y1 = min(max(xyxy[1], 0.0), float(height))
            x2 = min(max(xyxy[2], 0.0), float(width))
            y2 = min(max(xyxy[3], 0.0), float(height))

            if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
                skipped_degenerate += 1
                continue

            clipped = (x1, y1, x2, y2)
            key = class_key(target_id, target_names)
            if is_duplicate(clipped, target_id, existing, args.iou_dedup):
                skipped_duplicates_per_class[key] += 1
                continue

            existing.append((target_id, clipped))
            category_name = TARGET_TO_COCO_CATEGORY[target_id]
            added_this_image.append((category_name, clipped, float(conf)))
            added_per_class[key] += 1
            added_total += 1

        if added_this_image:
            images_with_additions += 1
            accepted_by_image[image_id] = added_this_image

    # Predict in chunks: one large single predict() call over the full path list
    # blocks for minutes on internal setup before yielding its first result, so
    # we slice the work into --chunk-size batches that start producing results
    # (and advancing the bar) almost immediately.
    with tqdm(total=len(present), desc="Enriching", unit="img") as progress:
        for start in range(0, len(present), args.chunk_size):
            chunk_images = present[start : start + args.chunk_size]
            chunk_paths = image_paths[start : start + args.chunk_size]
            results = model.predict(
                chunk_paths,
                stream=True,
                conf=args.conf,
                iou=args.predict_iou,
                imgsz=args.imgsz,
                device=args.device or None,
                verbose=False,
            )
            for image, result in zip(chunk_images, results):
                process_image(image, result)
                progress.update(1)

    out_labels: str | None = None
    if not args.dry_run:
        enriched = copy.deepcopy(coco)
        ann_id = next_annotation_id(enriched)
        for image_id, additions in accepted_by_image.items():
            for category_name, (x1, y1, x2, y2), conf in additions:
                category_id = ensure_category(enriched, category_name)
                add_annotation(
                    enriched,
                    image_id=image_id,
                    category_id=category_id,
                    bbox_xywh=[x1, y1, x2 - x1, y2 - y1],
                    ann_id=ann_id,
                    extra={"score": round(conf, 4), "source": "auto_label"},
                )
                ann_id += 1
        save_coco(enriched, out_path)
        out_labels = str(out_path)

    report = {
        "params": {
            "src_dir": str(src_dir),
            "out": str(out_path),
            "weights": args.weights,
            "conf": args.conf,
            "iou_dedup": args.iou_dedup,
            "predict_iou": args.predict_iou,
            "imgsz": args.imgsz,
            "device": args.device,
            "limit": args.limit,
            "seed": args.seed,
            "chunk_size": args.chunk_size,
            "dry_run": args.dry_run,
            "report": str(args.report),
        },
        "src_labels": str(json_path),
        "out_labels": out_labels,
        "images_processed": len(present),
        "images_with_additions": images_with_additions,
        "detections_total": detections_total,
        "added_total": added_total,
        "added_per_class": dict(added_per_class),
        "skipped_duplicates_per_class": dict(skipped_duplicates_per_class),
        "dropped_unmapped": dict(dropped_unmapped),
        "skipped_degenerate": skipped_degenerate,
        "missing_image_files": missing_image_files,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path: Path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    _print_summary(report)


def _print_summary(report: dict) -> None:
    """Print a human-readable summary mirroring the JSON report."""
    print("\n" + "=" * 60)
    print("Phase-6 auto-label enrichment summary")
    print("=" * 60)
    if report["params"]["dry_run"]:
        print("DRY-RUN: no v2 labels.json written (counts only).")
    else:
        print(f"v2 labels written to: {report['out_labels']}")
    print(f"v1 source (untouched): {report['src_labels']}")
    print(f"Images processed:      {report['images_processed']}")
    print(f"Images with additions: {report['images_with_additions']}")
    print(f"Missing image files:   {report['missing_image_files']}")
    print(f"Detections total:      {report['detections_total']}")
    print(f"Added total:           {report['added_total']}")
    print(f"Skipped degenerate:    {report['skipped_degenerate']}")

    print("\nAdded per class:")
    if report["added_per_class"]:
        for key in sorted(report["added_per_class"]):
            print(f"  {key}: {report['added_per_class'][key]}")
    else:
        print("  (none)")

    print("Skipped duplicates per class:")
    if report["skipped_duplicates_per_class"]:
        for key in sorted(report["skipped_duplicates_per_class"]):
            print(f"  {key}: {report['skipped_duplicates_per_class'][key]}")
    else:
        print("  (none)")

    print("Dropped unmapped detector names:")
    if report["dropped_unmapped"]:
        for name, count in sorted(
            report["dropped_unmapped"].items(), key=lambda kv: kv[1], reverse=True
        ):
            print(f"  {name}: {count}")
    else:
        print("  (none)")

    print(f"\nReport written to: {report['params']['report']}")


if __name__ == "__main__":
    main()
