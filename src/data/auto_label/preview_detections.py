"""
Phase-5 read-only preview for notebook 0.5: run a pretrained COCO-80 YOLO over a
few raw Open Images and show what the auto-labeling pass WOULD inject — without
writing anything to the COCO export.

This is the eyeball-the-quality step before any enrichment. For each sampled
image it prints, per detection, whether the detector's class name maps to one of
our 7 target classes (and to which) or is dropped, and renders the detector
boxes to PNGs under ``outputs/auto_label_checks/phase5_detections/``. It then
prints a summary: total detections, kept counts per target class, and dropped
names with counts. Use it to pick a sensible ``--conf`` threshold before running
the (separate) write pass.

The detector class name -> target id mapping comes from the ``coco_pretrained``
section of ``configs/label_mapping.yaml`` (see
:mod:`src.data.auto_label.coco_target_mapping`). Detections whose name is not in
that section are DROPPED.

Usage (local, against the already-extracted raw dir)::

    python -m src.data.auto_label.preview_detections --samples 5 --conf 0.4

Downloaded YOLO weights are kept under ``models/`` (gitignored), never the repo
root.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from src.data.auto_label.coco_io import category_id_to_name, load_coco
from src.data.auto_label.coco_target_mapping import (
    load_coco_pretrained_mapping,
    map_detection_name,
)
from src.data.auto_label.prepare_open_images_input import find_labels_json
from src.data.raw_setup.visualize_raw_bboxes import show_image_with_boxes
from src.utils.labels import load_class_names
from src.utils.paths import (
    get_model_path,
    get_models_dir,
    get_open_images_dataset_original_dir,
    get_outputs_dir,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHASE5_OUTPUT_DIR = get_outputs_dir() / "auto_label_checks" / "phase5_detections"


# ---------------------------------------------------------------------------
# Weights resolution
# ---------------------------------------------------------------------------
def resolve_weights(weights: str) -> Path:
    """
    Resolve a YOLO weights spec to a path under ``models/``.

    A bare name like ``"yolov8x.pt"`` (no directory component) is resolved to
    ``get_model_path(weights)``. Passing that full path to ``ultralytics.YOLO``
    downloads the asset directly into ``models/`` (verified empirically), so the
    repo root never receives a stray ``.pt``. An explicit path (containing a
    directory) is returned unchanged so callers can point at custom weights.

    Args:
        weights: Either a bare ultralytics asset name (``"yolov8x.pt"``) or an
            explicit filesystem path to a weights file.

    Returns:
        The path to hand to ``YOLO(...)``.
    """
    weights_path = Path(weights)
    if weights_path.parent == Path("."):
        get_models_dir().mkdir(parents=True, exist_ok=True)
        return get_model_path(weights_path.name)
    return weights_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for the phase-5 detection preview."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a pretrained COCO YOLO over a few raw Open Images and preview "
            "(read-only) how each detection would map to our 7 target classes."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of sample images to run the detector on (>= 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Detector confidence threshold.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8x.pt",
        help="Pretrained YOLO weights name or path (kept under models/).",
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
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    return args


def main(argv=None) -> None:
    """Run the detector on sampled raw images and print the would-be mapping."""
    # Import here so the module stays importable without torch/ultralytics.
    from ultralytics import YOLO

    args = parse_args(argv)

    mapping = load_coco_pretrained_mapping()
    target_names = load_class_names()

    coco_dir = get_open_images_dataset_original_dir()
    json_path = find_labels_json(coco_dir)
    coco = load_coco(json_path)
    # category_id_to_name is loaded for parity with the rest of the pass even
    # though the preview labels come from the detector, not the COCO export.
    _ = category_id_to_name(coco)

    images = coco.get("images", [])
    rng = random.Random(args.seed)
    n = min(args.samples, len(images))
    chosen = rng.sample(images, n) if n > 0 else []

    weights_path = resolve_weights(args.weights)
    print(f"Loading weights: {weights_path}")
    model = YOLO(str(weights_path))

    PHASE5_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_detections = 0
    kept_per_target: Counter[int] = Counter()
    dropped_names: Counter[str] = Counter()
    written: list[Path] = []

    print("\n" + "=" * 60)
    print(f"Phase-5 detection preview — {len(chosen)} image(s), conf={args.conf}")
    print("=" * 60)

    for image in chosen:
        file_name = image["file_name"]
        image_path = coco_dir / "data" / file_name

        print(f"\n{file_name}")
        if not image_path.exists():
            print(f"  WARN: image file missing, skipping: {image_path}")
            continue

        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        result = results[0]

        boxes: list[list[float]] = []
        labels: list[str] = []

        if result.boxes is None or len(result.boxes) == 0:
            print("  (no detections)")

        for xyxy, conf, cls in zip(
            result.boxes.xyxy.tolist() if result.boxes is not None else [],
            result.boxes.conf.tolist() if result.boxes is not None else [],
            result.boxes.cls.tolist() if result.boxes is not None else [],
        ):
            total_detections += 1
            name = model.names[int(cls)]
            target_id = map_detection_name(name, mapping)

            if target_id is None:
                dropped_names[name] += 1
                print(f"  {name} (conf={conf:.2f}) -> DROPPED (unmapped)")
                box_target_label = "DROP"
            else:
                kept_per_target[target_id] += 1
                target_name = target_names[target_id]
                print(f"  {name} (conf={conf:.2f}) -> {target_id} {target_name}")
                box_target_label = target_name

            boxes.append(xyxy)
            labels.append(f"{name} {conf:.2f} -> {box_target_label}")

        stem = Path(file_name).stem
        output_path = PHASE5_OUTPUT_DIR / f"{stem}.png"
        title = f"{file_name} — {len(boxes)} detections (pretrained YOLO, conf={args.conf})"
        show_image_with_boxes(image_path, boxes, labels, output_path, title=title)
        written.append(output_path)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total detections: {total_detections}")

    print("Kept per target class:")
    if kept_per_target:
        for target_id in sorted(kept_per_target):
            name = target_names.get(target_id, f"id={target_id}")
            print(f"  {target_id} {name}: {kept_per_target[target_id]}")
    else:
        print("  (none)")

    print("Dropped names:")
    if dropped_names:
        for name, count in sorted(dropped_names.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {name}: {count}")
    else:
        print("  (none)")

    print("\nWritten PNGs:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
