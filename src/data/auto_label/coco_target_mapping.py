"""
Pure-logic mapping between a pretrained COCO-80 detector and our 7 target
classes, used by the notebook-0.5 auto-labeling pass.

This module is deliberately importable WITHOUT torch/ultralytics: it only reads
``configs/label_mapping.yaml`` and turns detector class names into target YOLO
ids. The detector side lives in :mod:`src.data.auto_label.preview_detections`,
which imports these helpers and supplies the runtime names from ``model.names``.

Two distinct concepts live here:

- ``load_coco_pretrained_mapping`` reads the ``coco_pretrained`` section of the
  label-mapping config: ``{detector_name (lowercase): target_id}``. It is how a
  detection like ``"wine glass"`` becomes target id 1 (cup). Names absent from
  the config are dropped by :func:`map_detection_name`.
- ``TARGET_TO_COCO_CATEGORY`` is the reverse-facing decision: for each target
  id, which COCO-export *category name* an injected annotation should carry so
  that notebook 01's name-based YOLO converter maps it back to the same target.
  All of these names except ``"Food"`` already exist in the v1 Open Images
  export; ``"Food"`` is the category added by the enrichment pass.
"""

from __future__ import annotations

from pathlib import Path

from src.data.common.convert_to_yolo import load_label_mapping

# Representative COCO-export category name written for each target id when
# injecting annotations. All except "Food" already exist in the v1 export;
# "Food" is created by the enrichment pass (notebook 0.5).
TARGET_TO_COCO_CATEGORY: dict[int, str] = {
    0: "Food",
    1: "Coffee cup",
    2: "Bottle",
    3: "Bowl",
    4: "Spoon",
    5: "Fork",
    6: "Knife",
}


def load_coco_pretrained_mapping(
    config_path: str | Path = "configs/label_mapping.yaml",
) -> dict[str, int]:
    """
    Load the ``coco_pretrained`` detector-name -> target-id mapping.

    Reuses :func:`load_label_mapping` to read the full YAML doc, then takes the
    ``coco_pretrained`` section. Keys are lowercased (to match ultralytics
    ``model.names`` values), values coerced to ``int``. Every target id must
    have an entry in :data:`TARGET_TO_COCO_CATEGORY`, so an enriched annotation
    can later be written with a downstream-recognized category name.

    Args:
        config_path: Path to ``label_mapping.yaml``.

    Returns:
        ``{detector_name_lowercase: target_id}``.

    Raises:
        ValueError: If the ``coco_pretrained`` section is missing/empty, or if
            any mapped target id is not present in
            :data:`TARGET_TO_COCO_CATEGORY`.
    """
    doc = load_label_mapping(config_path)
    section = (doc or {}).get("coco_pretrained")
    if not section:
        raise ValueError(
            "Missing or empty 'coco_pretrained' section in "
            f"{config_path}. Add the pretrained COCO-80 class name -> target "
            "YOLO id mapping used by notebook 0.5 auto-labeling."
        )

    mapping: dict[str, int] = {}
    for name, target_id in section.items():
        target_id = int(target_id)
        if target_id not in TARGET_TO_COCO_CATEGORY:
            raise ValueError(
                f"coco_pretrained maps '{name}' to target id {target_id}, which "
                "has no entry in TARGET_TO_COCO_CATEGORY. Valid target ids are "
                f"{sorted(TARGET_TO_COCO_CATEGORY)}."
            )
        mapping[str(name).lower()] = target_id

    return mapping


def map_detection_name(name: str, mapping: dict[str, int]) -> int | None:
    """
    Map a detector class name to a target YOLO id, or ``None`` to drop it.

    Lookup is case-insensitive (the name is lowercased before lookup, matching
    the keys produced by :func:`load_coco_pretrained_mapping`). A ``None`` result
    means the detection is not one of our target classes and should be dropped.

    Args:
        name: Detector class name, e.g. ``model.names[int(cls)]``.
        mapping: A ``{name_lowercase: target_id}`` mapping.

    Returns:
        The target YOLO id, or ``None`` if the name is not mapped.
    """
    return mapping.get(name.lower())


# ---------------------------------------------------------------------------
# Geometry helpers (pure, no torch) used by the enrichment de-duplication pass
# ---------------------------------------------------------------------------
def xywh_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    """
    Convert a COCO ``[x, y, w, h]`` box to ``(x1, y1, x2, y2)`` corners.

    COCO stores the top-left corner plus width and height; the de-duplication
    logic and clipping work in corner coordinates, so v1 annotations are
    converted once via this helper before being compared against detections.

    Args:
        bbox: COCO pixel box ``[x_min, y_min, width, height]``.

    Returns:
        ``(x1, y1, x2, y2)`` where ``x2 = x + w`` and ``y2 = y + h``.
    """
    x, y, w, h = bbox
    return x, y, x + w, y + h


def iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """
    Intersection-over-union of two axis-aligned boxes in xyxy corners.

    Standard IoU: area of overlap divided by area of union. Returns ``0.0``
    when the boxes do not overlap and, defensively, when the union area is
    zero (degenerate boxes) to avoid a division by zero.

    Args:
        box_a: First box as ``(x1, y1, x2, y2)``.
        box_b: Second box as ``(x1, y1, x2, y2)``.

    Returns:
        IoU in ``[0.0, 1.0]``.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


def is_duplicate(
    det_xyxy: tuple[float, float, float, float],
    det_target_id: int,
    existing: list[tuple[int, tuple[float, float, float, float]]],
    iou_threshold: float,
) -> bool:
    """
    Decide whether a detection duplicates an already-present box of the same class.

    A detection is a duplicate when some ``existing`` box shares its target id
    AND overlaps it with ``iou >= iou_threshold``. Boxes of a different target
    id never count as duplicates, regardless of overlap: the enrichment pass is
    allowed to add (say) a ``cup`` box that overlaps an existing ``bottle`` box.

    Args:
        det_xyxy: The candidate detection box as ``(x1, y1, x2, y2)``.
        det_target_id: The candidate detection's target YOLO id.
        existing: ``[(target_id, (x1, y1, x2, y2)), ...]`` already present for
            this image (v1 boxes plus boxes accepted earlier in the same pass).
        iou_threshold: IoU at or above which a same-class box is a duplicate.

    Returns:
        True if the detection should be dropped as a duplicate.
    """
    for existing_target_id, existing_xyxy in existing:
        if existing_target_id != det_target_id:
            continue
        if iou(det_xyxy, existing_xyxy) >= iou_threshold:
            return True
    return False
