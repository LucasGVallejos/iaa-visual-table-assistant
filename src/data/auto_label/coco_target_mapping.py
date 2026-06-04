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
