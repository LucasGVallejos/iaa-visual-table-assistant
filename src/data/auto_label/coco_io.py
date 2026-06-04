"""
COCO read/inject/write primitive for the auto-labeling pass.

This module is the write primitive the later enrichment phases build on. The
auto-labeling workstream runs a pretrained detector over the raw Open Images
COCO export (``datasets/raw_datasets/open_images_subset/labels.json``) and
injects the detector's boxes as new annotations — typically a new ``Food``
category that the original 10-class export lacks. Afterwards notebook 01's
``convert_open_images_to_yolo`` reads the enriched COCO and produces the YOLO
dataset, mapping each annotation's category NAME to a target id via
``configs/label_mapping.yaml``.

The helpers here keep that contract minimal and explicit:

- ``load_coco`` / ``save_coco`` read and write the document. Writes are
  compact (no indentation): the real file is ~9 MB for ~13k images and
  pretty-printing would bloat it for no benefit.
- ``ensure_category`` adds a category by name idempotently and returns its id.
- ``add_annotation`` appends a well-formed COCO annotation.

Injected annotations only strictly need a valid ``category_id`` + ``bbox`` for
the downstream YOLO converter, but we also write ``id``, ``image_id``,
``area`` and ``iscrowd`` so the document stays well-formed COCO.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_coco(path: Path) -> dict:
    """
    Load a COCO Detection JSON document.

    Args:
        path: Path to the COCO ``labels.json``.

    Returns:
        The parsed COCO document as a dict.

    Raises:
        FileNotFoundError: If ``path`` does not exist, with guidance on where
            the Open Images COCO export is expected to live.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"COCO labels file not found: {path}. Expected the Open Images "
            "COCO export (e.g. datasets/raw_datasets/open_images_subset/"
            "labels.json). Did the raw dataset extract correctly?"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_coco(coco: dict, path: Path) -> None:
    """
    Write a COCO document to disk compactly.

    Uses compact separators and no indentation on purpose: the real export is
    large and indentation only inflates the file. Creates the parent directory
    if it does not already exist.

    Args:
        coco: The COCO document to serialize.
        path: Destination path for the JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(coco, f, separators=(",", ":"))


def max_category_id(coco: dict) -> int:
    """Return the largest existing category id, or 0 when there are none."""
    categories = coco.get("categories", []) or []
    if not categories:
        return 0
    return max(int(cat["id"]) for cat in categories)


def max_annotation_id(coco: dict) -> int:
    """Return the largest existing annotation id, or 0 when there are none."""
    annotations = coco.get("annotations", []) or []
    if not annotations:
        return 0
    return max(int(ann["id"]) for ann in annotations)


def next_annotation_id(coco: dict) -> int:
    """Return the next free annotation id (``max_annotation_id + 1``)."""
    return max_annotation_id(coco) + 1


def category_name_to_id(coco: dict) -> dict[str, int]:
    """Return a ``{category_name: category_id}`` lookup for all categories."""
    categories = coco.get("categories", []) or []
    return {str(cat["name"]): int(cat["id"]) for cat in categories}


def category_id_to_name(coco: dict) -> dict[int, str]:
    """Return a ``{category_id: category_name}`` lookup for all categories."""
    categories = coco.get("categories", []) or []
    return {int(cat["id"]): str(cat["name"]) for cat in categories}


def ensure_category(coco: dict, name: str, supercategory: str | None = None) -> int:
    """
    Return the id of the category named ``name``, creating it if absent.

    Idempotent: if a category with ``name`` already exists, its existing id is
    returned and no new category is appended. Otherwise a new category is
    appended with ``id = max_category_id(coco) + 1``. Mutates
    ``coco["categories"]`` in place, creating the list if it is missing.

    Args:
        coco: The COCO document to inspect and mutate.
        name: The category name to ensure (e.g. ``"Food"``).
        supercategory: Optional supercategory for a newly created category.
            Ignored when the category already exists.

    Returns:
        The id of the (existing or newly created) category.
    """
    categories = coco.setdefault("categories", [])

    for cat in categories:
        if str(cat["name"]) == name:
            return int(cat["id"])

    new_id = max_category_id(coco) + 1
    categories.append({"id": new_id, "name": name, "supercategory": supercategory})
    return new_id


def add_annotation(
    coco: dict,
    image_id: int,
    category_id: int,
    bbox_xywh: list[float],
    ann_id: int,
    extra: dict | None = None,
) -> dict:
    """
    Append a well-formed COCO annotation and return it.

    Builds ``{"id", "image_id", "category_id", "bbox", "area", "iscrowd"}``
    where ``bbox`` is the COCO pixel ``[x, y, w, h]`` and ``area = w * h``.
    The caller supplies ``ann_id`` from a running counter that starts at
    ``next_annotation_id(coco)``, so injecting many boxes stays cheap (no
    repeated scans of the annotation list).

    When ``extra`` is given, its keys are merged onto the annotation before it
    is appended. The enrichment pass uses this to tag injected boxes with
    provenance, e.g. ``{"score": conf, "source": "auto_label"}``, which keeps
    the document well-formed COCO (the extra keys are simply ignored by the
    downstream name-based YOLO converter) while letting later steps tell
    auto-labeled boxes apart from the original export.

    Args:
        coco: The COCO document to mutate. ``coco["annotations"]`` is created
            if absent.
        image_id: The image this annotation belongs to.
        category_id: The category id for this annotation.
        bbox_xywh: COCO pixel bounding box ``[x_min, y_min, width, height]``.
        ann_id: The id to assign to the new annotation.
        extra: Optional extra fields merged onto the annotation (e.g. a
            detector ``score`` and a ``source`` tag). ``None`` leaves the
            annotation as the bare well-formed COCO shape.

    Returns:
        The annotation dict that was appended.
    """
    x, y, w, h = bbox_xywh
    annotation = {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [x, y, w, h],
        "area": w * h,
        "iscrowd": 0,
    }
    if extra:
        annotation.update(extra)
    coco.setdefault("annotations", []).append(annotation)
    return annotation
