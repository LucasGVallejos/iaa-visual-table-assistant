"""
Annotation conversion utilities for YOLO format.

This module contains small helper functions for converting bounding boxes
from common annotation formats to YOLO normalized xywh format.

Dataset-specific converters will be implemented later, once the selected
datasets are inspected individually.
"""

from pathlib import Path
from PIL import Image
import yaml


def coco_to_yolo(bbox: list[float], img_width: int, img_height: int) -> list[float]:
    """
    Convert COCO bbox [x_min, y_min, width, height] to YOLO [x_center, y_center, w, h].

    All values are normalized to [0, 1].
    """
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return [x_center, y_center, w_norm, h_norm]


def voc_to_yolo(bbox: list[float], img_width: int, img_height: int) -> list[float]:
    """
    Convert Pascal VOC bbox [x_min, y_min, x_max, y_max] to YOLO format.

    All values are normalized to [0, 1].
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height
    return [x_center, y_center, w, h]


def write_yolo_label(output_path: Path, class_id: int, annotations: list[list[float]]) -> None:
    """Write annotations in YOLO format to a label file.

    Use when every bbox in the image shares the same class_id (e.g. UEC FOOD-256,
    where every box is `food`). For per-bbox class IDs, use `write_yolo_annotations`.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for bbox in annotations:
            line = f"{class_id} {' '.join(f'{v:.6f}' for v in bbox)}\n"
            f.write(line)


def write_yolo_annotations(
    output_path: Path,
    annotations: list[tuple[int, list[float]]],
) -> None:
    """Write YOLO annotations where each bbox carries its own class_id.

    Each entry in `annotations` is a `(class_id, bbox)` pair, where `bbox` is the
    already-normalized YOLO `[x_center, y_center, w, h]`.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for class_id, bbox in annotations:
            line = f"{class_id} {' '.join(f'{v:.6f}' for v in bbox)}\n"
            f.write(line)


def write_image_in_yolo(image_path: Path, output_path: Path) -> None:
  """Write JPEG image to output path, converting to RGB if necessary."""
  Image.open(image_path).convert("RGB").save(output_path, "JPEG", quality=95)

def load_class_mapping(config_path: str | Path = "configs/classes.yaml") -> dict[str, int]:
    """Load model class name -> class_id mapping from `configs/classes.yaml`."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {cls["name"]: cls["id"] for cls in config["classes"]}


def load_classes_config(config_path: str | Path = "configs/classes.yaml") -> list[dict]:
    """Load the full class entries from `configs/classes.yaml`.

    Returns the raw list of dicts under the top-level ``classes`` key, preserving
    every field declared in the config (``id``, ``name``, ``color``,
    ``color_name``, ``description``, ...).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return list(config["classes"])


def load_label_mapping(config_path: str | Path = "configs/label_mapping.yaml") -> dict:
    """Load source-dataset label -> target YOLO class_id mapping.
    Returns the full YAML document.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    """Run annotation conversion pipeline."""
    class_map = load_class_mapping()
    print(f"Loaded {len(class_map)} classes: {class_map}")


if __name__ == "__main__":
    main()
