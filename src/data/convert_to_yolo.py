"""
Annotation conversion utilities for YOLO format.

This module contains small helper functions for converting bounding boxes
from common annotation formats to YOLO normalized xywh format.

Dataset-specific converters will be implemented later, once the selected
datasets are inspected individually.
"""

from pathlib import Path

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


def write_yolo_label(output_path: Path, annotations: list[tuple[int, list[float]]]) -> None:
    """Write annotations in YOLO format to a label file."""
    with open(output_path, "w") as f:
        for class_id, bbox in annotations:
            line = f"{class_id} {' '.join(f'{v:.6f}' for v in bbox)}\n"
            f.write(line)


def load_class_mapping(config_path: str = "configs/classes.yaml") -> dict[str, int]:
    """Load class name to ID mapping from config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return {cls["name"]: cls["id"] for cls in config["classes"]}


def main():
    """Run annotation conversion pipeline."""
    class_map = load_class_mapping()
    print(f"Loaded {len(class_map)} classes: {class_map}")


if __name__ == "__main__":
    main()
