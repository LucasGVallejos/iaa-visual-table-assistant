"""
Label utilities for working with YOLO annotations.

These helpers are used to inspect class mappings and label files
during dataset preparation and validation.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.utils.paths import get_config_path


YoloLabel = Tuple[int, List[float]]


def load_class_names(config_path: str | Path | None = None) -> Dict[int, str]:
    """
    Load class ID to class name mapping from the YOLO data config.

    Args:
        config_path: Optional path to the data YAML config.
            If not provided, configs/data.yaml is used.

    Returns:
        Dictionary mapping class IDs to class names.
    """
    path = Path(config_path) if config_path else get_config_path("data.yaml")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return {int(class_id): name for class_id, name in config.get("names", {}).items()}


def read_yolo_labels(label_path: str | Path) -> List[YoloLabel]:
    """
    Read a YOLO format label file.

    Each line is expected to follow:
    class_id x_center y_center width height

    Args:
        label_path: Path to the .txt label file.

    Returns:
        List of tuples containing class_id and bounding box values.
    """
    path = Path(label_path)
    labels: List[YoloLabel] = []

    if not path.exists():
        return labels

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            bbox = [float(value) for value in parts[1:]]
            labels.append((class_id, bbox))

    return labels


def count_classes(labels_dir: str | Path) -> Dict[int, int]:
    """
    Count instances of each class across YOLO label files.

    Args:
        labels_dir: Path to a directory containing .txt label files.

    Returns:
        Dictionary mapping class IDs to instance counts.
    """
    labels_path = Path(labels_dir)
    counts: Dict[int, int] = {}

    for label_file in labels_path.glob("*.txt"):
        labels = read_yolo_labels(label_file)

        for class_id, _ in labels:
            counts[class_id] = counts.get(class_id, 0) + 1

    return counts