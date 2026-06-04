"""
Unit tests for ``src.data.auto_label.coco_target_mapping``.

These run against the REAL ``configs/label_mapping.yaml`` and are
dependency-free (no torch/ultralytics/PIL): they only exercise the pure mapping
logic and verify the config is internally consistent with the target classes.
"""

from __future__ import annotations

from src.data.auto_label.coco_target_mapping import (
    TARGET_TO_COCO_CATEGORY,
    load_coco_pretrained_mapping,
    map_detection_name,
)
from src.data.common.convert_to_yolo import load_label_mapping


# ---------------------------------------------------------------------------
# load_coco_pretrained_mapping
# ---------------------------------------------------------------------------
def test_mapping_loads_and_spot_checks():
    mapping = load_coco_pretrained_mapping()

    assert mapping["cup"] == 1
    assert mapping["wine glass"] == 1
    assert mapping["bowl"] == 3
    assert mapping["banana"] == 0
    assert mapping["knife"] == 6


def test_mapping_values_within_target_range():
    mapping = load_coco_pretrained_mapping()
    assert all(0 <= target_id <= 6 for target_id in mapping.values())


def test_mapping_values_covered_by_target_to_coco_category():
    mapping = load_coco_pretrained_mapping()
    assert all(target_id in TARGET_TO_COCO_CATEGORY for target_id in mapping.values())


# ---------------------------------------------------------------------------
# map_detection_name
# ---------------------------------------------------------------------------
def test_map_detection_name_is_case_insensitive():
    mapping = load_coco_pretrained_mapping()
    assert map_detection_name("Cup", mapping) == 1


def test_map_detection_name_drops_unmapped():
    mapping = load_coco_pretrained_mapping()
    assert map_detection_name("person", mapping) is None


# ---------------------------------------------------------------------------
# open_images section now carries injected "Food"
# ---------------------------------------------------------------------------
def test_open_images_maps_food_to_zero():
    doc = load_label_mapping()
    assert doc["open_images"]["Food"] == 0
