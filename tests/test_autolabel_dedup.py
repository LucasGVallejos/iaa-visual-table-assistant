"""
Unit tests for the phase-6 geometry/de-duplication helpers in
``src.data.auto_label.coco_target_mapping``.

These are dependency-free (no torch/ultralytics/PIL): they only exercise the
pure box math — ``xywh_to_xyxy``, ``iou`` and ``is_duplicate`` — that the
enrichment pass relies on to drop boxes already present in the v1 export.
"""

from __future__ import annotations

import pytest

from src.data.auto_label.coco_target_mapping import (
    is_duplicate,
    iou,
    xywh_to_xyxy,
)


# ---------------------------------------------------------------------------
# xywh_to_xyxy
# ---------------------------------------------------------------------------
def test_xywh_to_xyxy_basic():
    assert xywh_to_xyxy([10.0, 20.0, 30.0, 40.0]) == (10.0, 20.0, 40.0, 60.0)


def test_xywh_to_xyxy_zero_origin():
    assert xywh_to_xyxy([0.0, 0.0, 5.0, 7.0]) == (0.0, 0.0, 5.0, 7.0)


# ---------------------------------------------------------------------------
# iou
# ---------------------------------------------------------------------------
def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (20.0, 20.0, 30.0, 30.0)
    assert iou(a, b) == 0.0


def test_iou_half_overlap_hand_computed():
    # a and b are both 10x10 (area 100 each); b is shifted right by 5 so the
    # overlap region is 5 wide x 10 tall = 50. union = 100 + 100 - 50 = 150.
    # IoU = 50 / 150 = 1/3.
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    assert iou(a, b) == pytest.approx(50.0 / 150.0)


def test_iou_degenerate_union_is_zero():
    # Two zero-area boxes -> union 0 -> defensively 0.0, not a ZeroDivisionError.
    a = (5.0, 5.0, 5.0, 5.0)
    b = (5.0, 5.0, 5.0, 5.0)
    assert iou(a, b) == 0.0


def test_iou_touching_edges_is_zero():
    # Share only an edge: intersection area is 0.
    a = (0.0, 0.0, 10.0, 10.0)
    b = (10.0, 0.0, 20.0, 10.0)
    assert iou(a, b) == 0.0


# ---------------------------------------------------------------------------
# is_duplicate
# ---------------------------------------------------------------------------
def test_is_duplicate_same_class_overlap_is_true():
    det = (0.0, 0.0, 10.0, 10.0)
    existing = [(0, (1.0, 1.0, 11.0, 11.0))]  # same class, high overlap
    assert is_duplicate(det, 0, existing, iou_threshold=0.5) is True


def test_is_duplicate_different_class_same_overlap_is_false():
    det = (0.0, 0.0, 10.0, 10.0)
    existing = [(1, (1.0, 1.0, 11.0, 11.0))]  # different class id
    assert is_duplicate(det, 0, existing, iou_threshold=0.5) is False


def test_is_duplicate_below_threshold_is_false():
    det = (0.0, 0.0, 10.0, 10.0)
    # Same class but only ~1/3 IoU (the half-overlap case), below 0.5.
    existing = [(0, (5.0, 0.0, 15.0, 10.0))]
    assert is_duplicate(det, 0, existing, iou_threshold=0.5) is False


def test_is_duplicate_empty_existing_is_false():
    det = (0.0, 0.0, 10.0, 10.0)
    assert is_duplicate(det, 0, [], iou_threshold=0.5) is False


def test_is_duplicate_at_threshold_is_true():
    # Exactly 1/3 IoU and threshold set to that value -> >= so duplicate.
    det = (0.0, 0.0, 10.0, 10.0)
    existing = [(0, (5.0, 0.0, 15.0, 10.0))]
    assert is_duplicate(det, 0, existing, iou_threshold=50.0 / 150.0) is True
