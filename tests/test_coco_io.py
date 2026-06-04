"""
Unit tests for ``src.data.auto_label.coco_io``.

These are fast and dependency-free: they build a small in-memory COCO dict and
exercise the read/inject/write primitives without touching the real
``labels.json`` (round-trips use ``tmp_path``). No torch/ultralytics/PIL.
"""

from __future__ import annotations

import copy

from src.data.auto_label.coco_io import (
    add_annotation,
    category_id_to_name,
    category_name_to_id,
    ensure_category,
    load_coco,
    max_annotation_id,
    max_category_id,
    next_annotation_id,
    save_coco,
)


def make_coco() -> dict:
    """Build a small, valid in-memory COCO document for tests."""
    return {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "Fork", "supercategory": None},
            {"id": 2, "name": "Bottle", "supercategory": None},
        ],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "b.jpg", "width": 800, "height": 600},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 30],
             "area": 600, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [5, 5, 40, 50],
             "area": 2000, "iscrowd": 0},
        ],
    }


# ---------------------------------------------------------------------------
# ensure_category
# ---------------------------------------------------------------------------
def test_ensure_category_adds_fresh_id():
    coco = make_coco()
    new_id = ensure_category(coco, "Food")

    # Fresh id is max existing (2) + 1.
    assert new_id == 3
    assert category_name_to_id(coco)["Food"] == 3
    assert category_id_to_name(coco)[3] == "Food"
    assert len(coco["categories"]) == 3


def test_ensure_category_is_idempotent():
    coco = make_coco()
    first = ensure_category(coco, "Food")
    count_after_first = len(coco["categories"])

    second = ensure_category(coco, "Food")

    # Same id, no duplicate appended.
    assert first == second == 3
    assert len(coco["categories"]) == count_after_first


def test_ensure_category_existing_name_returns_existing_id():
    coco = make_coco()
    before = len(coco["categories"])

    returned = ensure_category(coco, "Fork")

    assert returned == 1
    assert len(coco["categories"]) == before


def test_ensure_category_creates_categories_list_when_absent():
    coco = {"images": [], "annotations": []}
    new_id = ensure_category(coco, "Food")

    # max_category_id of an empty/absent list is 0, so first id is 1.
    assert new_id == 1
    assert coco["categories"] == [{"id": 1, "name": "Food", "supercategory": None}]


def test_ensure_category_supercategory_propagates():
    coco = make_coco()
    ensure_category(coco, "Food", supercategory="edible")
    food = next(c for c in coco["categories"] if c["name"] == "Food")
    assert food["supercategory"] == "edible"


# ---------------------------------------------------------------------------
# id helpers
# ---------------------------------------------------------------------------
def test_max_ids():
    coco = make_coco()
    assert max_category_id(coco) == 2
    assert max_annotation_id(coco) == 2


def test_next_annotation_id():
    coco = make_coco()
    assert next_annotation_id(coco) == 3


def test_next_annotation_id_when_empty():
    coco = {"categories": [], "images": [], "annotations": []}
    assert max_annotation_id(coco) == 0
    assert next_annotation_id(coco) == 1


# ---------------------------------------------------------------------------
# add_annotation
# ---------------------------------------------------------------------------
def test_add_annotation_appends_and_computes_area():
    coco = make_coco()
    before = len(coco["annotations"])
    ann_id = next_annotation_id(coco)

    ann = add_annotation(
        coco,
        image_id=2,
        category_id=3,
        bbox_xywh=[15, 25, 10, 8],
        ann_id=ann_id,
    )

    assert len(coco["annotations"]) == before + 1
    assert ann is coco["annotations"][-1]
    assert ann["id"] == ann_id == 3
    assert ann["image_id"] == 2
    assert ann["category_id"] == 3
    assert ann["bbox"] == [15, 25, 10, 8]
    assert ann["area"] == 80  # 10 * 8
    assert ann["iscrowd"] == 0


def test_add_annotation_creates_annotations_list_when_absent():
    coco = {"categories": [], "images": []}
    add_annotation(coco, image_id=1, category_id=1, bbox_xywh=[0, 0, 2, 3], ann_id=1)
    assert coco["annotations"][0]["area"] == 6


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------
def test_round_trip_inject_save_load(tmp_path):
    coco = make_coco()
    original_image_count = len(coco["images"])

    food_id = ensure_category(coco, "Food")
    ann_id = next_annotation_id(coco)
    add_annotation(
        coco,
        image_id=1,
        category_id=food_id,
        bbox_xywh=[100.5, 50.0, 30.0, 40.0],
        ann_id=ann_id,
    )

    out_path = tmp_path / "labels.json"
    save_coco(coco, out_path)
    assert out_path.exists()

    reloaded = load_coco(out_path)

    # New category survived the round-trip with the right id and name.
    assert category_name_to_id(reloaded)["Food"] == food_id

    # New annotation survived with correct values.
    food_anns = [a for a in reloaded["annotations"] if a["category_id"] == food_id]
    assert len(food_anns) == 1
    ann = food_anns[0]
    assert ann["id"] == ann_id
    assert ann["image_id"] == 1
    assert ann["bbox"] == [100.5, 50.0, 30.0, 40.0]
    assert ann["area"] == 30.0 * 40.0
    assert ann["iscrowd"] == 0

    # Untouched parts are preserved.
    assert len(reloaded["images"]) == original_image_count


def test_save_coco_is_compact(tmp_path):
    coco = make_coco()
    out_path = tmp_path / "labels.json"
    save_coco(coco, out_path)
    text = out_path.read_text(encoding="utf-8")

    # Compact separators: no ", " or ": " spacing, no newlines from indent.
    assert ", " not in text
    assert ": " not in text
    assert "\n" not in text


def test_save_coco_creates_parent_dir(tmp_path):
    coco = make_coco()
    out_path = tmp_path / "nested" / "deep" / "labels.json"
    save_coco(coco, out_path)
    assert out_path.exists()


def test_load_coco_missing_raises(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    try:
        load_coco(missing)
    except FileNotFoundError as e:
        assert "not found" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_functions_do_not_mutate_unrelated_state():
    coco = make_coco()
    snapshot = copy.deepcopy(coco["images"])
    ensure_category(coco, "Food")
    add_annotation(coco, image_id=1, category_id=3, bbox_xywh=[0, 0, 1, 1], ann_id=3)
    # Images list must be untouched by category/annotation injection.
    assert coco["images"] == snapshot
