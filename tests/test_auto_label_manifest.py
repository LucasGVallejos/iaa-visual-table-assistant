"""Unit tests for ``src.data.auto_label.manifest`` (pure CSV logic, no torch)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.data.auto_label.manifest import (
    load_manifest_rows,
    select_open_images_stems,
    summarize_manifest,
)
from src.data.preparation.prepare_dataset import MANIFEST_HEADER


def _write_manifest(path: Path, rows: list[list[str]]) -> Path:
    """Write a manifest CSV with the canonical header plus ``rows``."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(rows)
    return path


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    rows = [
        ["00000003_food_fork.jpg", "open_images", "data/aaa.jpg", "aaa"],
        ["00000001_food.jpg", "uec_food", "1/100.jpg", ""],
        ["00000002_bottle_cup.jpg", "open_images", "data/bbb.jpg", "bbb"],
        ["00000004_food.jpg", "uec_food", "2/200.jpg", ""],
        # Duplicate stem with a different extension casing to exercise dedup.
        ["00000003_food_fork.jpg", "open_images", "data/aaa.jpg", "aaa"],
    ]
    return _write_manifest(tmp_path / "dataset_manifest.csv", rows)


def test_load_manifest_rows_missing_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_manifest_rows(missing)


def test_select_open_images_stems_sorted_unique_no_extension(manifest: Path) -> None:
    stems = select_open_images_stems(manifest)
    # Only open_images rows, stripped of .jpg, sorted, deduplicated.
    assert stems == ["00000002_bottle_cup", "00000003_food_fork"]
    assert all(not s.endswith(".jpg") and not s.endswith(".txt") for s in stems)


def test_summarize_manifest_counts(manifest: Path) -> None:
    counts = summarize_manifest(manifest)
    assert counts["open_images"] == 3
    assert counts["uec_food"] == 2
    assert counts["total"] == 5
