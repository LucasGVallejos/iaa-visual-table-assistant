"""
Common dataset I/O helpers shared across YOLO converters.

Designed to be reused by per-source converters (UEC FOOD-256, Open Images, ...)
so that staging reset, skipped-image logging, bbox clipping/validation, and
atomic image+label writes are consistent everywhere.
"""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

from src.data.common.convert_to_yolo import write_image_in_yolo, write_yolo_annotations
from src.utils.paths import get_reports_dir


_SKIPPED_CSV_HEADER = ["source", "original_path", "reason"]


def reset_yolo_staging_dir(staging_dir: Path) -> Path:
  """Wipe and re-create a YOLO staging directory.

  Removes ``staging_dir`` if present, then re-creates ``images/`` and ``labels/``
  subdirectories. Returns ``staging_dir`` for convenience.
  """
  if staging_dir.exists():
    shutil.rmtree(staging_dir)
  (staging_dir / "images").mkdir(parents=True, exist_ok=True)
  (staging_dir / "labels").mkdir(parents=True, exist_ok=True)
  return staging_dir


def _sanitize_source_name(source: str) -> str:
  """Turn an arbitrary source label into a safe filename stem."""
  cleaned = re.sub(r"[^a-z0-9_-]+", "_", source.strip().lower())
  cleaned = cleaned.strip("_") or "source"
  return cleaned


def get_skipped_images_csv_path(source: str) -> Path:
  """Return the per-source skipped-images CSV path.

  Example: ``source="uec_food_256"`` -> ``reports/skipped_images/uec_food_256.csv``.
  """
  filename = f"{_sanitize_source_name(source)}.csv"
  return get_reports_dir() / "skipped_images" / filename


def init_skipped_images_csv(csv_path: Path, overwrite: bool = True) -> None:
  """Ensure the skipped-images CSV exists with the standard header.

  If ``overwrite`` is True, the file is (re)created with the header. Otherwise
  the file is left untouched when present, and only created when missing.
  """
  csv_path.parent.mkdir(parents=True, exist_ok=True)
  if not overwrite and csv_path.exists():
    return
  with open(csv_path, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(_SKIPPED_CSV_HEADER)


def append_skipped_image(
  csv_path: Path,
  source: str,
  original_path: Path,
  reason: str,
) -> None:
  """Append one ``source, original_path, reason`` row to the skipped CSV."""
  with open(csv_path, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow([source, str(original_path), reason])


def clip_yolo_bbox(bbox: list[float]) -> list[float]:
  """Clip every coordinate of a YOLO bbox to ``[0.0, 1.0]``."""
  return [min(1.0, max(0.0, v)) for v in bbox]


def is_valid_yolo_bbox(bbox: list[float], min_size: float = 1e-6) -> bool:
  """Return True if a YOLO ``[cx, cy, w, h]`` bbox has positive width and height."""
  _, _, w, h = bbox
  return w > min_size and h > min_size


def write_yolo_sample(
  source_image_path: Path,
  output_image_path: Path,
  output_label_path: Path,
  annotations: list[tuple[int, list[float]]],
) -> None:
  """Write one YOLO sample (image + label) atomically.

  The image is written first, the label second, so a label without its image
  cannot exist on disk. If either step fails, both output files are removed
  before re-raising the original exception.
  """
  try:
    write_image_in_yolo(source_image_path, output_image_path)
    write_yolo_annotations(output_label_path, annotations)
  except Exception:
    if output_image_path.exists():
      output_image_path.unlink()
    if output_label_path.exists():
      output_label_path.unlink()
    raise
