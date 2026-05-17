"""Open Images COCO Detection -> YOLO staging converter.

Reads a COCO Detection JSON (``labels.json``) plus its images, maps source
category names to YOLO class IDs via ``configs/label_mapping.yaml``, and writes
re-encoded images + multi-class YOLO labels to the Open Images staging area.

Run with::

    python -m src.data.conversion.convert_open_images_to_yolo
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from PIL import Image

from src.data.common.convert_to_yolo import coco_to_yolo, load_label_mapping
from src.data.common.dataset_io import (
  append_skipped_image,
  clip_yolo_bbox,
  get_skipped_images_csv_path,
  init_skipped_images_csv,
  is_valid_yolo_bbox,
  reset_yolo_staging_dir,
  write_yolo_sample,
)
from src.utils.paths import (
  get_open_images_dataset_original_dir,
  get_open_images_staging_dir,
)


SOURCE_LABEL = "open_images"
STAGING_PREFIX = "oi"


# ---------------------------------------------------------------------------
# Skipped logging
# ---------------------------------------------------------------------------
def log_skipped(skipped_csv: Path, original_path: Path, reason: str) -> None:
  """Append a skip row to the Open Images skipped-images CSV."""
  append_skipped_image(
    csv_path=skipped_csv,
    source=SOURCE_LABEL,
    original_path=original_path,
    reason=reason,
  )
  print(f"SKIPPED IMAGE({reason}): {original_path}")


# ---------------------------------------------------------------------------
# Step 01 — staging reset (only Open Images)
# ---------------------------------------------------------------------------
def step_01_reset_staging() -> tuple[Path, Path]:
  """Reset the Open Images staging directory and re-init its skipped-images log.

  Returns ``(staging_dir, skipped_csv_path)``. Only touches Open Images
  artifacts; UEC staging and other dataset CSVs are left untouched.
  """
  staging = reset_yolo_staging_dir(get_open_images_staging_dir())

  skipped_csv = get_skipped_images_csv_path(SOURCE_LABEL)
  init_skipped_images_csv(skipped_csv, overwrite=True)

  return staging, skipped_csv


# ---------------------------------------------------------------------------
# COCO discovery / loading
# ---------------------------------------------------------------------------
def find_coco_json(open_images_dir: Path) -> Path:
  """Find the COCO labels JSON under ``open_images_dir``.

  Picks the alphabetically first ``*.json`` file directly inside the directory.
  Raises ``FileNotFoundError`` if none is found.
  """
  candidates = sorted(open_images_dir.glob("*.json"))
  if not candidates:
    raise FileNotFoundError(
      f"No COCO labels JSON found in {open_images_dir}"
    )
  if len(candidates) > 1:
    print("> multiple COCO JSON files found:")
    for c in candidates:
      print(f"  - {c.name}")
    print(f"> using first alphabetically: {candidates[0].name}")
  return candidates[0]


def resolve_open_images_path(base_dir: Path, file_name: str) -> Path:
  """Resolve an image file path inside the Open Images export.

  Tries common candidate locations first (root and ``data/``), then falls
  back to ``rglob`` by the bare filename. Raises ``FileNotFoundError`` if
  nothing matches.
  """
  bare_name = Path(file_name).name
  candidates = [
    base_dir / file_name,
    base_dir / "data" / file_name,
    base_dir / bare_name,
    base_dir / "data" / bare_name,
  ]
  for candidate in candidates:
    if candidate.exists():
      return candidate

  for found in base_dir.rglob(bare_name):
    if found.is_file():
      return found

  raise FileNotFoundError(f"Image not found in {base_dir}: {file_name}")


def load_open_images_coco(coco_json_path: Path) -> dict:
  """Load the COCO Detection JSON document."""
  with open(coco_json_path, "r", encoding="utf-8") as f:
    return json.load(f)


def build_coco_indexes(
  coco: dict,
) -> tuple[dict[int, str], dict[int, dict], dict[int, list[dict]]]:
  """Build lookup tables from the COCO document.

  Returns:
    categories_by_id: ``category_id -> source_label`` (the *name* in COCO).
    images_by_id: ``image_id -> image_info`` dict.
    annotations_by_image_id: ``image_id -> list[annotation]``.

  Note: ``category_id`` is **not** a YOLO class_id. It is only used to look
  up the source label, which is then mapped via ``label_mapping.yaml``.
  """
  categories_by_id: dict[int, str] = {
    int(cat["id"]): cat["name"] for cat in coco.get("categories", [])
  }

  images_by_id: dict[int, dict] = {
    int(img["id"]): img for img in coco.get("images", [])
  }

  annotations_by_image_id: dict[int, list[dict]] = {}
  for ann in coco.get("annotations", []):
    annotations_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

  return categories_by_id, images_by_id, annotations_by_image_id


# ---------------------------------------------------------------------------
# Per-image helpers
# ---------------------------------------------------------------------------
def get_image_size(image_info: dict, image_path: Path) -> tuple[int, int]:
  """Resolve ``(width, height)`` for an image.

  Prefers the COCO ``width``/``height`` fields. Falls back to opening the file
  with PIL when either is missing or non-positive. PIL ``image.size`` is
  ``(width, height)``.
  """
  width = image_info.get("width")
  height = image_info.get("height")
  if width and height and int(width) > 0 and int(height) > 0:
    return int(width), int(height)

  with Image.open(image_path) as image:
    return image.size  # (width, height)


def convert_annotations_for_image(
  image_annotations: list[dict],
  categories_by_id: dict[int, str],
  open_images_mapping: dict[str, int],
  img_width: int,
  img_height: int,
) -> tuple[list[tuple[int, list[float]]], int, int]:
  """Convert all COCO annotations for one image to YOLO ``(class_id, bbox)`` pairs.

  Drops annotations whose source label is not in ``open_images_mapping`` and
  bboxes that are degenerate after clipping. Does not touch the filesystem.

  Returns ``(valid_annotations, dropped_unmapped, dropped_degenerate)``.
  """
  valid_annotations: list[tuple[int, list[float]]] = []
  dropped_unmapped = 0
  dropped_degenerate = 0

  for ann in image_annotations:
    category_id = ann.get("category_id")
    if category_id is None:
      dropped_unmapped += 1
      continue

    source_label = categories_by_id.get(int(category_id))
    if source_label is None or source_label not in open_images_mapping:
      dropped_unmapped += 1
      continue

    target_class_id = int(open_images_mapping[source_label])

    yolo_bbox = coco_to_yolo(
      ann["bbox"], img_width=img_width, img_height=img_height
    )
    yolo_bbox = clip_yolo_bbox(yolo_bbox)

    if not is_valid_yolo_bbox(yolo_bbox):
      dropped_degenerate += 1
      continue

    valid_annotations.append((target_class_id, yolo_bbox))

  return valid_annotations, dropped_unmapped, dropped_degenerate


def make_staging_stem(image_id, image_info: dict) -> str:
  """Build a stable, prefixed staging filename stem for an Open Images image.

  Uses ``oi_<image_id_zero_padded>`` for numeric IDs. For non-numeric IDs,
  derives a safe stem from the ID or the file name and applies the same
  prefix.
  """
  try:
    return f"{STAGING_PREFIX}_{int(image_id):08d}"
  except (TypeError, ValueError):
    pass

  raw = str(image_id)
  if not raw or raw == "None":
    raw = Path(image_info.get("file_name", "")).stem or "image"
  safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_") or "image"
  return f"{STAGING_PREFIX}_{safe}"


def store_image_and_labels(
  source_image_path: Path,
  output_image_path: Path,
  output_label_path: Path,
  valid_annotations: list[tuple[int, list[float]]],
) -> None:
  """Write one Open Images sample (image + multi-class label) to staging.

  Delegates to ``write_yolo_sample`` which writes the image first, then the
  label, and rolls both back atomically if either step fails. Caller is
  responsible for logging skipped images on failure.
  """
  write_yolo_sample(
    source_image_path=source_image_path,
    output_image_path=output_image_path,
    output_label_path=output_label_path,
    annotations=valid_annotations,
  )


# ---------------------------------------------------------------------------
# Step 02 — main processing loop
# ---------------------------------------------------------------------------
def step_02_process_open_images_folder(staging: Path, skipped_csv: Path) -> None:
  """Convert the Open Images COCO export into YOLO staging."""
  open_images_dir = get_open_images_dataset_original_dir()

  coco_json_path = find_coco_json(open_images_dir)
  print(f"> using COCO labels: {coco_json_path}")

  label_mapping = load_label_mapping()
  open_images_mapping = label_mapping.get(SOURCE_LABEL, {}) or {}
  if not open_images_mapping:
    print(
      f"> WARNING: empty '{SOURCE_LABEL}' section in configs/label_mapping.yaml; "
      "all annotations will be dropped"
    )

  print("> loading COCO JSON...")
  coco = load_open_images_coco(coco_json_path)

  print("> building indexes...")
  categories_by_id, images_by_id, annotations_by_image_id = build_coco_indexes(coco)
  print(
    f"  categories={len(categories_by_id)} "
    f"images={len(images_by_id)} "
    f"annotations={sum(len(v) for v in annotations_by_image_id.values())}"
  )

  images_total = 0
  annotations_total = 0
  stored_images = 0
  skipped_images = 0
  written_bboxes = 0
  dropped_unmapped_annotations = 0
  dropped_degenerate_bboxes = 0
  missing_images = 0
  write_failures = 0

  staging_images_dir = staging / "images"
  staging_labels_dir = staging / "labels"

  print("> processing images...")
  for image_id, image_info in images_by_id.items():

    images_total += 1
    file_name = image_info.get("file_name", "")

    try:
      source_image_path = resolve_open_images_path(open_images_dir, file_name)
    except FileNotFoundError:
      log_skipped(skipped_csv, open_images_dir / file_name, "image not found")
      missing_images += 1
      skipped_images += 1
      continue

    try:
      img_width, img_height = get_image_size(image_info, source_image_path)
    except Exception as e:
      log_skipped(
        skipped_csv, source_image_path, f"open failed: {type(e).__name__}"
      )
      skipped_images += 1
      continue

    image_annotations = annotations_by_image_id.get(image_id, [])
    annotations_total += len(image_annotations)

    valid_annotations, dropped_unmapped, dropped_degenerate = (
      convert_annotations_for_image(
        image_annotations,
        categories_by_id,
        open_images_mapping,
        img_width,
        img_height,
      )
    )
    dropped_unmapped_annotations += dropped_unmapped
    dropped_degenerate_bboxes += dropped_degenerate

    if not valid_annotations:
      log_skipped(skipped_csv, source_image_path, "no valid annotations after mapping")
      skipped_images += 1
      continue

    stem = make_staging_stem(image_id, image_info)
    output_image_path = staging_images_dir / f"{stem}.jpg"
    output_label_path = staging_labels_dir / f"{stem}.txt"

    try:
      store_image_and_labels(
        source_image_path=source_image_path,
        output_image_path=output_image_path,
        output_label_path=output_label_path,
        valid_annotations=valid_annotations,
      )
      stored_images += 1
      written_bboxes += len(valid_annotations)
      print(f"Image {image_id} processed with {len(valid_annotations)} annotations of {len(image_annotations)}")
    except Exception as e:
      log_skipped(skipped_csv, source_image_path, f"write failed: {type(e).__name__}")
      write_failures += 1
      skipped_images += 1

  print()
  print("=== Open Images conversion summary ===")
  print(f"  images_total                  = {images_total}")
  print(f"  annotations_total             = {annotations_total}")
  print(f"  stored_images                 = {stored_images}")
  print(f"  written_bboxes                = {written_bboxes}")
  print(f"  skipped_images                = {skipped_images}")
  print(f"    missing_images              = {missing_images}")
  print(f"    write_failures              = {write_failures}")
  print(f"  dropped_unmapped_annotations  = {dropped_unmapped_annotations}")
  print(f"  dropped_degenerate_bboxes     = {dropped_degenerate_bboxes}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> Path:
  """Start the Open Images COCO -> YOLO conversion."""
  staging, skipped_csv = step_01_reset_staging()
  step_02_process_open_images_folder(staging, skipped_csv)
  return staging


# Run with `python -m src.data.conversion.convert_open_images_to_yolo`
if __name__ == "__main__":
  main()
