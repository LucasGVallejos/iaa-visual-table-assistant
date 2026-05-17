from PIL import Image
from pathlib import Path

from src.utils.paths import (
  get_uec_staging_dir,
  get_uecfood256_dataset_original_dir,
)
from src.data.common.convert_to_yolo import (
  load_label_mapping,
  voc_to_yolo,
)
from src.data.common.dataset_io import (
  append_skipped_image,
  clip_yolo_bbox,
  get_skipped_images_csv_path,
  init_skipped_images_csv,
  is_valid_yolo_bbox,
  reset_yolo_staging_dir,
  write_yolo_sample,
)

SOURCE_LABEL = "uec_food_256"

# NOTE 1: UECFOOD256 only has bounding boxes for food. Open Images conversion
# will be a separate module that adds bboxes for non-food classes.
# NOTE 2: No origin -> destination map is needed here; staging filenames
# `<3-digit-category>_<imgid>.jpg` already encode source provenance per plan
# Etapa C. The global manifest is produced later by Etapa D (merge + rename).


def get_food_class_id() -> int:
  """Resolve the YOLO class_id for UEC FOOD-256 from `configs/label_mapping.yaml`."""
  label_mapping = load_label_mapping()
  try:
    return int(label_mapping[SOURCE_LABEL]["default"])
  except KeyError as exc:
    raise KeyError(
      f"Missing '{SOURCE_LABEL}.default' in configs/label_mapping.yaml"
    ) from exc


def step_01_reset_staging() -> tuple[Path, Path]:
  """Reset the UEC FOOD-256 staging directory and re-init its skipped-images log.

  Returns ``(staging_dir, skipped_csv_path)``. The skipped CSV is per-source so
  this never touches logs from other datasets.
  """
  staging = reset_yolo_staging_dir(get_uec_staging_dir())

  skipped_csv = get_skipped_images_csv_path(SOURCE_LABEL)
  init_skipped_images_csv(skipped_csv, overwrite=True)

  return staging, skipped_csv


def step_02_process_uec_folder(
  destination_route: Path,
  food_class_id: int,
  skipped_csv: Path,
) -> None:
  original_route = get_uecfood256_dataset_original_dir()

  def log_skipped(original_path: Path, reason: str) -> None:
    append_skipped_image(skipped_csv, SOURCE_LABEL, original_path, reason)

  def get_all_category_folders() -> list[Path]:
    """Returns a sorted list of all category folders in the UEC FOOD-256 dataset."""
    return sorted(
      (p for p in original_route.iterdir() if p.is_dir() and p.name.isdigit()),
      key=lambda p: int(p.name),
    )

  category_folders = get_all_category_folders()

  def get_bb_info_txt(folder: Path) -> list[str]:
    """Returns data lines of `bb_info.txt` for a category folder (header stripped)."""
    bb_info_path = folder / "bb_info.txt"
    if not bb_info_path.exists():
      return []
    with open(bb_info_path, "r", encoding="utf-8") as f:
      return f.readlines()[1:]

  def get_map_images_with_bb(bb_info: list[str]) -> dict[str, list[tuple[int, int, int, int]]]:
    """Returns a dictionary mapping image IDs to their bounding boxes."""
    map_image_to_bb: dict[str, list[tuple[int, int, int, int]]] = {}
    for line in bb_info:
      parts = line.strip().split()
      if len(parts) < 5:
        continue
      img_id = parts[0]
      try:
        x1, y1, x2, y2 = (int(v) for v in parts[1:5])
      except ValueError:
        continue
      map_image_to_bb.setdefault(img_id, []).append((x1, y1, x2, y2))

    return map_image_to_bb

  def get_map_images_with_bb_yolo(
    folder: Path,
    map_image_to_bb: dict[str, list[tuple[int, int, int, int]]],
  ) -> tuple[dict[str, list[list[float]]], int, int]:
    """
    Convert each image's bboxes to YOLO format, clip to [0, 1], drop degenerate.

    Returns (yolo_map, skipped_images, dropped_bboxes). An image is skipped if
    it cannot be opened, or if zero valid bboxes remain after filtering.
    """
    yolo_map: dict[str, list[list[float]]] = {}
    skipped_images = 0
    dropped_bboxes = 0

    for img_id, bbs in map_image_to_bb.items():
      image_path = folder / f"{img_id}.jpg"
      try:
        with Image.open(image_path) as image:
          width, height = image.size
      except Exception as e:
        log_skipped(image_path, f"open failed: {type(e).__name__}")
        skipped_images += 1
        continue

      valid_bboxes: list[list[float]] = []
      for bb in bbs:
        yolo = voc_to_yolo(bb, img_width=width, img_height=height)
        clipped = clip_yolo_bbox(yolo)

        if not is_valid_yolo_bbox(clipped):
          log_skipped(image_path, f"degenerate bbox after clip: {clipped}")
          dropped_bboxes += 1
          continue

        valid_bboxes.append(clipped)

      if not valid_bboxes:
        log_skipped(image_path, "no valid bboxes after filtering")
        skipped_images += 1
        continue

      yolo_map[img_id] = valid_bboxes

    return yolo_map, skipped_images, dropped_bboxes

  def store_images_in_yolo_format(
    staging_folder: Path,
    source_folder: Path,
    map_image_bb_yolo: dict[str, list[list[float]]],
  ) -> tuple[int, int]:
    """
    Write YOLO labels and re-encoded images to staging.

    Image filename in staging: `<3-digit-category>_<imgid>.jpg` per plan Etapa C.
    `write_yolo_sample` writes the image first, then the label, and rolls both
    back if either step fails.
    """
    category = source_folder.name.zfill(3)
    stored = 0
    write_failed = 0

    for img_id, bbs in map_image_bb_yolo.items():
      stem = f"{category}_{img_id}"
      label_path = staging_folder / "labels" / f"{stem}.txt"
      image_path = staging_folder / "images" / f"{stem}.jpg"
      source_image_path = source_folder / f"{img_id}.jpg"
      annotations = [(food_class_id, bbox) for bbox in bbs]

      try:
        write_yolo_sample(
          source_image_path=source_image_path,
          output_image_path=image_path,
          output_label_path=label_path,
          annotations=annotations,
        )
        stored += 1
      except Exception as e:
        log_skipped(source_image_path, f"write failed: {type(e).__name__}")
        write_failed += 1

    return stored, write_failed

  def process_category_folders() -> None:
    for folder in category_folders:
      print(f"===Processing category {folder.name}===")

      bb_info = get_bb_info_txt(folder)
      if not bb_info:
        print("  bb_info.txt missing or empty, skipping category")
        continue

      map_image_to_bb = get_map_images_with_bb(bb_info)
      print("> parsed bb_info, found", len(map_image_to_bb), "images with bboxes")
      print("> converting to YOLO format...")
      yolo_map, skipped_imgs, dropped_bbs = get_map_images_with_bb_yolo(folder, map_image_to_bb)
      print("> storing images and labels...")
      stored, write_failed = store_images_in_yolo_format(destination_route, folder, yolo_map)
      print(
        f"  stored={stored} skipped_imgs={skipped_imgs} "
        f"dropped_bboxes={dropped_bbs} write_failed={write_failed}"
      )

  process_category_folders()


def main() -> Path:
  """Start the UEC FOOD-256 → YOLO conversion migration."""
  food_class_id = get_food_class_id()
  staging, skipped_csv = step_01_reset_staging()

  step_02_process_uec_folder(staging, food_class_id, skipped_csv)
  # yes, 2 steps

  return staging


# Run with `python -m src.data.conversion.convert_uec_food_to_yolo`
if __name__ == "__main__":
  main()
