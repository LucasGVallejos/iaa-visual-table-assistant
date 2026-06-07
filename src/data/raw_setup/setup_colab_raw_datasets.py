"""
Prepare and inspect raw datasets from Google Drive in Colab.

This script assumes Google Drive is already mounted at /content/drive.
It locates zip files uploaded to Drive, extracts them into the repo's
``datasets/raw_datasets/`` directory, and inspects the structure of both
Open Images (COCO format) and UEC FOOD-256 datasets.

This script does NOT:
- Convert annotations to YOLO format
- Split into train/val/test
- Train any model
- Use DVC or MLflow

Usage (from Colab, after mounting Drive)::

    !python -m src.data.raw_setup.setup_colab_raw_datasets
"""

import json
import shutil
import zipfile
from pathlib import Path

from src.utils.paths import (
    get_open_images_dataset_original_dir,
    get_uec_food_dataset_extract_dir,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DRIVE_BASE_DIR = Path("/content/drive/MyDrive/iaa-table-assistant")
DRIVE_RAW_DATASETS_DIR = DRIVE_BASE_DIR / "raw_datasets"
DRIVE_OPEN_IMAGES_DIR = DRIVE_RAW_DATASETS_DIR / "open_images_subset_v2"
DRIVE_UEC_FOOD_DIR = DRIVE_RAW_DATASETS_DIR / "uec_food_256"

# Local extraction targets live inside the repo under datasets/raw_datasets/.
LOCAL_OPEN_IMAGES_DIR = get_open_images_dataset_original_dir()
LOCAL_UEC_FOOD_DIR = get_uec_food_dataset_extract_dir()


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def find_zip(directory: Path, dataset_name: str) -> Path:
    """
    Find a zip file in the given directory.

    If multiple zips exist, prints all and returns the first
    alphabetically.

    Args:
        directory: Path to search for zip files.
        dataset_name: Human-readable name for log messages.

    Returns:
        Path to the selected zip file.

    Raises:
        FileNotFoundError: If no zip files are found.
    """
    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found for {dataset_name}: {directory}"
        )

    zips = sorted(directory.glob("*.zip"))

    if not zips:
        raise FileNotFoundError(
            f"No .zip files found for {dataset_name} in {directory}"
        )

    if len(zips) > 1:
        print(f"  Multiple zips found for {dataset_name}:")
        for z in zips:
            size_mb = z.stat().st_size / (1024 * 1024)
            print(f"    {z.name} ({size_mb:.1f} MB)")
        print(f"  Using: {zips[0].name}")
    else:
        size_mb = zips[0].stat().st_size / (1024 * 1024)
        print(f"  {dataset_name}: {zips[0].name} ({size_mb:.1f} MB)")

    return zips[0]


def unzip_to_dir(
    zip_path: Path, output_dir: Path, overwrite: bool = False
) -> None:
    """
    Extract a zip file to the given directory.

    Args:
        zip_path: Path to the zip file.
        output_dir: Destination directory.
        overwrite: If True, remove output_dir before extracting.
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            print(f"  Already extracted: {output_dir} (skipping)")
            return
        print(f"  Removing existing: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {zip_path.name} -> {output_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    file_count = len(list(output_dir.rglob("*")))
    print(f"  Done. {file_count} files/dirs extracted.")


def show_tree(base_dir: Path, max_items: int = 40) -> None:
    """
    Print a partial listing of files and directories.

    Args:
        base_dir: Root directory to list.
        max_items: Maximum number of entries to print.
    """
    if not base_dir.exists():
        print(f"  (not found: {base_dir})")
        return

    items = sorted(base_dir.rglob("*"))
    for i, item in enumerate(items):
        if i >= max_items:
            remaining = len(items) - max_items
            print(f"  ... and {remaining} more")
            break
        rel = item.relative_to(base_dir)
        prefix = "d" if item.is_dir() else "f"
        print(f"  {prefix} {rel}")


def inspect_open_images_coco(local_open_images_dir: Path) -> dict:
    """
    Inspect the Open Images COCO export.

    Finds the first JSON annotation file, reads it, and prints
    a summary of images, annotations, and categories.

    Args:
        local_open_images_dir: Path to the extracted Open Images data.

    Returns:
        Dict with json_path, images_count, annotations_count, categories.
    """
    jsons = sorted(local_open_images_dir.rglob("*.json"))

    if not jsons:
        print("  WARNING: No JSON files found in Open Images directory.")
        return {
            "json_path": None,
            "images_count": 0,
            "annotations_count": 0,
            "categories": [],
        }

    json_path = jsons[0]
    print(f"  JSON path: {json_path}")

    with open(json_path, "r") as f:
        coco = json.load(f)

    images_count = len(coco.get("images", []))
    annotations_count = len(coco.get("annotations", []))
    categories = coco.get("categories", [])

    print(f"  Images:      {images_count}")
    print(f"  Annotations: {annotations_count}")
    print(f"  Categories:  {categories}")

    return {
        "json_path": str(json_path),
        "images_count": images_count,
        "annotations_count": annotations_count,
        "categories": categories,
    }


def inspect_uec_food256(local_uec_food_dir: Path) -> dict:
    """
    Inspect the UEC FOOD-256 dataset structure.

    Looks for the UECFOOD256 folder, category.txt, and bb_info.txt files.

    Args:
        local_uec_food_dir: Path to the extracted UEC FOOD-256 data.

    Returns:
        Dict with uec_root, category_path, bb_info_count.
    """
    # The zip may extract into a UECFOOD256 subfolder
    uec_root = local_uec_food_dir / "UECFOOD256"
    if not uec_root.exists():
        # Try the directory itself
        uec_root = local_uec_food_dir
        # Search for UECFOOD256 anywhere inside
        candidates = list(local_uec_food_dir.rglob("UECFOOD256"))
        if candidates:
            uec_root = candidates[0]

    print(f"  UEC root: {uec_root}")

    # Check category.txt
    category_path = uec_root / "category.txt"
    category_exists = category_path.exists()
    print(f"  category.txt exists: {category_exists}")

    if category_exists:
        print("  First 10 lines of category.txt:")
        with open(category_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(f"    {line.rstrip()}")

    # Count bb_info.txt files
    bb_infos = sorted(uec_root.rglob("bb_info.txt"))
    print(f"  bb_info.txt files: {len(bb_infos)}")

    return {
        "uec_root": str(uec_root),
        "category_path": str(category_path) if category_exists else None,
        "bb_info_count": len(bb_infos),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Raw Dataset Setup from Google Drive")
    print("=" * 60)

    # --- Find zips ---
    print("\nLocating zip files on Drive...")
    oi_zip = find_zip(DRIVE_OPEN_IMAGES_DIR, "Open Images")
    uec_zip = find_zip(DRIVE_UEC_FOOD_DIR, "UEC FOOD-256")

    # --- Extract ---
    print("\nExtracting Open Images...")
    unzip_to_dir(oi_zip, LOCAL_OPEN_IMAGES_DIR)

    print("\nExtracting UEC FOOD-256...")
    unzip_to_dir(uec_zip, LOCAL_UEC_FOOD_DIR)

    # --- Show structure ---
    print("\n" + "=" * 60)
    print("Open Images — extracted structure")
    print("=" * 60)
    show_tree(LOCAL_OPEN_IMAGES_DIR)

    print("\n" + "=" * 60)
    print("UEC FOOD-256 — extracted structure")
    print("=" * 60)
    show_tree(LOCAL_UEC_FOOD_DIR)

    # --- Inspect ---
    print("\n" + "=" * 60)
    print("Open Images — COCO inspection")
    print("=" * 60)
    oi_info = inspect_open_images_coco(LOCAL_OPEN_IMAGES_DIR)

    print("\n" + "=" * 60)
    print("UEC FOOD-256 — structure inspection")
    print("=" * 60)
    uec_info = inspect_uec_food256(LOCAL_UEC_FOOD_DIR)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Open Images images:      {oi_info['images_count']}")
    print(f"  Open Images annotations: {oi_info['annotations_count']}")
    print(f"  Open Images categories:  {len(oi_info['categories'])}")
    print(f"  UEC FOOD-256 bb_info:    {uec_info['bb_info_count']}")
    print("\nSetup complete.")


if __name__ == "__main__":
    main()