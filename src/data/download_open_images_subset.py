"""
Download a subset of Open Images V7 using FiftyOne.

This script runs locally (Conda/Kiro environment) to acquire raw
images with bounding box annotations from Open Images V7. It downloads
a configurable number of samples per class, filters out detections
from unwanted classes, merges everything into a single combined
dataset, exports to portable COCO Detection format, and creates
a zip archive for manual upload to Google Drive.

This script does NOT:
- Convert annotations to YOLO format
- Split into train/val/test
- Run any training or evaluation
- Use DVC or MLflow

Usage:
    python -m src.data.download_open_images_subset
"""

import re
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPEN_IMAGES_CLASSES = [
    "Bottle",
    "Coffee cup",
    "Wine glass",
    "Bowl",
    "Plate",
    "Mixing bowl",
    "Spoon",
    "Fork",
    "Knife",
    "Kitchen knife",
]
SPLIT = "train"
MAX_SAMPLES_PER_CLASS = 2000
MAX_WORKERS = 4
DATASET_NAME = "open_images_v7_table_objects_v1"

DOWNLOAD_DIR = Path("local_data/raw_datasets/open_images_subset_download")
EXPORT_DIR = Path("local_data/raw_datasets/open_images_table_objects_v1_coco")
CREATE_ZIP = True
ZIP_OUTPUT_PATH = Path("local_data/raw_datasets/open_images_table_objects_v1_coco.zip")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_name(class_name: str) -> str:
    """
    Convert a class name to a filesystem/dataset-safe identifier.

    Examples:
        "Coffee cup"  -> "coffee_cup"
        "Wine glass"  -> "wine_glass"
        "Bottle"      -> "bottle"
    """
    name = class_name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def filter_detections_by_requested_classes(
    dataset: fo.Dataset,
    requested_classes: list[str],
) -> dict:
    """
    Filter detections in-place, keeping only requested classes.

    Open Images images often contain annotations for multiple classes.
    This function removes detections whose label is not in the
    requested set, so the exported dataset only contains the classes
    we actually want.

    Comparison is case-sensitive to match Open Images label names
    exactly (e.g. "Bottle", not "bottle"). No label remapping is done.

    Args:
        dataset: FiftyOne dataset to filter.
        requested_classes: List of class names to keep.

    Returns:
        Dict with keys: detections_before, detections_after,
        samples_without_detections.
    """
    allowed = set(requested_classes)

    detections_before = 0
    detections_after = 0
    samples_without_detections = 0

    for sample in dataset:
        if not sample.ground_truth or not sample.ground_truth.detections:
            samples_without_detections += 1
            continue

        original = sample.ground_truth.detections
        detections_before += len(original)

        filtered = [det for det in original if det.label in allowed]
        detections_after += len(filtered)

        sample.ground_truth.detections = filtered
        sample.save()

        if len(filtered) == 0:
            samples_without_detections += 1

    return {
        "detections_before": detections_before,
        "detections_after": detections_after,
        "samples_without_detections": samples_without_detections,
    }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def download_dataset() -> fo.Dataset:
    """Download per-class subsets, filter, and merge into a combined dataset."""
    print("=" * 60)
    print("Open Images V7 — Per-Class Subset Download")
    print("=" * 60)
    print(f"  Classes:              {OPEN_IMAGES_CLASSES}")
    print(f"  Split:                {SPLIT}")
    print(f"  Max samples/class:    {MAX_SAMPLES_PER_CLASS}")
    print(f"  Max workers:          {MAX_WORKERS}")
    print(f"  Download dir:         {DOWNLOAD_DIR}")
    print(f"  Export dir:           {EXPORT_DIR}")
    print(f"  Zip path:             {ZIP_OUTPUT_PATH}")
    print("=" * 60)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Remove existing combined dataset to start fresh
    if fo.dataset_exists(DATASET_NAME):
        print(f"\nRemoving existing combined dataset '{DATASET_NAME}'...")
        fo.delete_dataset(DATASET_NAME)

    combined = fo.Dataset(name=DATASET_NAME, persistent=True)

    for class_name in OPEN_IMAGES_CLASSES:
        slug = safe_name(class_name)
        temp_name = f"_temp_oi7_{slug}"

        print(f"\n--- Downloading class: {class_name} (slug: {slug}) ---")

        # Clean up temp dataset if it exists
        if fo.dataset_exists(temp_name):
            fo.delete_dataset(temp_name)

        temp_dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=SPLIT,
            label_types=["detections"],
            classes=[class_name],
            max_samples=MAX_SAMPLES_PER_CLASS,
            dataset_name=temp_name,
            max_workers=MAX_WORKERS,
        )

        # Filter detections to keep only the requested class
        stats = filter_detections_by_requested_classes(temp_dataset, [class_name])

        print(f"  Samples downloaded:              {len(temp_dataset)}")
        print(f"  Detections before filtering:     {stats['detections_before']}")
        print(f"  Detections after filtering:       {stats['detections_after']}")
        print(f"  Samples without detections:      {stats['samples_without_detections']}")

        if stats["detections_after"] == 0:
            print(f"  WARNING: Class '{class_name}' produced 0 valid detections after filtering.")
            print(f"           Check that '{class_name}' is a valid Open Images class name.")

        # Add only samples that still have detections
        added = 0
        for sample in temp_dataset:
            if (
                sample.ground_truth
                and sample.ground_truth.detections
                and len(sample.ground_truth.detections) > 0
            ):
                combined.add_sample(sample)
                added += 1

        print(f"  Samples added to combined:       {added}")

        # Clean up temp dataset (keeps files on disk)
        fo.delete_dataset(temp_name)

    print(f"\nCombined dataset '{DATASET_NAME}': {len(combined)} samples")
    return combined


def inspect_dataset(dataset: fo.Dataset) -> None:
    """Inspect the combined dataset and verify detections exist."""
    print("\n" + "=" * 60)
    print("Dataset Inspection")
    print("=" * 60)

    num_samples = len(dataset)
    print(f"  Total samples:  {num_samples}")

    # Schema
    print("\n  Dataset schema:")
    for field_name, field in dataset.get_field_schema().items():
        print(f"    {field_name}: {field}")

    # Count detections
    total_detections = 0
    samples_without_detections = 0
    for sample in dataset:
        if sample.ground_truth and sample.ground_truth.detections:
            total_detections += len(sample.ground_truth.detections)
        else:
            samples_without_detections += 1

    print(f"\n  Total detections:            {total_detections}")
    print(f"  Samples without detections:  {samples_without_detections}")

    # Example samples
    print("\n  Example samples:")
    for i, sample in enumerate(dataset):
        if i >= 5:
            break
        print(f"    [{i}] {sample.filepath}")
        if sample.ground_truth and sample.ground_truth.detections:
            for j, det in enumerate(sample.ground_truth.detections[:3]):
                print(f"         det[{j}]: label={det.label}, bbox={det.bounding_box}")

    # Validate
    if total_detections == 0:
        raise RuntimeError(
            "No detections found in the combined dataset. "
            "Check OPEN_IMAGES_CLASSES and label_types configuration."
        )


def export_to_coco(dataset: fo.Dataset) -> None:
    """Export the combined dataset to portable COCO Detection format."""
    print("\n" + "=" * 60)
    print("Exporting to COCO Detection format")
    print("=" * 60)
    print(f"  Export dir: {EXPORT_DIR}")

    if EXPORT_DIR.exists():
        print(f"  Removing existing export dir: {EXPORT_DIR}")
        shutil.rmtree(EXPORT_DIR)

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    dataset.export(
        export_dir=str(EXPORT_DIR),
        dataset_type=fo.types.COCODetectionDataset,
        label_field="ground_truth",
        export_media=True,
    )

    print(f"  Export complete: {EXPORT_DIR}")


def create_zip_archive() -> None:
    """Create a zip archive of the exported COCO dataset."""
    if not CREATE_ZIP:
        return

    print("\n" + "=" * 60)
    print("Creating zip archive")
    print("=" * 60)

    if not EXPORT_DIR.exists():
        raise RuntimeError(
            f"Export directory does not exist: {EXPORT_DIR}. "
            "Run export_to_coco() before creating the zip."
        )

    if ZIP_OUTPUT_PATH.exists():
        print(f"  Removing existing zip: {ZIP_OUTPUT_PATH}")
        ZIP_OUTPUT_PATH.unlink()

    ZIP_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    archive_base = str(ZIP_OUTPUT_PATH.with_suffix(""))
    shutil.make_archive(archive_base, "zip", root_dir=str(EXPORT_DIR))

    zip_size_mb = ZIP_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Zip created: {ZIP_OUTPUT_PATH} ({zip_size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset = download_dataset()
    inspect_dataset(dataset)
    export_to_coco(dataset)
    create_zip_archive()
    print("\nDone.")


if __name__ == "__main__":
    main()