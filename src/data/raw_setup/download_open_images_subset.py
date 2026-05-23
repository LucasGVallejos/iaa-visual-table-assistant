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

    The full ``OPEN_IMAGES_CLASSES`` set should be passed here regardless
    of which class triggered the download, so co-occurring target classes
    in the same image are preserved (multi-class images are then merged by
    ``dedupe_and_merge_samples`` after the per-class loop ends).

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


def _detection_dedup_key(det) -> tuple:
    """Return a stable key for two detections that should be considered identical.

    Two detections collide when label and bbox match (bbox rounded to 6
    decimals to absorb float noise). Per-class downloads can re-download the
    same image and therefore the same annotations; this key catches that.
    """
    bbox = det.bounding_box or [0.0, 0.0, 0.0, 0.0]
    return (det.label, *(round(float(v), 6) for v in bbox))


def dedupe_and_merge_samples(dataset: fo.Dataset) -> dict:
    """Collapse multiple FiftyOne samples that point at the same image file.

    The per-class download loop in :func:`download_dataset` will visit the
    same Open Images file several times when it appears under more than one
    target class. Without dedupe each iteration would add a separate sample
    with only that iteration's class, so a single image with bottle + cup
    + plate would become three single-class samples. We avoid that here by:

    1. Grouping samples by image basename (Open Images filenames are
       content hashes, so the basename is a reliable cross-iteration key).
    2. Merging the detections of every sample in a group and dropping
       duplicates with :func:`_detection_dedup_key`.
    3. Keeping the first sample of each group with the merged detection
       list and deleting the rest.

    Returns counters useful for logging.
    """
    samples_before = len(dataset)
    detections_before = 0

    # Group sample IDs by basename. Walking the dataset twice (once to
    # group, once to mutate) keeps things simple and avoids modifying
    # while iterating.
    groups: dict[str, list[str]] = {}
    for sample in dataset:
        detections_before += (
            len(sample.ground_truth.detections)
            if sample.ground_truth and sample.ground_truth.detections
            else 0
        )
        basename = Path(sample.filepath).name
        groups.setdefault(basename, []).append(sample.id)

    duplicate_groups = sum(1 for ids in groups.values() if len(ids) > 1)
    samples_to_delete: list[str] = []

    for ids in groups.values():
        if len(ids) == 1:
            continue

        merged: dict[tuple, object] = {}
        keeper_id = ids[0]
        for sample_id in ids:
            sample = dataset[sample_id]
            if not sample.ground_truth or not sample.ground_truth.detections:
                continue
            for det in sample.ground_truth.detections:
                merged.setdefault(_detection_dedup_key(det), det)

        keeper = dataset[keeper_id]
        keeper.ground_truth.detections = list(merged.values())
        keeper.save()
        samples_to_delete.extend(ids[1:])

    if samples_to_delete:
        dataset.delete_samples(samples_to_delete)

    samples_after = len(dataset)
    detections_after = 0
    for sample in dataset:
        if sample.ground_truth and sample.ground_truth.detections:
            detections_after += len(sample.ground_truth.detections)

    return {
        "samples_before": samples_before,
        "samples_after": samples_after,
        "samples_removed": samples_before - samples_after,
        "duplicate_groups": duplicate_groups,
        "detections_before": detections_before,
        "detections_after": detections_after,
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

        # Filter detections to keep every target class, not just the one
        # that triggered this iteration. Co-occurring target classes are
        # then merged across iterations by ``dedupe_and_merge_samples``.
        stats = filter_detections_by_requested_classes(
            temp_dataset, OPEN_IMAGES_CLASSES
        )

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

    print("\nDeduplicating combined dataset...")
    dedupe_stats = dedupe_and_merge_samples(combined)
    print(f"  samples before dedupe:     {dedupe_stats['samples_before']}")
    print(f"  unique images:             {dedupe_stats['samples_after']}")
    print(f"  duplicate groups merged:   {dedupe_stats['duplicate_groups']}")
    print(f"  samples removed:           {dedupe_stats['samples_removed']}")
    print(f"  detections before dedupe:  {dedupe_stats['detections_before']}")
    print(f"  detections after dedupe:   {dedupe_stats['detections_after']}")

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

    # Walk the dataset once and gather: detection counts, distinct target
    # classes per image, and per-image detection counts (for distribution
    # stats). Doing it in a single pass avoids re-iterating thousands of
    # samples just to compute related metrics.
    target_classes = set(OPEN_IMAGES_CLASSES)
    total_detections = 0
    samples_without_detections = 0
    detections_per_image: list[int] = []
    distinct_classes_per_image: list[int] = []

    for sample in dataset:
        if sample.ground_truth and sample.ground_truth.detections:
            dets = sample.ground_truth.detections
            count = len(dets)
            total_detections += count
            detections_per_image.append(count)
            distinct_classes_per_image.append(
                len({det.label for det in dets if det.label in target_classes})
            )
        else:
            samples_without_detections += 1

    print(f"\n  Total detections:            {total_detections}")
    print(f"  Samples without detections:  {samples_without_detections}")

    # Distribution of distinct target classes per image. Sanity-checks the
    # dedupe step: after the per-class loop merges duplicates, multi-class
    # images should appear in numbers proportional to natural co-occurrence
    # in Open Images. If almost everything still lands in the "1 class"
    # bucket, dedupe didn't kick in or the upstream filter is too strict.
    images_with_n_classes: dict[int, int] = {}
    for n in distinct_classes_per_image:
        bucket = n if n < 3 else 3
        images_with_n_classes[bucket] = images_with_n_classes.get(bucket, 0) + 1

    print("\n  Images by distinct target classes:")
    print(f"    1 class:    {images_with_n_classes.get(1, 0)}")
    print(f"    2 classes:  {images_with_n_classes.get(2, 0)}")
    print(f"    3+ classes: {images_with_n_classes.get(3, 0)}")

    # Detections per image distribution. Useful to detect skew (e.g. a
    # single image holding hundreds of bottles inflating class counts).
    if detections_per_image:
        sorted_counts = sorted(detections_per_image)
        n = len(sorted_counts)
        p50 = sorted_counts[n // 2]
        mean = sum(sorted_counts) / n
        print("\n  Detections per image:")
        print(
            f"    min={sorted_counts[0]}  p50={p50}  "
            f"mean={mean:.2f}  max={sorted_counts[-1]}"
        )

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