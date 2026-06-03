"""
Phase-3 sanity step for notebook 0.5: extract the raw Open Images COCO export
that notebook 00 uploaded to Drive (the same input notebook 01 uses — NO DVC),
inspect it, and render a few samples with their current boxes.

This is the first step of the auto-labeling workstream. It does not run any
detector and does not modify the COCO export; it only verifies that the raw
Open Images data reads and renders correctly before any pseudo-labeling.

Usage (from Colab, after mounting Drive)::

    !python -m src.data.auto_label.prepare_open_images_input --samples 3

Usage (local, against the already-extracted raw dir)::

    python -m src.data.auto_label.prepare_open_images_input --skip-extract --samples 3
"""

import argparse
import json
import random
from pathlib import Path

from src.data.raw_setup.setup_colab_raw_datasets import (
    DRIVE_OPEN_IMAGES_DIR,
    LOCAL_OPEN_IMAGES_DIR,
    find_zip,
    inspect_open_images_coco,
    unzip_to_dir,
)
from src.data.raw_setup.visualize_raw_bboxes import show_image_with_boxes
from src.utils.paths import get_outputs_dir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHASE3_RAW_OUTPUT_DIR = get_outputs_dir() / "auto_label_checks" / "phase3_raw"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def extract_open_images(overwrite: bool = False) -> Path:
    """
    Extract the raw Open Images zip from Drive into the local raw dir.

    Reuses the same Drive-extraction helpers notebook 01 relies on. This only
    works in Colab where Drive is mounted at ``/content/drive``; that is the
    expected runtime for this step.

    Args:
        overwrite: If True, wipe the local target dir before extracting.

    Returns:
        Path to the local extracted Open Images directory.
    """
    zip_path = find_zip(DRIVE_OPEN_IMAGES_DIR, "Open Images")
    unzip_to_dir(zip_path, LOCAL_OPEN_IMAGES_DIR, overwrite)
    return LOCAL_OPEN_IMAGES_DIR


def find_labels_json(coco_dir: Path) -> Path:
    """
    Locate the COCO annotation JSON inside the extracted Open Images dir.

    Args:
        coco_dir: Root of the extracted Open Images data.

    Returns:
        Path to the first ``*.json`` found (sorted), which is the COCO export.

    Raises:
        FileNotFoundError: If no JSON file exists under ``coco_dir``.
    """
    jsons = sorted(coco_dir.rglob("*.json"))
    if not jsons:
        raise FileNotFoundError(
            f"No COCO annotation JSON found under {coco_dir}. "
            "Did the Open Images zip extract correctly? Expected a "
            "labels.json alongside a data/ image directory."
        )
    return jsons[0]


def render_raw_samples(coco_dir: Path, samples: int, seed: int) -> list[Path]:
    """
    Render a few raw Open Images samples with their current COCO boxes.

    Loads the COCO JSON, picks ``samples`` random images deterministically,
    and draws each image's source bounding boxes (in pixels) with their source
    category names. Images whose file is missing are skipped with a WARN.

    Args:
        coco_dir: Root of the extracted Open Images data.
        samples: How many images to render (capped at the number available).
        seed: Random seed for reproducible image selection.

    Returns:
        List of written PNG paths.
    """
    json_path = find_labels_json(coco_dir)
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}

    # Index annotations by image id so we can pull a single image's boxes fast.
    anns_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    rng = random.Random(seed)
    n = min(samples, len(images))
    chosen = rng.sample(images, n) if n > 0 else []

    PHASE3_RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for image in chosen:
        file_name = image["file_name"]
        image_path = coco_dir / "data" / file_name

        if not image_path.exists():
            print(f"  WARN: image file missing, skipping: {image_path}")
            continue

        img_anns = anns_by_image.get(image["id"], [])
        boxes = []
        labels = []
        for ann in img_anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(cat_id_to_name.get(ann["category_id"], f"id={ann['category_id']}"))

        stem = Path(file_name).stem
        output_path = PHASE3_RAW_OUTPUT_DIR / f"{stem}.png"
        title = f"{file_name} — {len(boxes)} boxes (raw Open Images)"
        show_image_with_boxes(image_path, boxes, labels, output_path, title=title)
        written.append(output_path)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for the phase-3 raw input sanity step."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract and inspect the raw Open Images COCO export and render a "
            "few samples with their current boxes (phase-3 sanity step)."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample images to render (>= 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Wipe the local raw dir before extracting from Drive.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help=(
            "Skip the Drive extraction and use the already-extracted local "
            "dir (needed for local testing outside Colab)."
        ),
    )
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    return args


def main(argv=None) -> None:
    """Extract (unless skipped), inspect, and render raw Open Images samples."""
    args = parse_args(argv)

    if not args.skip_extract:
        extract_open_images(args.overwrite)
    else:
        print(f"Skipping extraction; using local dir: {LOCAL_OPEN_IMAGES_DIR}")

    print("\n" + "=" * 60)
    print("Open Images — COCO inspection")
    print("=" * 60)
    inspect_open_images_coco(LOCAL_OPEN_IMAGES_DIR)

    print("\n" + "=" * 60)
    print("Open Images — raw sample renders")
    print("=" * 60)
    written = render_raw_samples(LOCAL_OPEN_IMAGES_DIR, args.samples, args.seed)

    print("\nWritten PNGs:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
