"""
Generate sanity-check images with bounding boxes drawn on raw datasets.

Produces PNG files to visually verify that bounding box annotations
are being read correctly from both Open Images (COCO format) and
UEC FOOD-256 before any conversion to YOLO.

This script does NOT:
- Convert annotations to YOLO format
- Split into train/val/test
- Train any model
- Use DVC or MLflow

Usage::

    python -m src.data.raw_setup.visualize_raw_bboxes
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving PNGs

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.paths import (
    get_open_images_dataset_original_dir,
    get_outputs_dir,
    get_uec_food_dataset_extract_dir,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOCAL_OPEN_IMAGES_DIR = get_open_images_dataset_original_dir()
LOCAL_UEC_FOOD_DIR = get_uec_food_dataset_extract_dir()
OUTPUT_DIR = get_outputs_dir() / "bbox_checks"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def show_image_with_boxes(
    image_path,
    boxes,
    labels,
    output_path,
    title=None,
):
    """
    Draw bounding boxes on an image and save to disk.

    Args:
        image_path: Path to the source image.
        boxes: List of [x1, y1, x2, y2] bounding boxes.
        labels: List of label strings, one per box.
        output_path: Where to save the annotated PNG.
        title: Optional title for the figure.
    """
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    colors = plt.cm.Set2.colors
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 4, label,
            fontsize=9, color="white",
            bbox=dict(facecolor=color, alpha=0.7, pad=1),
        )

    if title:
        ax.set_title(title, fontsize=12)
    ax.axis("off")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def resolve_open_images_path(base_dir: Path, file_name: str) -> Path:
    """
    Resolve an image file path within the Open Images export.

    Tries several common locations since the COCO export may store
    images in different subdirectories.

    Args:
        base_dir: Root of the extracted Open Images data.
        file_name: File name from the COCO JSON images list.

    Returns:
        Resolved Path to the image file.

    Raises:
        FileNotFoundError: If the image cannot be located.
    """
    name_only = Path(file_name).name
    candidates = [
        base_dir / file_name,
        base_dir / "data" / file_name,
        base_dir / name_only,
        base_dir / "data" / name_only,
    ]

    for c in candidates:
        if c.exists():
            return c

    # Fallback: recursive search
    matches = list(base_dir.rglob(name_only))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find image '{file_name}' under {base_dir}"
    )


# ---------------------------------------------------------------------------
# Open Images check
# ---------------------------------------------------------------------------

def create_open_images_bbox_check() -> None:
    """Generate a sample image with COCO bounding boxes from Open Images."""
    print("\n--- Open Images bbox check ---")

    jsons = sorted(LOCAL_OPEN_IMAGES_DIR.rglob("*.json"))
    if not jsons:
        print("  WARNING: No JSON files found. Skipping.")
        return

    json_path = jsons[0]
    print(f"  Using: {json_path}")

    with open(json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    annotations = coco.get("annotations", [])
    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

    if not annotations:
        print("  WARNING: No annotations found. Skipping.")
        return

    # Find an image that has annotations
    target_image_id = annotations[0]["image_id"]
    target_image = images[target_image_id]
    file_name = target_image["file_name"]

    image_path = resolve_open_images_path(LOCAL_OPEN_IMAGES_DIR, file_name)
    print(f"  Image: {image_path}")

    # Collect all annotations for this image
    img_anns = [a for a in annotations if a["image_id"] == target_image_id]

    boxes = []
    labels = []
    for ann in img_anns:
        # COCO bbox: [x, y, width, height] -> [x1, y1, x2, y2]
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        cat_name = categories.get(ann["category_id"], f"id={ann['category_id']}")
        labels.append(cat_name)

    print(f"  Annotations: {len(boxes)}")

    output_path = OUTPUT_DIR / "open_images_sample.png"
    show_image_with_boxes(
        image_path, boxes, labels, output_path,
        title=f"Open Images — {Path(file_name).name} ({len(boxes)} boxes)",
    )


# ---------------------------------------------------------------------------
# UEC FOOD-256 check
# ---------------------------------------------------------------------------

def create_uec_food_bbox_check() -> None:
    """Generate a sample image with bounding boxes from UEC FOOD-256."""
    print("\n--- UEC FOOD-256 bbox check ---")

    # Locate UECFOOD256 root
    uec_root = LOCAL_UEC_FOOD_DIR / "UECFOOD256"
    if not uec_root.exists():
        candidates = list(LOCAL_UEC_FOOD_DIR.rglob("UECFOOD256"))
        if candidates:
            uec_root = candidates[0]
        else:
            uec_root = LOCAL_UEC_FOOD_DIR

    print(f"  UEC root: {uec_root}")

    # Read category.txt
    category_path = uec_root / "category.txt"
    categories = {}
    if category_path.exists():
        with open(category_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    try:
                        categories[parts[0]] = parts[1]
                    except ValueError:
                        pass

    # Use category 1 by default
    category_id = "1"
    category_dir = uec_root / category_id
    if not category_dir.exists():
        # Try first numbered directory
        numbered = sorted(
            [d for d in uec_root.iterdir() if d.is_dir() and d.name.isdigit()]
        )
        if numbered:
            category_dir = numbered[0]
            category_id = category_dir.name
        else:
            print("  WARNING: No numbered category directories found. Skipping.")
            return

    bb_info_path = category_dir / "bb_info.txt"
    if not bb_info_path.exists():
        print(f"  WARNING: bb_info.txt not found in {category_dir}. Skipping.")
        return

    print(f"  Category: {category_id} ({categories.get(category_id, 'unknown')})")
    print(f"  bb_info:  {bb_info_path}")

    # Read first valid bbox line
    # Format: img x1 y1 x2 y2
    boxes = []
    target_img_id = None
    with open(bb_info_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                img_id = parts[0]
                x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                if target_img_id is None:
                    target_img_id = img_id
                if img_id == target_img_id:
                    boxes.append([x1, y1, x2, y2])
            except ValueError:
                continue

    if not boxes or target_img_id is None:
        print("  WARNING: No valid bboxes found. Skipping.")
        return

    # Resolve image path (try common extensions)
    image_path = None
    for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
        candidate = category_dir / f"{target_img_id}{ext}"
        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        print(f"  WARNING: Image {target_img_id}.* not found in {category_dir}. Skipping.")
        return

    print(f"  Image: {image_path}")
    print(f"  Boxes: {len(boxes)}")

    cat_label = categories.get(category_id, f"cat_{category_id}")
    labels = [cat_label] * len(boxes)

    output_path = OUTPUT_DIR / "uec_food_sample.png"
    show_image_with_boxes(
        image_path, boxes, labels, output_path,
        title=f"UEC FOOD-256 — cat {category_id}: {cat_label} ({len(boxes)} boxes)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Raw Bounding Box Sanity Check")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}")

    create_open_images_bbox_check()
    create_uec_food_bbox_check()

    print("\n" + "=" * 60)
    print("Generated images:")
    print(f"  {OUTPUT_DIR / 'open_images_sample.png'}")
    print(f"  {OUTPUT_DIR / 'uec_food_sample.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()