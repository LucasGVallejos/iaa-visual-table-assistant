"""
Phase-6 verification for notebook 0.5: render before/after panels so the
auto-labeled boxes can be eyeballed for precision before the full run.

The enrichment pass (:mod:`src.data.auto_label.auto_label_open_images`) writes a
v2 ``labels.json`` next to the read-only v1 export. This script loads both and,
for each image that gained at least one ``source == "auto_label"`` annotation,
produces ONE PNG with two side-by-side panels: the LEFT panel shows only the v1
boxes (faint, with their COCO category names) and the RIGHT panel shows the same
v1 boxes plus the injected boxes drawn in red with ``+<name> <score>`` labels.
Use it to tune ``--conf`` / ``--iou-dedup`` on a small sample before enriching
the full dataset.

This step reads only; it never writes to either labels.json. Images are read
from the v1 ``data/`` directory (v2 reuses them by ``file_name``).

Usage (local, against the already-extracted raw dir + the v2 output)::

    python -m src.data.auto_label.verify_autolabel --samples 8
    python -m src.data.auto_label.verify_autolabel --samples 4 --seed 7
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; we only save PNGs

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

from src.data.auto_label.auto_label_open_images import default_out_path
from src.data.auto_label.coco_io import category_id_to_name, load_coco
from src.data.auto_label.prepare_open_images_input import find_labels_json
from src.utils.paths import (
    get_open_images_dataset_original_dir,
    get_outputs_dir,
)

PHASE6_OUTPUT_DIR = get_outputs_dir() / "auto_label_checks" / "phase6_verify"

EXISTING_EDGE = "lightgray"
ADDED_EDGE = "red"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def index_annotations(coco: dict) -> dict[int, list[dict]]:
    """Group a COCO doc's annotations by ``image_id``."""
    by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        by_image.setdefault(int(ann["image_id"]), []).append(ann)
    return by_image


def draw_boxes(
    ax,
    annotations: list[dict],
    cat_id_to_name: dict[int, str],
    edge_color: str,
    label_prefix: str = "",
    with_score: bool = False,
) -> None:
    """Draw a set of COCO ``[x, y, w, h]`` boxes onto a matplotlib axis."""
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        rect = patches.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2,
            edgecolor=edge_color,
            facecolor="none",
        )
        ax.add_patch(rect)

        name = cat_id_to_name.get(int(ann["category_id"]), f"id={ann['category_id']}")
        if with_score and "score" in ann:
            text = f"{label_prefix}{name} {ann['score']:.2f}"
        else:
            text = f"{label_prefix}{name}"
        ax.text(
            x,
            max(0, y - 4),
            text,
            fontsize=8,
            color="white",
            bbox=dict(facecolor=edge_color, alpha=0.75, pad=1),
        )


def render_before_after(
    image_path: Path,
    v1_anns: list[dict],
    added_anns: list[dict],
    cat_id_to_name: dict[int, str],
    output_path: Path,
    title: str,
) -> None:
    """Render one two-panel before/after PNG for a single image."""
    img = Image.open(image_path).convert("RGB")

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(20, 8))

    ax_before.imshow(img)
    ax_before.set_title("before (v1)", fontsize=11)
    draw_boxes(ax_before, v1_anns, cat_id_to_name, EXISTING_EDGE)
    ax_before.axis("off")

    ax_after.imshow(img)
    ax_after.set_title("after (v2: + auto-labeled)", fontsize=11)
    draw_boxes(ax_after, v1_anns, cat_id_to_name, EXISTING_EDGE)
    draw_boxes(
        ax_after,
        added_anns,
        cat_id_to_name,
        ADDED_EDGE,
        label_prefix="+",
        with_score=True,
    )
    ax_after.axis("off")

    fig.suptitle(title, fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for the phase-6 verification renders."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.auto_label.verify_autolabel",
        description=(
            "Render before/after panels for images the enrichment pass added "
            "auto-labeled boxes to, so precision can be eyeballed."
        ),
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=get_open_images_dataset_original_dir(),
        help="Root of the v1 Open Images export (labels.json + data/ images).",
    )
    parser.add_argument(
        "--enriched",
        type=Path,
        default=default_out_path(),
        help="Path to the enriched v2 labels.json produced by the enrichment pass.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="Number of images-with-additions to render (>= 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    return args


def main(argv=None) -> None:
    """Load v1 + v2, sample images with additions, and render before/after PNGs."""
    args = parse_args(argv)

    src_dir: Path = args.src_dir
    enriched_path: Path = args.enriched

    if not enriched_path.exists():
        raise FileNotFoundError(
            f"Enriched v2 labels not found: {enriched_path}. Run the enrichment "
            "pass first, e.g.:\n"
            "    python -m src.data.auto_label.auto_label_open_images --limit 50"
        )

    v1_json = find_labels_json(src_dir)
    v1 = load_coco(v1_json)
    v2 = load_coco(enriched_path)

    cat_id_to_name = category_id_to_name(v2)
    images_by_id = {int(img["id"]): img for img in v2.get("images", [])}

    v1_by_image = index_annotations(v1)
    v2_by_image = index_annotations(v2)

    # Image ids that gained at least one auto-labeled annotation.
    images_with_additions = [
        image_id
        for image_id, anns in v2_by_image.items()
        if any(ann.get("source") == "auto_label" for ann in anns)
    ]

    if not images_with_additions:
        print(
            "NOTE: the enriched v2 labels contain no annotations tagged "
            "source='auto_label'. Nothing to render. The enrichment pass added "
            "no boxes (try a lower --conf or a larger --limit)."
        )
        return

    rng = random.Random(args.seed)
    n = min(args.samples, len(images_with_additions))
    chosen = rng.sample(images_with_additions, n)

    data_dir = src_dir / "data"
    written: list[Path] = []

    for image_id in chosen:
        image = images_by_id.get(image_id)
        if image is None:
            continue
        file_name = image["file_name"]
        image_path = data_dir / file_name
        if not image_path.exists():
            print(f"  WARN: image file missing, skipping: {image_path}")
            continue

        added_anns = [
            ann for ann in v2_by_image.get(image_id, []) if ann.get("source") == "auto_label"
        ]
        v1_anns = v1_by_image.get(image_id, [])

        stem = Path(file_name).stem
        output_path = PHASE6_OUTPUT_DIR / f"{stem}.png"
        title = f"{file_name} — {len(v1_anns)} existing + {len(added_anns)} added"
        render_before_after(image_path, v1_anns, added_anns, cat_id_to_name, output_path, title)
        written.append(output_path)

    print("\nWritten PNGs:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
