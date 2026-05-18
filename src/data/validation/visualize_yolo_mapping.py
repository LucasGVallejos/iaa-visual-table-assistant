"""
Visual sanity check for the per-source YOLO staging conversions.

Run with::

    python -m src.data.validation.visualize_yolo_mapping
    python -m src.data.validation.visualize_yolo_mapping --seed 7
    python -m src.data.validation.visualize_yolo_mapping --samples-per-class 10
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

from src.data.common.convert_to_yolo import load_classes_config
from src.utils.paths import (
    get_open_images_staging_dir,
    get_outputs_dir,
    get_uec_staging_dir,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_SAMPLES_PER_CLASS = 5
DEFAULT_RANDOM_SEED = 42
FALLBACK_COLOR = (0.5, 0.5, 0.5)  # gray, only used for class IDs not in the config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_class_lookups(
    classes: list[dict],
) -> tuple[dict[int, str], dict[int, tuple[float, float, float]]]:
    """Return ``(id -> name, id -> rgb_float)`` from classes.yaml.

    Colors come straight from the config in 0-255 RGB and are converted to
    matplotlib's 0-1 float tuples here.
    """
    id_to_name: dict[int, str] = {}
    id_to_color: dict[int, tuple[float, float, float]] = {}
    for entry in classes:
        class_id = int(entry["id"])
        id_to_name[class_id] = entry["name"]
        rgb = entry.get("color")
        if rgb and len(rgb) == 3:
            id_to_color[class_id] = tuple(channel / 255.0 for channel in rgb)
        else:
            id_to_color[class_id] = FALLBACK_COLOR
    return id_to_name, id_to_color


def yolo_to_xyxy(
    bbox: list[float], img_width: int, img_height: int
) -> tuple[float, float, float, float]:
    """Convert YOLO normalized ``[cx, cy, w, h]`` to absolute ``[x1, y1, x2, y2]``."""
    cx, cy, w, h = bbox
    x1 = (cx - w / 2.0) * img_width
    y1 = (cy - h / 2.0) * img_height
    x2 = (cx + w / 2.0) * img_width
    y2 = (cy + h / 2.0) * img_height
    return x1, y1, x2, y2


def parse_yolo_label_file(label_path: Path) -> list[tuple[int, list[float]]]:
    """Read a YOLO label file as ``[(class_id, [cx, cy, w, h]), ...]``.
    Invalid lines are skipped; returns an empty list for a missing file.
    """
    annotations: list[tuple[int, list[float]]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                bbox = [float(value) for value in parts[1:]]
            except ValueError:
                continue
            annotations.append((class_id, bbox))
    return annotations


def list_staging_images(staging_dir: Path) -> list[Path]:
    """Return image paths in ``<staging>/images/`` sorted for determinism."""
    images_dir = staging_dir / "images"
    if not images_dir.is_dir():
        return []
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def label_path_for(staging_dir: Path, image_path: Path) -> Path:
    return staging_dir / "labels" / f"{image_path.stem}.txt"


# ---------------------------------------------------------------------------
# Per-class indexing across staging dirs
# ---------------------------------------------------------------------------
def index_images_by_class(
    staging_dirs: list[tuple[str, Path]],
) -> dict[int, list[tuple[str, Path]]]:
    """Build ``{class_id: [(dataset, image_path), ...]}`` from every staged label.

    An image is recorded once per distinct class it contains, so the same
    image can appear under multiple class buckets but never twice in the
    same bucket.
    """
    index: dict[int, list[tuple[str, Path]]] = {}

    for dataset, staging_dir in staging_dirs:
        for image_path in list_staging_images(staging_dir):
            label_path = label_path_for(staging_dir, image_path)
            if not label_path.exists():
                continue

            class_ids_in_image = {
                class_id for class_id, _ in parse_yolo_label_file(label_path)
            }
            for class_id in class_ids_in_image:
                index.setdefault(class_id, []).append((dataset, image_path))

    return index


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def render_sample(
    image_path: Path,
    annotations: list[tuple[int, list[float]]],
    id_to_name: dict[int, str],
    id_to_color: dict[int, tuple[float, float, float]],
    output_path: Path,
    title: str,
) -> None:
    """Draw bboxes on the image and save the figure as PNG."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    for class_id, bbox in annotations:
        x1, y1, x2, y2 = yolo_to_xyxy(bbox, width, height)
        color = id_to_color.get(class_id, FALLBACK_COLOR)

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        label_text = f"{id_to_name.get(class_id, f'id={class_id}')} ({class_id})"
        ax.text(
            x1,
            max(0, y1 - 4),
            label_text,
            fontsize=9,
            color="white",
            bbox=dict(facecolor=color, alpha=0.75, pad=1),
        )

    ax.set_title(title, fontsize=11)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-class driver
# ---------------------------------------------------------------------------
def _output_dir(seed: int) -> Path:
    """Per-seed output directory so different runs don't overwrite each other."""
    return get_outputs_dir() / "staging_bbox_checks" / f"seed_{seed:02d}"


def visualize_class(
    class_id: int,
    class_name: str,
    candidates: list[tuple[str, Path]],
    id_to_name: dict[int, str],
    id_to_color: dict[int, tuple[float, float, float]],
    samples: int,
    rng: random.Random,
    output_dir: Path,
) -> list[Path]:
    """Render ``samples`` random images that contain ``class_id``.

    The chosen images are drawn with all of their bboxes (not just the ones
    matching ``class_id``) so context is preserved.
    """
    print(f"\n--- {class_name} (id={class_id}) ---")
    print(f"  candidates: {len(candidates)}")

    if not candidates:
        print("  WARNING: no staged images contain this class, skipping.")
        return []

    chosen = rng.sample(candidates, k=min(samples, len(candidates)))
    chosen.sort(key=lambda pair: (pair[0], pair[1].name))  # deterministic output order

    written: list[Path] = []
    for index, (dataset, image_path) in enumerate(chosen, start=1):
        label_path = label_path_for(_staging_dir_for(dataset), image_path)
        annotations = parse_yolo_label_file(label_path)

        output_path = output_dir / f"{class_name}_sample_{index:02d}.png"
        title = (
            f"{class_name} (id={class_id}) — {dataset}/{image_path.name} "
            f"— {len(annotations)} bbox{'es' if len(annotations) != 1 else ''}"
        )
        render_sample(image_path, annotations, id_to_name, id_to_color, output_path, title)
        written.append(output_path)
        print(f"  wrote: {output_path}")

    return written


# Cache resolved per-dataset staging dirs once so visualize_class can map
# back from the dataset label stored in the index.
_DATASET_STAGING_DIRS: dict[str, Path] = {
    "open_images": get_open_images_staging_dir(),
    "uec_food": get_uec_staging_dir(),
}


def _staging_dir_for(dataset: str) -> Path:
    return _DATASET_STAGING_DIRS[dataset]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.validation.visualize_yolo_mapping",
        description=(
            "Render random staged images per YOLO class as a visual sanity "
            "check of the per-source staging conversions."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Seed for the per-class random sampling. Same seed + same "
            "staging contents reproduce the exact same PNGs. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help=(
            "How many random images to render per class. If a class has fewer "
            "candidates, all of them are used. Default: %(default)s."
        ),
    )
    args = parser.parse_args(argv)

    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be >= 1")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print("=" * 60)
    print("YOLO staging visual sanity check (per class)")
    print("=" * 60)
    print(f"seed:              {args.seed}")
    print(f"samples_per_class: {args.samples_per_class}")

    class_entries = load_classes_config()
    id_to_name, id_to_color = build_class_lookups(class_entries)
    print(f"Loaded {len(id_to_name)} target classes: {id_to_name}")

    rng = random.Random(args.seed)
    output_dir = _output_dir(args.seed)
    print(f"output_dir:        {output_dir}")

    staging_dirs = list(_DATASET_STAGING_DIRS.items())
    index = index_images_by_class(staging_dirs)

    all_written: list[Path] = []
    # Iterate classes in id order (food, cup, bottle, ...) for stable output.
    for class_id in sorted(id_to_name.keys()):
        class_name = id_to_name[class_id]
        candidates = index.get(class_id, [])
        written = visualize_class(
            class_id=class_id,
            class_name=class_name,
            candidates=candidates,
            id_to_name=id_to_name,
            id_to_color=id_to_color,
            samples=args.samples_per_class,
            rng=rng,
            output_dir=output_dir,
        )
        all_written.extend(written)

    print("\nGenerated samples:")
    if not all_written:
        print("  (none)")
        return
    for output_path in all_written:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
