"""
Refresh the auto-managed distribution section in ``reports/dataset_notes.md``.

This script populates ``reports/dataset_notes.md`` with general dataset
statistics for the merged YOLO dataset under
``datasets/table_assistant_yolo/`` — total images, total label files, total
boxes, and per-class image and box counts. It does **not** generate any
train/val/test split.

The split is generated later in ``02_training_colab.ipynb`` via
``src/data/preparation/split_dataset.py``. To keep both scripts compatible,
this script reuses the exact same ``<!-- BEGIN_SPLIT_DISTRIBUTION -->`` /
``<!-- END_SPLIT_DISTRIBUTION -->`` markers used by ``split_dataset.py``: when
the training notebook later runs the split, it overwrites the same section
with split-aware tables.

Idempotent: running it twice produces the same file content.

Run with::

    python -m src.data.preparation.update_dataset_notes
"""

from __future__ import annotations

from pathlib import Path

from src.data.common.convert_to_yolo import load_classes_config
from src.data.common.dataset_io import parse_yolo_label_file
from src.utils.paths import (
    get_reports_dir,
    get_table_assistant_yolo_dir,
)


# Reuse the same markers as split_dataset.py so the section can be replaced
# later by the split-aware version without leaving duplicate blocks behind.
NOTES_BEGIN_MARKER = "<!-- BEGIN_SPLIT_DISTRIBUTION -->"
NOTES_END_MARKER = "<!-- END_SPLIT_DISTRIBUTION -->"

IMBALANCE_RISK_RATIO = 10.0


# ---------------------------------------------------------------------------
# Dataset scan
# ---------------------------------------------------------------------------
def _list_image_stems(images_dir: Path) -> set[str]:
    """Return the set of image stems present under ``images_dir``."""
    if not images_dir.is_dir():
        return set()
    return {
        p.stem for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".jpg"
    }


def _list_label_stems(labels_dir: Path) -> set[str]:
    """Return the set of label stems present under ``labels_dir``."""
    if not labels_dir.is_dir():
        return set()
    return {
        p.stem for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    }


def _aggregate_counts(
    labels_dir: Path,
    label_stems: set[str],
) -> tuple[dict[int, int], dict[int, int], int]:
    """Walk every label file once and accumulate counts.

    Returns:
        - ``image_counts``: per class id, number of images that contain at
          least one bbox of that class.
        - ``box_counts``: per class id, total number of bboxes.
        - ``total_boxes``: total number of bboxes across the dataset.
    """
    image_counts: dict[int, int] = {}
    box_counts: dict[int, int] = {}
    total_boxes = 0

    for stem in sorted(label_stems):
        label_path = labels_dir / f"{stem}.txt"
        annotations = parse_yolo_label_file(label_path)
        if not annotations:
            continue
        classes_in_image: set[int] = set()
        for class_id, _ in annotations:
            box_counts[class_id] = box_counts.get(class_id, 0) + 1
            classes_in_image.add(class_id)
            total_boxes += 1
        for class_id in classes_in_image:
            image_counts[class_id] = image_counts.get(class_id, 0) + 1

    return image_counts, box_counts, total_boxes


# ---------------------------------------------------------------------------
# Section composition
# ---------------------------------------------------------------------------
def _build_totals_table(
    total_images: int,
    total_labels: int,
    total_boxes: int,
) -> str:
    return "\n".join(
        [
            "| Metric | Value |",
            "|--------|------:|",
            f"| Images | {total_images} |",
            f"| Label files | {total_labels} |",
            f"| Boxes | {total_boxes} |",
        ]
    )


def _build_per_class_table(
    image_counts: dict[int, int],
    box_counts: dict[int, int],
    id_to_name: dict[int, str],
    total_images: int,
    total_boxes: int,
) -> str:
    lines = [
        "| Class | Images | Image share | Boxes | Box share |",
        "|-------|-------:|------------:|------:|----------:|",
    ]
    for class_id in sorted(id_to_name.keys()):
        name = id_to_name[class_id]
        n_images = image_counts.get(class_id, 0)
        n_boxes = box_counts.get(class_id, 0)
        image_share = (n_images / total_images * 100.0) if total_images else 0.0
        box_share = (n_boxes / total_boxes * 100.0) if total_boxes else 0.0
        lines.append(
            f"| {name} | {n_images} | {image_share:.2f}% | "
            f"{n_boxes} | {box_share:.2f}% |"
        )
    return "\n".join(lines)


def _build_risks_section(
    box_counts: dict[int, int],
    id_to_name: dict[int, str],
) -> str:
    """Flag class imbalance based on dataset-wide box counts."""
    nonzero = {cid: n for cid, n in box_counts.items() if n > 0}
    if not nonzero:
        return "_No boxes found; cannot evaluate class imbalance._"

    max_id = max(nonzero, key=nonzero.get)
    min_id = min(nonzero, key=nonzero.get)
    max_count = nonzero[max_id]
    min_count = nonzero[min_id]
    ratio = max_count / min_count

    if ratio < IMBALANCE_RISK_RATIO:
        return (
            f"- Class imbalance: max/min ratio across the dataset is "
            f"**{ratio:.1f}** "
            f"({id_to_name.get(max_id, max_id)}={max_count} boxes vs "
            f"{id_to_name.get(min_id, min_id)}={min_count}). "
            f"Below the {IMBALANCE_RISK_RATIO:.0f}:1 threshold; no action "
            f"needed yet."
        )

    return (
        f"- **Class imbalance**: max/min ratio across the dataset is "
        f"**{ratio:.1f}** "
        f"({id_to_name.get(max_id, max_id)}={max_count} boxes vs "
        f"{id_to_name.get(min_id, min_id)}={min_count}). "
        f"Threshold of >{IMBALANCE_RISK_RATIO:.0f}:1 exceeded. Consider class "
        f"weighting or focal loss during training. Underrepresented classes "
        f"may produce noisy mAP@class metrics."
    )


def build_distribution_section(
    total_images: int,
    total_labels: int,
    total_boxes: int,
    image_counts: dict[int, int],
    box_counts: dict[int, int],
    id_to_name: dict[int, str],
) -> str:
    """Compose the auto-managed section as a single string."""
    return "\n\n".join(
        [
            f"{NOTES_BEGIN_MARKER}",
            "## Class Distribution",
            (
                "_Auto-generated by "
                "`src/data/preparation/update_dataset_notes.py`. "
                "Do not edit by hand inside the markers._"
            ),
            (
                "Layout: flat YOLO under `datasets/table_assistant_yolo/"
                "{images,labels}/`. The `train.txt`, `val.txt` and `test.txt` "
                "split files are not produced here; they are generated in "
                "`02_training_colab.ipynb` via "
                "`src/data/preparation/split_dataset.py` (stratified by the "
                "rarest class per image, seed 42, ratios 60/15/25)."
            ),
            "### Totals",
            _build_totals_table(total_images, total_labels, total_boxes),
            "### Per-class counts",
            _build_per_class_table(
                image_counts,
                box_counts,
                id_to_name,
                total_images,
                total_boxes,
            ),
            "### Risks",
            _build_risks_section(box_counts, id_to_name),
            f"{NOTES_END_MARKER}",
        ]
    )


# ---------------------------------------------------------------------------
# dataset_notes.md update
# ---------------------------------------------------------------------------
def update_dataset_notes(notes_path: Path, section: str) -> None:
    """Replace (or append) the auto-managed section in ``dataset_notes.md``.

    Behavior:
    - If the markers exist, the block between them is replaced.
    - If the markers don't exist but a legacy ``## Class Distribution`` heading
      is present, that heading and any following content up to the next
      ``## `` heading are replaced by the new marked block.
    - Otherwise the section is appended at the end of the file.
    """
    notes_path.parent.mkdir(parents=True, exist_ok=True)

    if not notes_path.exists():
        notes_path.write_text(section + "\n", encoding="utf-8")
        return

    existing = notes_path.read_text(encoding="utf-8")

    if NOTES_BEGIN_MARKER in existing and NOTES_END_MARKER in existing:
        before, _, rest = existing.partition(NOTES_BEGIN_MARKER)
        _, _, after = rest.partition(NOTES_END_MARKER)
        new_content = f"{before.rstrip()}\n\n{section}\n{after.lstrip()}"
        notes_path.write_text(new_content, encoding="utf-8")
        return

    legacy_heading = "## Class Distribution"
    if legacy_heading in existing:
        before, _, rest = existing.partition(legacy_heading)
        next_heading_idx = rest.find("\n## ")
        after = rest[next_heading_idx:] if next_heading_idx != -1 else ""
        new_content = f"{before.rstrip()}\n\n{section}\n{after.lstrip()}"
        notes_path.write_text(new_content, encoding="utf-8")
        return

    new_content = existing.rstrip() + "\n\n" + section + "\n"
    notes_path.write_text(new_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_totals(
    total_images: int,
    total_labels: int,
    total_boxes: int,
) -> None:
    print("\nTotals:")
    print(f"  images:      {total_images}")
    print(f"  label files: {total_labels}")
    print(f"  boxes:       {total_boxes}")


def _print_per_class(
    image_counts: dict[int, int],
    box_counts: dict[int, int],
    id_to_name: dict[int, str],
) -> None:
    print("\nPer-class counts:")
    print(f"  {'class':<14}{'images':>8}{'boxes':>8}")
    for class_id in sorted(id_to_name.keys()):
        name = id_to_name[class_id]
        n_images = image_counts.get(class_id, 0)
        n_boxes = box_counts.get(class_id, 0)
        print(f"  {name:<14}{n_images:>8}{n_boxes:>8}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Update dataset_notes.md (general distribution, no splits)")
    print("=" * 60)

    yolo_dir = get_table_assistant_yolo_dir()
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    notes_path = get_reports_dir() / "dataset_notes.md"

    print(f"input:   {yolo_dir}")
    print(f"output:  {notes_path}")

    image_stems = _list_image_stems(images_dir)
    label_stems = _list_label_stems(labels_dir)

    if not image_stems and not label_stems:
        print("\nNo images or labels found. Run prepare_dataset.py first:")
        print("  python -m src.data.preparation.prepare_dataset")
        return

    pair_stems = image_stems & label_stems
    only_image = image_stems - label_stems
    only_label = label_stems - image_stems
    if only_image:
        print(
            f"\n[WARN] {len(only_image)} image(s) without a matching label "
            f"file (skipped from box counts)."
        )
    if only_label:
        print(
            f"[WARN] {len(only_label)} label file(s) without a matching "
            f"image (skipped from box counts)."
        )

    classes = load_classes_config()
    id_to_name = {int(c["id"]): c["name"] for c in classes}

    image_counts, box_counts, total_boxes = _aggregate_counts(
        labels_dir, pair_stems
    )

    total_images = len(image_stems)
    total_labels = len(label_stems)

    _print_totals(total_images, total_labels, total_boxes)
    _print_per_class(image_counts, box_counts, id_to_name)

    section = build_distribution_section(
        total_images=total_images,
        total_labels=total_labels,
        total_boxes=total_boxes,
        image_counts=image_counts,
        box_counts=box_counts,
        id_to_name=id_to_name,
    )
    update_dataset_notes(notes_path, section)
    print(f"\n  {notes_path}   (updated ## Class Distribution section)")


if __name__ == "__main__":
    main()
