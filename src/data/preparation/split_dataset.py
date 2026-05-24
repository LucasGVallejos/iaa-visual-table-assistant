"""
Stratified train/val/test split of the prepared YOLO dataset.

Walks ``datasets/table_assistant_yolo/labels/*.txt``, computes a
representative class for each image (the rarest class present, where rarity
is measured by image count across the full dataset), then performs a
stratified 60/15/25 split.

The split is materialized as three text files of image paths under
``reports/dataset_splits/``. YOLO consumes those paths through
``configs/data.yaml`` without moving any image on disk.

The script also refreshes a ``## Class Distribution`` section in
``reports/dataset_notes.md`` so the latest counts (and any class-imbalance
risk) are persisted alongside the split files. The rest of the notes file
is left untouched.

If a future iteration shows that minority classes (knife, fork, spoon) are
still poorly distributed across splits, the next step is to switch to
multi-label iterative stratification.

Run with::

    python -m src.data.preparation.split_dataset
    python -m src.data.preparation.split_dataset --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data.common.convert_to_yolo import load_classes_config
from src.data.common.dataset_io import parse_yolo_label_file
from src.utils.paths import (
    get_reports_dir,
    get_table_assistant_yolo_dir,
)


# Ratios per the phase 1 plan: 60% train, 15% val, 25% test.
TRAIN_RATIO = 0.60
VAL_RATIO = 0.15
TEST_RATIO = 0.25
DEFAULT_SEED = 42

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

# Markers used to delimit the auto-generated section in dataset_notes.md.
# Anything between these markers is rewritten on every split run; anything
# outside is left untouched.
NOTES_BEGIN_MARKER = "<!-- BEGIN_SPLIT_DISTRIBUTION -->"
NOTES_END_MARKER = "<!-- END_SPLIT_DISTRIBUTION -->"

# Threshold at which class imbalance is flagged as a risk in dataset_notes.md.
IMBALANCE_RISK_RATIO = 10.0


# ---------------------------------------------------------------------------
# Dataset scan
# ---------------------------------------------------------------------------
def collect_images_with_classes(
    labels_dir: Path,
) -> dict[str, set[int]]:
    """Map ``image_stem -> {class_id, ...}`` for every label file present."""
    out: dict[str, set[int]] = {}
    for label_path in sorted(labels_dir.iterdir()):
        if label_path.suffix != ".txt":
            continue
        annotations = parse_yolo_label_file(label_path)
        if not annotations:
            continue
        out[label_path.stem] = {class_id for class_id, _ in annotations}
    return out


def compute_images_per_class(
    images_with_classes: dict[str, set[int]],
) -> dict[int, int]:
    """Count, for each class id, how many images contain at least one bbox of it."""
    counts: dict[int, int] = {}
    for class_set in images_with_classes.values():
        for class_id in class_set:
            counts[class_id] = counts.get(class_id, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Representative class
# ---------------------------------------------------------------------------
def assign_representative_classes(
    images_with_classes: dict[str, set[int]],
    images_per_class: dict[int, int],
) -> dict[str, int]:
    """Pick the rarest class present in each image as its stratification bucket.

    Rarity is measured at the image level: a class is considered rarer when
    fewer images contain at least one of its bboxes. Ties resolve to the
    smaller class id for determinism.
    """
    representative: dict[str, int] = {}
    for stem, class_set in images_with_classes.items():
        # ``min`` with a tuple key gives us the rarest class plus a tiebreaker.
        representative[stem] = min(
            class_set,
            key=lambda cid: (images_per_class[cid], cid),
        )
    return representative


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------
def stratified_split(
    image_stems: list[str],
    representatives: list[int],
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Run a 60/15/25 stratified split using two ``train_test_split`` passes.

    First pass: 75% trainval / 25% test.
    Second pass: of the trainval slice, 80% train / 20% val (so 60/15 of total).
    """
    trainval_stems, test_stems, trainval_reps, _ = train_test_split(
        image_stems,
        representatives,
        test_size=TEST_RATIO,
        stratify=representatives,
        random_state=seed,
    )

    val_share_of_trainval = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_stems, val_stems = train_test_split(
        trainval_stems,
        test_size=val_share_of_trainval,
        stratify=trainval_reps,
        random_state=seed,
    )

    return train_stems, val_stems, test_stems


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def split_files_dir() -> Path:
    return get_reports_dir() / "dataset_splits"


def write_split_files(
    splits: dict[str, list[str]],
    images_dir: Path,
) -> dict[str, Path]:
    """Write one ``.txt`` per split with absolute image paths, one per line."""
    out_dir = split_files_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for split_name, stems in splits.items():
        out_path = out_dir / f"{split_name}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for stem in stems:
                # YOLO accepts absolute paths or paths relative to data.yaml.
                # We write absolute to remove ambiguity about cwd.
                image_path = (images_dir / f"{stem}.jpg").resolve()
                f.write(f"{image_path}\n")
        written[split_name] = out_path

    return written


# ---------------------------------------------------------------------------
# dataset_notes.md update
# ---------------------------------------------------------------------------
def _build_split_totals_table(splits: dict[str, list[str]]) -> str:
    total = sum(len(s) for s in splits.values())
    lines = [
        "| Split | Images | Share |",
        "|-------|-------:|------:|",
    ]
    for split_name in SPLIT_NAMES:
        n = len(splits[split_name])
        share = (n / total * 100.0) if total else 0.0
        lines.append(f"| {split_name} | {n} | {share:.2f}% |")
    lines.append(f"| total | {total} | 100% |")
    return "\n".join(lines)


def _build_per_class_image_table(
    image_counts: dict[int, dict[str, int]],
    id_to_name: dict[int, str],
) -> str:
    lines = [
        "| Class | Total | Train | Val | Test |",
        "|-------|------:|------:|----:|-----:|",
    ]
    for class_id in sorted(image_counts.keys()):
        row = image_counts[class_id]
        name = id_to_name.get(class_id, str(class_id))
        lines.append(
            f"| {name} | {row['total']} | {row['train']} | {row['val']} | {row['test']} |"
        )
    return "\n".join(lines)


def _build_per_class_box_table(
    box_counts: dict[int, dict[str, int]],
    id_to_name: dict[int, str],
) -> str:
    lines = [
        "| Class | Train | Val | Test |",
        "|-------|------:|----:|-----:|",
    ]
    for class_id in sorted(box_counts.keys()):
        row = box_counts[class_id]
        name = id_to_name.get(class_id, str(class_id))
        lines.append(
            f"| {name} | {row['train']} | {row['val']} | {row['test']} |"
        )
    return "\n".join(lines)


def _build_risks_section(
    box_counts: dict[int, dict[str, int]],
    id_to_name: dict[int, str],
) -> str:
    """Build the Risks section for dataset_notes.md based on train-set imbalance."""
    train_boxes = {
        cid: row["train"] for cid, row in box_counts.items() if row["train"] > 0
    }
    if not train_boxes:
        return "_No training boxes found; cannot evaluate class imbalance._"

    max_id = max(train_boxes, key=train_boxes.get)
    min_id = min(train_boxes, key=train_boxes.get)
    max_count = train_boxes[max_id]
    min_count = train_boxes[min_id]
    ratio = max_count / min_count

    if ratio < IMBALANCE_RISK_RATIO:
        return (
            f"- Class imbalance: max/min ratio in train is **{ratio:.1f}** "
            f"({id_to_name.get(max_id, max_id)}={max_count} vs "
            f"{id_to_name.get(min_id, min_id)}={min_count}). "
            f"Below the {IMBALANCE_RISK_RATIO:.0f}:1 threshold; no action needed yet."
        )

    return (
        f"- **Class imbalance**: max/min ratio in train is **{ratio:.1f}** "
        f"({id_to_name.get(max_id, max_id)}={max_count} boxes vs "
        f"{id_to_name.get(min_id, min_id)}={min_count}). "
        f"Threshold of >{IMBALANCE_RISK_RATIO:.0f}:1 exceeded. Consider class "
        f"weighting or focal loss during training. Underrepresented classes "
        f"may produce noisy mAP@class metrics."
    )


def build_distribution_section(
    seed: int,
    splits: dict[str, list[str]],
    image_counts: dict[int, dict[str, int]],
    box_counts: dict[int, dict[str, int]],
    id_to_name: dict[int, str],
) -> str:
    """Compose the auto-managed ``## Class Distribution`` section as a string."""
    return "\n\n".join(
        [
            f"{NOTES_BEGIN_MARKER}",
            "## Class Distribution",
            (
                f"_Auto-generated by `src/data/preparation/split_dataset.py` "
                f"(seed {seed}). Do not edit by hand inside the markers._"
            ),
            "### Splits",
            _build_split_totals_table(splits),
            "### Per-class images (stratification buckets)",
            _build_per_class_image_table(image_counts, id_to_name),
            "### Per-class boxes",
            _build_per_class_box_table(box_counts, id_to_name),
            "### Risks",
            _build_risks_section(box_counts, id_to_name),
            f"{NOTES_END_MARKER}",
        ]
    )


def update_dataset_notes(notes_path: Path, section: str) -> None:
    """Replace (or append) the auto-managed section in ``reports/dataset_notes.md``.

    Behavior:
    - If the markers exist, the block between them is replaced.
    - If the markers don't exist but a legacy ``## Class Distribution`` heading
      is present, that heading and any following content up to the next ``## ``
      heading are replaced by the new marked block.
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

    # Legacy placeholder migration: drop the old "## Class Distribution"
    # heading (and the table that follows) so we replace it with the
    # marker-wrapped version.
    legacy_heading = "## Class Distribution"
    if legacy_heading in existing:
        before, _, rest = existing.partition(legacy_heading)
        # Find the next top-level section after the legacy block.
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
def _print_class_rarity(
    images_per_class: dict[int, int],
    id_to_name: dict[int, str],
) -> None:
    print("\nComputing class rarity (images per class):")
    for class_id, count in sorted(images_per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {id_to_name.get(class_id, str(class_id)):<14} {count:>6}")


def _print_representative_buckets(
    representatives: dict[str, int],
    id_to_name: dict[int, str],
) -> None:
    bucket_counts: dict[int, int] = {}
    for class_id in representatives.values():
        bucket_counts[class_id] = bucket_counts.get(class_id, 0) + 1

    print("\nRepresentative class assignment (rarest present per image):")
    total = 0
    for class_id, count in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {id_to_name.get(class_id, str(class_id)):<14} {count:>6}")
        total += count
    print(f"  {'-' * 22}")
    print(f"  {'total':<14} {total:>6}")


def _print_split_totals(splits: dict[str, list[str]]) -> None:
    total = sum(len(s) for s in splits.values())
    print("\nSplits:")
    for split_name in SPLIT_NAMES:
        n = len(splits[split_name])
        share = n / total * 100.0 if total else 0.0
        print(f"  {split_name:<6} {n:>6}  {share:>6.2f}%")


def _per_class_image_counts(
    splits: dict[str, list[str]],
    images_with_classes: dict[str, set[int]],
    representatives: dict[str, int],
) -> dict[int, dict[str, int]]:
    """Count images whose representative class is X in each split."""
    counts: dict[int, dict[str, int]] = {}
    for split_name, stems in splits.items():
        for stem in stems:
            class_id = representatives[stem]
            bucket = counts.setdefault(
                class_id, {name: 0 for name in SPLIT_NAMES} | {"total": 0}
            )
            bucket[split_name] += 1
            bucket["total"] += 1
    return counts


def _per_class_box_counts(
    splits: dict[str, list[str]],
    images_with_classes: dict[str, set[int]],
    labels_dir: Path,
) -> dict[int, dict[str, int]]:
    """Count bboxes per class in each split (informational, all classes per image)."""
    out: dict[int, dict[str, int]] = {}
    for split_name, stems in splits.items():
        for stem in stems:
            label_path = labels_dir / f"{stem}.txt"
            for class_id, _ in parse_yolo_label_file(label_path):
                bucket = out.setdefault(
                    class_id, {name: 0 for name in SPLIT_NAMES}
                )
                bucket[split_name] += 1
    return out


def _print_per_class_table(
    title: str,
    counts: dict[int, dict[str, int]],
    id_to_name: dict[int, str],
    include_total_column: bool,
) -> None:
    print(f"\n{title}:")
    if include_total_column:
        header = f"  {'class':<14}{'total':>8}{'train':>8}{'val':>7}{'test':>8}"
    else:
        header = f"  {'class':<14}{'train':>8}{'val':>7}{'test':>8}"
    print(header)

    for class_id in sorted(counts.keys()):
        row = counts[class_id]
        name = id_to_name.get(class_id, str(class_id))
        if include_total_column:
            total = row["total"]
            print(
                f"  {name:<14}{total:>8}{row['train']:>8}{row['val']:>7}{row['test']:>8}"
            )
        else:
            print(
                f"  {name:<14}{row['train']:>8}{row['val']:>7}{row['test']:>8}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.preparation.split_dataset",
        description=(
            "Stratified 60/15/25 train/val/test split of the prepared YOLO "
            "dataset. Writes path lists to reports/dataset_splits/."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Random seed for the stratified split. Same seed + same dataset "
            "reproduce the exact same splits. Default: %(default)s."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print("=" * 60)
    print("Dataset split (stratified by rarest class per image)")
    print("=" * 60)
    print(f"seed:    {args.seed}")
    print(f"ratios:  train={TRAIN_RATIO:.2f} val={VAL_RATIO:.2f} test={TEST_RATIO:.2f}")

    yolo_dir = get_table_assistant_yolo_dir()
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    print(f"input:   {yolo_dir}")
    print(f"output:  {split_files_dir()}/{{train,val,test}}.txt")

    images_with_classes = collect_images_with_classes(labels_dir)
    if not images_with_classes:
        print("\nNo labelled images found. Run prepare_dataset.py first:")
        print("  python -m src.data.preparation.prepare_dataset")
        return
    print(f"images found: {len(images_with_classes)}")

    classes = load_classes_config()
    id_to_name = {int(c["id"]): c["name"] for c in classes}

    images_per_class = compute_images_per_class(images_with_classes)
    _print_class_rarity(images_per_class, id_to_name)

    representatives = assign_representative_classes(
        images_with_classes, images_per_class
    )
    _print_representative_buckets(representatives, id_to_name)

    image_stems = list(representatives.keys())
    rep_list = [representatives[stem] for stem in image_stems]

    train_stems, val_stems, test_stems = stratified_split(
        image_stems=image_stems,
        representatives=rep_list,
        seed=args.seed,
    )
    splits = {"train": train_stems, "val": val_stems, "test": test_stems}

    _print_split_totals(splits)

    image_counts = _per_class_image_counts(splits, images_with_classes, representatives)
    _print_per_class_table(
        "Per-class images (after stratification)",
        image_counts,
        id_to_name,
        include_total_column=True,
    )

    box_counts = _per_class_box_counts(splits, images_with_classes, labels_dir)
    _print_per_class_table(
        "Per-class boxes (across all classes per image, informational)",
        box_counts,
        id_to_name,
        include_total_column=False,
    )

    written = write_split_files(splits, images_dir)
    print("\nWrote:")
    for split_name in SPLIT_NAMES:
        n = len(splits[split_name])
        print(f"  {written[split_name]}   ({n:>6} paths)")

    notes_section = build_distribution_section(
        seed=args.seed,
        splits=splits,
        image_counts=image_counts,
        box_counts=box_counts,
        id_to_name=id_to_name,
    )
    notes_path = get_reports_dir() / "dataset_notes.md"
    update_dataset_notes(notes_path, notes_section)
    print(f"  {notes_path}   (updated ## Class Distribution section)")


if __name__ == "__main__":
    main()
