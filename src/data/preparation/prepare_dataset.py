"""
Merge per-source YOLO staging dirs into the final flat YOLO dataset.

Reads ``datasets/_staging/{open_images,uec_food}/``, copies every image+label pair to
``datasets/table_assistant_yolo/{images,labels}/`` under a deterministic
sequence-and-classes filename, writes a provenance manifest at
``reports/dataset_manifest.csv``, and cleans up the staging dirs at the
end.

Filename scheme (Decision 3 of the phase 1 plan)::

    <NNNNNNNN>_<class_name_a>_<class_name_b>_..._<class_name_n>.jpg

Where ``NNNNNNNN`` is an 8-digit zero-padded sequential id and class names
are ordered by class id ascending.

Run with::

    python -m src.data.preparation.prepare_dataset
    python -m src.data.preparation.prepare_dataset --keep-staging
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

from src.data.common.convert_to_yolo import load_class_mapping
from src.data.common.dataset_io import (
    list_staging_images,
    parse_yolo_label_file,
)
from src.utils.paths import (
    get_open_images_staging_dir,
    get_reports_dir,
    get_table_assistant_yolo_dir,
    get_uec_staging_dir,
)


SOURCES_IN_ORDER: list[str] = ["open_images", "uec_food"]
TOP_COMBINATIONS_PRINTED = 10
MANIFEST_HEADER = ["new_name", "source", "original_relative_path", "original_id"]


# ---------------------------------------------------------------------------
# Provenance derivation
# ---------------------------------------------------------------------------
def _staging_dirs() -> dict[str, Path]:
    """Resolve once and reuse so we don't pay path lookups per file."""
    return {
        "open_images": get_open_images_staging_dir(),
        "uec_food": get_uec_staging_dir(),
    }


def derive_provenance(source: str, image_path: Path) -> tuple[str, str]:
    """Re-derive ``(original_relative_path, original_id)`` from a staging file.

    UEC FOOD-256 staging filenames follow ``<3-digit-category>_<imgid>.jpg``,
    e.g. ``001_100.jpg`` (that scheme exists to disambiguate ``100.jpg``
    collisions across categories). We parse the prefix back to the raw layout
    ``<category>/<imgid>.jpg``.

    Open Images staging keeps the original filename (e.g. ``0000608cc97a2b17``)
    which is itself the Open Images image id; the raw path is always
    ``data/<stem>.jpg``.

    The returned paths are relative to the raw dataset root resolved by
    ``get_uec_food_dataset_extract_dir()`` and
    ``get_open_images_dataset_original_dir()`` respectively.
    """
    stem = image_path.stem

    if source == "uec_food":
        match = re.match(r"^(\d{1,3})_(.+)$", stem)
        if not match:
            # Fall back to the raw stem; provenance is best-effort.
            return f"{stem}.jpg", stem
        category, imgid = match.group(1), match.group(2)
        category_int = str(int(category))  # strip leading zeros for the raw path
        return f"{category_int}/{imgid}.jpg", f"{category_int}/{imgid}"

    if source == "open_images":
        return f"data/{stem}.jpg", stem

    raise ValueError(f"unknown source: {source!r}")


# ---------------------------------------------------------------------------
# Final filename
# ---------------------------------------------------------------------------
def _ordered_class_names(
    class_ids: set[int], id_to_name: dict[int, str]
) -> list[str]:
    return [id_to_name[cid] for cid in sorted(class_ids) if cid in id_to_name]


def build_final_stem(
    sequence: int, class_ids: set[int], id_to_name: dict[int, str]
) -> str:
    """Return the final filename stem ``<NNNNNNNN>_<class_a>_<class_b>...``."""
    seq = f"{sequence:08d}"
    classes = _ordered_class_names(class_ids, id_to_name)
    if not classes:
        return seq  # defensive; converters should never let this happen
    return f"{seq}_{'_'.join(classes)}"


# ---------------------------------------------------------------------------
# Wipe + manifest io
# ---------------------------------------------------------------------------
def _wipe_destination(images_dir: Path, labels_dir: Path, manifest_path: Path) -> None:
    if images_dir.exists():
        shutil.rmtree(images_dir)
    if labels_dir.exists():
        shutil.rmtree(labels_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest_path.unlink()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)


def _open_manifest(manifest_path: Path):
    """Open the manifest CSV for writing and return ``(file, writer)``."""
    handle = open(manifest_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow(MANIFEST_HEADER)
    return handle, writer


# ---------------------------------------------------------------------------
# Per-source merge loop
# ---------------------------------------------------------------------------
def merge_source(
    source: str,
    staging_dir: Path,
    images_dir: Path,
    labels_dir: Path,
    manifest_writer: csv.writer,
    id_to_name: dict[int, str],
    sequence_start: int,
) -> tuple[int, int, dict[int, int], dict[str, int]]:
    """Copy every (image, label) pair of ``source`` to the final layout.

    Returns ``(pairs_found, pairs_written, boxes_per_class, suffix_counts)``
    so the caller can aggregate cross-source totals.
    """
    pairs_found = 0
    pairs_written = 0
    boxes_per_class: dict[int, int] = {}
    suffix_counts: dict[str, int] = {}

    print(f"\nProcessing {source}...")

    if not staging_dir.exists():
        print(f"  staging dir not found: {staging_dir}")
        return pairs_found, pairs_written, boxes_per_class, suffix_counts

    images = list_staging_images(staging_dir)
    sequence = sequence_start

    for image_path in images:
        label_path = staging_dir / "labels" / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        pairs_found += 1

        annotations = parse_yolo_label_file(label_path)
        if not annotations:
            continue

        class_ids_in_image = {class_id for class_id, _ in annotations}
        for class_id in class_ids_in_image:
            # accounting against the per-image set keeps "boxes per class"
            # below faithful: we'll add per-bbox below.
            pass
        for class_id, _ in annotations:
            boxes_per_class[class_id] = boxes_per_class.get(class_id, 0) + 1

        sequence += 1
        final_stem = build_final_stem(sequence, class_ids_in_image, id_to_name)
        suffix = "_" + "_".join(_ordered_class_names(class_ids_in_image, id_to_name))
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

        new_image_path = images_dir / f"{final_stem}.jpg"
        new_label_path = labels_dir / f"{final_stem}.txt"

        try:
            shutil.copy2(image_path, new_image_path)
            shutil.copy2(label_path, new_label_path)
        except Exception:
            # roll back partial writes; sequence advanced but file isn't on
            # disk -> pairs_written < pairs_found and WARN-1 will fire.
            if new_image_path.exists():
                new_image_path.unlink()
            if new_label_path.exists():
                new_label_path.unlink()
            continue

        original_relative_path, original_id = derive_provenance(source, image_path)
        manifest_writer.writerow(
            [f"{final_stem}.jpg", source, original_relative_path, original_id]
        )
        pairs_written += 1

    print(f"  pairs found: {pairs_found}")
    print(f"  written:     {pairs_written}")

    return pairs_found, pairs_written, boxes_per_class, suffix_counts


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
def _format_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def cleanup_staging(staging_dirs: dict[str, Path], keep: bool) -> None:
    if keep:
        print("\nCleanup:")
        print("  --keep-staging set, leaving _staging dirs intact.")
        return

    print("\nCleanup:")
    for source, staging_dir in staging_dirs.items():
        if not staging_dir.exists():
            continue
        size = _dir_size(staging_dir)
        shutil.rmtree(staging_dir)
        print(f"  Removed {staging_dir} (~{_format_size(size)})")


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------
def _print_pairs_table(pairs_per_source: dict[str, int]) -> None:
    print("\nPairs written:")
    total = 0
    for source in SOURCES_IN_ORDER:
        n = pairs_per_source.get(source, 0)
        print(f"  {source:<13} {n}")
        total += n
    print(f"  {'total':<13} {total}")


def _print_boxes_table(boxes_per_class: dict[int, int], id_to_name: dict[int, str]) -> None:
    total = sum(boxes_per_class.values())
    print("\nBoxes per class (final dataset):")
    print(f"  {'class':<14}{'id':>4}{'boxes':>10}{'share':>10}")
    for class_id in sorted(id_to_name.keys()):
        boxes = boxes_per_class.get(class_id, 0)
        share = (boxes / total * 100.0) if total else 0.0
        print(f"  {id_to_name[class_id]:<14}{class_id:>4}{boxes:>10}{share:>9.2f}%")
    print(f"  {'-' * 38}")
    print(f"  {'total':<18}{total:>10}{100.0 if total else 0.0:>9.2f}%")


def _print_suffix_table(suffix_counts: dict[str, int]) -> None:
    if not suffix_counts:
        return
    total = sum(suffix_counts.values())
    sorted_suffixes = sorted(suffix_counts.items(), key=lambda kv: kv[1], reverse=True)
    head = sorted_suffixes[:TOP_COMBINATIONS_PRINTED]
    tail = sorted_suffixes[TOP_COMBINATIONS_PRINTED:]

    print("\nTop class combinations (filename suffix):")
    for suffix, count in head:
        share = count / total * 100.0
        print(f"  {suffix:<28}{count:>8}{share:>8.2f}%")
    if tail:
        tail_count = sum(c for _, c in tail)
        tail_share = tail_count / total * 100.0
        print(
            f"  ... and {len(tail)} more combinations "
            f"({tail_share:.2f}% of total)"
        )


def _print_warnings(warnings: list[tuple[str, str, str]]) -> None:
    """Each warning is ``(code, what, why)``."""
    if not warnings:
        return
    print(f"\nWARNINGS ({len(warnings)}):\n")
    for code, what, why in warnings:
        print(f"  [{code}]")
        print(f"      What: {what}")
        print(f"      Why:  {why}")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.preparation.prepare_dataset",
        description=(
            "Merge per-source YOLO staging dirs into the final flat YOLO "
            "dataset, write the provenance manifest, and clean up staging."
        ),
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Skip cleanup of datasets/_staging at the end (debug aid).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print("=" * 60)
    print("Prepare dataset (merge + rename)")
    print("=" * 60)

    staging_dirs = _staging_dirs()
    yolo_dir = get_table_assistant_yolo_dir()
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    manifest_path = get_reports_dir() / "dataset_manifest.csv"

    print("staging:")
    for source in SOURCES_IN_ORDER:
        print(f"  {source:<13} {staging_dirs[source]}")

    print("destination:")
    print(f"  images:   {images_dir}")
    print(f"  labels:   {labels_dir}")
    print(f"  manifest: {manifest_path}")

    print("\nWiping destination and manifest... ", end="")
    _wipe_destination(images_dir, labels_dir, manifest_path)
    print("done.")

    class_mapping = load_class_mapping()  # {name: id}
    id_to_name: dict[int, str] = {int(cid): name for name, cid in class_mapping.items()}

    handle, writer = _open_manifest(manifest_path)
    try:
        sequence = 0
        pairs_per_source: dict[str, int] = {}
        cumulative_boxes: dict[int, int] = {}
        cumulative_suffixes: dict[str, int] = {}
        per_source_diff: list[tuple[str, int, int]] = []  # (source, found, written)

        for source in SOURCES_IN_ORDER:
            found, written, boxes, suffixes = merge_source(
                source=source,
                staging_dir=staging_dirs[source],
                images_dir=images_dir,
                labels_dir=labels_dir,
                manifest_writer=writer,
                id_to_name=id_to_name,
                sequence_start=sequence,
            )
            sequence += written
            pairs_per_source[source] = written
            per_source_diff.append((source, found, written))
            for cid, n in boxes.items():
                cumulative_boxes[cid] = cumulative_boxes.get(cid, 0) + n
            for suffix, n in suffixes.items():
                cumulative_suffixes[suffix] = cumulative_suffixes.get(suffix, 0) + n
    finally:
        handle.close()

    # Count actual rows written to the manifest by reopening it. Cheap and
    # gives us an authoritative count to compare against files copied.
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_rows = sum(1 for _ in f) - 1  # subtract header

    files_written = sum(1 for _ in images_dir.iterdir())

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    _print_pairs_table(pairs_per_source)
    _print_boxes_table(cumulative_boxes, id_to_name)
    _print_suffix_table(cumulative_suffixes)
    cleanup_staging(staging_dirs, keep=args.keep_staging)

    warnings: list[tuple[str, str, str]] = []
    for source, found, written in per_source_diff:
        if found != written:
            warnings.append(
                (
                    "WARN-1",
                    f"{found - written} pair(s) failed to copy from {source} "
                    f"(found={found}, written={written}).",
                    "every staged pair must reach the final dataset. "
                    "Missing pairs mean the dataset is incomplete and the "
                    "manifest may not match what's on disk.",
                )
            )

    if files_written != manifest_rows:
        warnings.append(
            (
                "WARN-2",
                f"wrote {files_written} files but manifest has {manifest_rows} rows.",
                "the manifest must be 1:1 with the final dataset for "
                "provenance lookups to work. A row missing or duplicated "
                "breaks debugging downstream.",
            )
        )

    _print_warnings(warnings)

    print(f"\nWrote: {manifest_path}  ({manifest_rows} rows)")


if __name__ == "__main__":
    main()
