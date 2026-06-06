"""
Phase-8 packaging step for notebook 0.5: build the self-contained enriched v2 zip.

The auto-labeling pass (:mod:`src.data.auto_label.auto_label_open_images`) wrote a
v2 ``labels.json`` next to the read-only v1 export, reusing the v1 ``data/`` images
by their original ``file_name``. This script bundles those two pieces — the
enriched ``labels.json`` plus the v1 images it references — into ONE portable zip
(``open_images_table_objects_v2_coco.zip``).

The zip's internal layout mirrors the v1 zip exactly: a root-level ``labels.json``
and ``data/<file_name>`` entries. That symmetry matters because the downstream
pipeline (notebook 01's ``setup_colab_raw_datasets`` extraction) already knows how
to consume the v1 layout; pointing it at v2 then needs no extraction-code change.

The packager is deliberately strict about inputs. The DRIVE v1 zip that produced
v2 on Colab is self-consistent (11,251 images), but the LOCAL v1 zip's
``labels.json`` over-lists image entries that ``data/`` does not actually contain.
So before building anything, this script HARD-VALIDATES that every ``file_name``
the v2 labels reference exists under ``<v1-dir>/data/``; if any are missing it
fails with guidance to run packaging in Colab from the Drive-extracted v1 instead.
After building it re-opens the zip and reads ``labels.json`` back out of it to
confirm the member count and the image/annotation counts match the source doc.

This step reads v1 and the v2 labels as read-only inputs and never modifies them.
It does NOT upload anything to Drive — that is a notebook cell the author runs in
Colab.

Usage (local or Colab, against the already-extracted raw dir + the v2 labels)::

    python -m src.data.auto_label.package_open_images_v2 --check-only
    python -m src.data.auto_label.package_open_images_v2
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from src.data.auto_label.coco_io import load_coco
from src.utils.paths import (
    get_datasets_dir,
    get_open_images_dataset_original_dir,
    get_raw_datasets_dir,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ARCHIVE_DATA_PREFIX = "data"


def default_v2_labels_path() -> Path:
    """Default v2 labels path: the sibling ``open_images_subset_v2`` dir."""
    return get_raw_datasets_dir() / "open_images_subset_v2" / "labels.json"


def default_out_path() -> Path:
    """Default output zip path under ``datasets/``."""
    return get_datasets_dir() / "open_images_table_objects_v2_coco.zip"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_inputs(coco: dict, v1_dir: Path) -> list[dict]:
    """
    Validate the v2 labels against the v1 image directory and print a summary.

    Prints image / annotation / auto-box / category counts, asserts the ``Food``
    category is present, and HARD-CHECKS that every ``images[].file_name`` the v2
    labels reference exists under ``<v1-dir>/data/``. The Drive v1 zip is
    self-consistent at 11,251 images, but the local v1 zip's labels over-list
    files that ``data/`` lacks; if any referenced file is missing this raises a
    ``RuntimeError`` advising to package in Colab from the Drive-extracted v1.
    It also warns (only) about files present in ``data/`` that v2 never
    references.

    Args:
        coco: The loaded v2 COCO document.
        v1_dir: Root of the extracted v1 export (holds the ``data/`` images).

    Returns:
        The list of v2 image entries (each a COCO ``images`` dict), so the
        caller can reuse it as the build manifest without re-reading.

    Raises:
        RuntimeError: If any referenced image file is missing under ``data/``.
        AssertionError: If the ``Food`` category is absent.
    """
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    category_names = {str(cat["name"]) for cat in categories}
    auto_boxes = sum(1 for ann in annotations if ann.get("source") == "auto_label")

    print("=" * 60)
    print("Pre-validation: v2 labels vs v1 images")
    print("=" * 60)
    print(f"v1 dir:           {v1_dir}")
    print(f"Images:           {len(images)}")
    print(f"Annotations:      {len(annotations)}")
    print(f"  auto_label:     {auto_boxes}")
    print(f"  original:       {len(annotations) - auto_boxes}")
    print(f"Categories:       {len(categories)} -> {sorted(category_names)}")

    assert "Food" in category_names, (
        "Expected a 'Food' category in the v2 labels (added by the auto-labeling "
        f"pass), but found only: {sorted(category_names)}."
    )

    data_dir = v1_dir / "data"
    referenced = {str(image["file_name"]) for image in images}

    missing = [name for name in sorted(referenced) if not (data_dir / name).exists()]
    if missing:
        preview = "\n".join(f"  - {name}" for name in missing[:5])
        raise RuntimeError(
            f"{len(missing)} image file(s) referenced by the v2 labels are missing "
            f"under {data_dir}. First few:\n{preview}\n"
            "The LOCAL v1 zip's labels.json over-lists images that data/ does not "
            "contain; the DRIVE v1 zip (which produced v2 on Colab) is "
            "self-consistent. Run this packaging step in Colab from the "
            "Drive-extracted v1 export so every referenced image is present."
        )

    # Warn-only: files on disk that v2 never references (e.g. leftover extras).
    on_disk = {p.name for p in data_dir.glob("*") if p.is_file()}
    unreferenced = on_disk - referenced
    if unreferenced:
        print(
            f"WARN: {len(unreferenced)} file(s) in {data_dir} are NOT referenced by "
            "the v2 labels; they will be left out of the zip."
        )

    print(f"OK: all {len(referenced)} referenced image(s) present under {data_dir}.")
    return images


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_zip(coco: dict, images: list[dict], v1_dir: Path, v2_labels: Path, out: Path) -> None:
    """
    Build the self-contained v2 zip mirroring the v1 internal layout.

    Writes a root-level ``labels.json`` (deflated — JSON compresses well) and one
    ``data/<file_name>`` entry per referenced image (stored, not deflated — JPEGs
    are already compressed, so deflate only burns CPU). Refuses to silently
    overwrite: an existing ``out`` is deleted first with a printed notice, making
    the rebuild idempotent.

    Args:
        coco: The loaded v2 COCO document (unused for content beyond ``images``,
            kept for signature symmetry / future use).
        images: The v2 image entries to bundle, as returned by
            :func:`validate_inputs`.
        v1_dir: Root of the v1 export holding the ``data/`` images.
        v2_labels: Path to the enriched v2 ``labels.json`` to bundle at the root.
        out: Destination zip path.
    """
    from tqdm import tqdm

    if out.exists():
        print(f"NOTICE: output zip already exists, removing for a clean rebuild: {out}")
        out.unlink()

    out.parent.mkdir(parents=True, exist_ok=True)
    data_dir = v1_dir / "data"

    print(f"Building zip: {out}")
    with zipfile.ZipFile(out, "w") as zf:
        # labels.json at the archive root, deflated (text compresses well).
        zf.write(v2_labels, arcname="labels.json", compress_type=zipfile.ZIP_DEFLATED)

        # Each image under data/, stored (JPEGs don't meaningfully compress).
        for image in tqdm(images, desc="Packaging", unit="img"):
            file_name = str(image["file_name"])
            zf.write(
                data_dir / file_name,
                arcname=f"{ARCHIVE_DATA_PREFIX}/{file_name}",
                compress_type=zipfile.ZIP_STORED,
            )


def validate_zip(out: Path, coco: dict, expected_images: int) -> None:
    """
    Re-open the built zip and confirm its contents match the source doc.

    Asserts the member count equals ``1 + expected_images`` (the root
    ``labels.json`` plus one entry per image), reads ``labels.json`` back FROM
    the zip, and asserts its image and annotation counts equal the source v2
    document's. Prints a final summary (path, size in GB, counts).

    Args:
        out: The built zip path.
        coco: The source v2 COCO document the zip was built from.
        expected_images: Number of image entries that should be bundled.

    Raises:
        AssertionError: If member counts or round-tripped label counts mismatch.
    """
    src_images = len(coco.get("images", []))
    src_annotations = len(coco.get("annotations", []))

    with zipfile.ZipFile(out, "r") as zf:
        members = zf.namelist()
        expected_members = 1 + expected_images
        assert len(members) == expected_members, (
            f"Zip member count mismatch: expected {expected_members} "
            f"(1 labels.json + {expected_images} images), found {len(members)}."
        )

        with zf.open("labels.json") as f:
            zipped = json.load(f)

    zipped_images = len(zipped.get("images", []))
    zipped_annotations = len(zipped.get("annotations", []))
    assert zipped_images == src_images, (
        f"Round-tripped image count mismatch: zip has {zipped_images}, "
        f"source has {src_images}."
    )
    assert zipped_annotations == src_annotations, (
        f"Round-tripped annotation count mismatch: zip has {zipped_annotations}, "
        f"source has {src_annotations}."
    )

    size_gb = out.stat().st_size / (1024**3)
    print("\n" + "=" * 60)
    print("Post-validation: zip contents match the v2 labels")
    print("=" * 60)
    print(f"Zip path:    {out}")
    print(f"Size:        {size_gb:.2f} GB")
    print(f"Members:     {len(members)} (1 labels.json + {expected_images} images)")
    print(f"Images:      {zipped_images}")
    print(f"Annotations: {zipped_annotations}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for the v2 packaging step."""
    parser = argparse.ArgumentParser(
        prog="python -m src.data.auto_label.package_open_images_v2",
        description=(
            "Build the self-contained enriched v2 zip (root labels.json + data/ "
            "images) mirroring the v1 zip layout. v1 and the v2 labels are "
            "read-only inputs; nothing is uploaded."
        ),
    )
    parser.add_argument(
        "--v1-dir",
        type=Path,
        default=get_open_images_dataset_original_dir(),
        help="Root of the extracted v1 export (labels.json + data/ images).",
    )
    parser.add_argument(
        "--v2-labels",
        type=Path,
        default=default_v2_labels_path(),
        help="Path to the enriched v2 labels.json to bundle.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out_path(),
        help="Destination zip path (under datasets/ by default).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs only; do not build the zip.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Validate inputs, then (unless --check-only) build and re-validate the zip."""
    args = parse_args(argv)

    v1_dir: Path = args.v1_dir
    v2_labels: Path = args.v2_labels
    out: Path = args.out

    coco = load_coco(v2_labels)
    images = validate_inputs(coco, v1_dir)

    if args.check_only:
        print("\n--check-only: validation passed; no zip built.")
        return

    build_zip(coco, images, v1_dir, v2_labels, out)
    validate_zip(out, coco, expected_images=len(images))


if __name__ == "__main__":
    main()
