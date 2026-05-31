"""
Restore the DVC-tracked dataset package to its active layout.

The dataset is versioned with DVC as a single ZIP::

    datasets/table_assistant_yolo_package.zip

This script unpacks that ZIP into ``datasets/table_assistant_yolo_package/``
and exposes the dataset at the canonical project path
``datasets/table_assistant_yolo/`` via a symlink, so every other tool
(training, validators, ``configs/data.yaml``, ...) keeps addressing the same
location regardless of whether it was generated locally or restored from
DVC.

Run with::

    python -m src.data.preparation.restore_dataset_package
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

from src.utils.paths import get_datasets_dir


PACKAGE_ZIP = get_datasets_dir() / "table_assistant_yolo_package.zip"
PACKAGE_DIR = get_datasets_dir() / "table_assistant_yolo_package"
PACKAGE_DATASET_DIR = PACKAGE_DIR / "table_assistant_yolo"
ACTIVE_DATASET_PATH = get_datasets_dir() / "table_assistant_yolo"

IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png"}
LABEL_EXTENSION = ".txt"


# ---------------------------------------------------------------------------
# ZIP extraction
# ---------------------------------------------------------------------------
def safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract ``zip_path`` into ``output_dir`` while blocking path traversal.

    Each archive member's resolved target path must stay inside the resolved
    ``output_dir``. Any member resolving to a parent directory or to an
    absolute path outside ``output_dir`` raises ``RuntimeError`` and aborts
    the extraction without writing partial state.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = (output_root / member.filename).resolve()
            try:
                target.relative_to(output_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Refusing to extract '{member.filename}': path escapes "
                    f"{output_root}."
                ) from exc

        zf.extractall(output_root)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def _remove_path(path: Path) -> None:
    """Remove ``path`` regardless of whether it is a symlink, dir or file."""
    if path.is_symlink():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    if path.exists():
        path.unlink()


def _link_active_dataset(target: Path, link_path: Path) -> None:
    """Create ``link_path`` as a symlink pointing at ``target``.

    Falls back to a clear error if the platform refuses symlinks (e.g.
    Windows without developer mode). No silent copy fallback is performed:
    duplicating tens of thousands of files needs to be an explicit decision.
    """
    try:
        os.symlink(target.resolve(), link_path)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to create symlink {link_path} -> {target}. "
            "This script expects a POSIX-like environment (e.g. Colab Linux). "
            "If you need a copy-based fallback, request it explicitly."
        ) from exc


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_extracted_layout() -> None:
    """Verify the ZIP unpacked into the documented layout."""
    required = [
        PACKAGE_DIR,
        PACKAGE_DATASET_DIR,
        PACKAGE_DATASET_DIR / "images",
        PACKAGE_DATASET_DIR / "labels",
        PACKAGE_DIR / "metadata",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        bullets = "\n  - ".join(str(p) for p in missing)
        raise RuntimeError(
            "Extracted package is missing required entries:\n  - "
            + bullets
        )


def _list_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _list_labels(labels_dir: Path) -> list[Path]:
    return sorted(
        p for p in labels_dir.iterdir()
        if p.is_file() and p.suffix.lower() == LABEL_EXTENSION
    )


def _validate_active_dataset() -> tuple[int, int]:
    """Verify the active dataset is reachable, balanced and stem-aligned."""
    images_dir = ACTIVE_DATASET_PATH / "images"
    labels_dir = ACTIVE_DATASET_PATH / "labels"

    if not images_dir.is_dir():
        raise RuntimeError(f"Active images dir not reachable: {images_dir}")
    if not labels_dir.is_dir():
        raise RuntimeError(f"Active labels dir not reachable: {labels_dir}")

    images = _list_images(images_dir)
    labels = _list_labels(labels_dir)

    image_count = len(images)
    label_count = len(labels)

    if image_count == 0:
        raise RuntimeError(f"No images found under {images_dir}.")
    if label_count == 0:
        raise RuntimeError(f"No label files found under {labels_dir}.")
    if image_count != label_count:
        raise RuntimeError(
            f"Image/label count mismatch: {image_count} images vs "
            f"{label_count} labels."
        )

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    images_without_label = sorted(image_stems - label_stems)
    labels_without_image = sorted(label_stems - image_stems)

    if images_without_label or labels_without_image:
        msg_parts: list[str] = []
        if images_without_label:
            sample = ", ".join(images_without_label[:5])
            msg_parts.append(
                f"{len(images_without_label)} image(s) without matching "
                f"label (e.g. {sample})"
            )
        if labels_without_image:
            sample = ", ".join(labels_without_image[:5])
            msg_parts.append(
                f"{len(labels_without_image)} label(s) without matching "
                f"image (e.g. {sample})"
            )
        raise RuntimeError(
            "Stem mismatch in active dataset: " + "; ".join(msg_parts)
        )

    return image_count, label_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Restore dataset package")
    print("=" * 60)

    if not PACKAGE_ZIP.exists():
        raise FileNotFoundError(
            f"Missing DVC package zip: {PACKAGE_ZIP}. "
            "Run `dvc pull datasets/table_assistant_yolo_package.zip.dvc` "
            "from the repo root first."
        )

    zip_size_gb = PACKAGE_ZIP.stat().st_size / (1024 ** 3)
    print(f"package zip:    {PACKAGE_ZIP}")
    print(f"zip size:       {zip_size_gb:.2f} GB")
    print(f"extract dir:    {PACKAGE_DIR}")
    print(f"active path:    {ACTIVE_DATASET_PATH}")

    # Wipe leftovers so the operation is reproducible.
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    _remove_path(ACTIVE_DATASET_PATH)

    # Extract.
    safe_extract_zip(PACKAGE_ZIP, get_datasets_dir())
    _validate_extracted_layout()
    print("\nExtraction OK.")

    # Symlink so the rest of the project addresses the canonical path.
    _link_active_dataset(PACKAGE_DATASET_DIR, ACTIVE_DATASET_PATH)
    if not ACTIVE_DATASET_PATH.is_symlink():
        raise RuntimeError(
            f"Expected {ACTIVE_DATASET_PATH} to be a symlink after creation."
        )
    print(f"symlink:        {ACTIVE_DATASET_PATH} -> {PACKAGE_DATASET_DIR}")

    # Validate the active dataset reachable through the symlink.
    image_count, label_count = _validate_active_dataset()
    print("\nActive dataset:")
    print(f"  images: {image_count}")
    print(f"  labels: {label_count}")
    print("\nDataset package restored successfully.")


if __name__ == "__main__":
    main()
