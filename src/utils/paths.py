"""
Path utilities for consistent file and directory resolution.

These helpers avoid hardcoded paths and keep project paths centralized.

Dataset layout (centralized under ``datasets/``)::

    datasets/
    ├── raw_datasets/          # extracted raw source datasets
    ├── _staging/              # intermediate per-source YOLO conversions
    └── table_assistant_yolo/  # final trainable YOLO dataset (DVC-tracked later)
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


def get_config_path(config_name: str) -> Path:
    """Return the path to a configuration file."""
    return get_project_root() / "configs" / config_name


# ---------------------------------------------------------------------------
# Datasets root and subtrees
# ---------------------------------------------------------------------------
def get_datasets_dir() -> Path:
    """Return the root datasets directory (``<repo>/datasets``)."""
    return get_project_root() / "datasets"


def get_raw_datasets_dir() -> Path:
    """Directory holding extracted raw source datasets.

    ``datasets/raw_datasets/`` contains one subfolder per source dataset
    (e.g. ``open_images_subset/``, ``uec_food_256/``).
    """
    return get_datasets_dir() / "raw_datasets"


def get_staging_dir() -> Path:
    """Per-source staging area used during dataset conversion.

    ``datasets/_staging/`` contains intermediate YOLO conversions, one
    subfolder per source dataset.
    """
    return get_datasets_dir() / "_staging"


def get_table_assistant_yolo_dir() -> Path:
    """Final trainable YOLO dataset directory.

    ``datasets/table_assistant_yolo/`` is the merged + split dataset
    consumed by training, and will be DVC-tracked once validated.
    """
    return get_datasets_dir() / "table_assistant_yolo"


def get_yolo_images_path(split: str) -> Path:
    """Return the image directory for a YOLO dataset split."""
    return get_table_assistant_yolo_dir() / "images" / split


def get_yolo_labels_path(split: str) -> Path:
    """Return the label directory for a YOLO dataset split."""
    return get_table_assistant_yolo_dir() / "labels" / split


# ---------------------------------------------------------------------------
# Raw datasets — per-source helpers
# ---------------------------------------------------------------------------
def get_open_images_dataset_original_dir() -> Path:
    """Local directory where the Open Images COCO export lives.

    Expected layout::

        <dir>/labels.json
        <dir>/data/<image_files>
    """
    return get_raw_datasets_dir() / "open_images_subset"


def get_uec_food_dataset_extract_dir() -> Path:
    """Directory where the UEC FOOD-256 zip is extracted.

    The zip unpacks into a nested ``UECFOOD256/`` folder; use
    :func:`get_uecfood256_dataset_original_dir` to address that inner root.
    """
    return get_raw_datasets_dir() / "uec_food_256"


def get_uecfood256_dataset_original_dir() -> Path:
    """Original UEC FOOD-256 dataset root (the inner ``UECFOOD256/`` folder)."""
    return get_uec_food_dataset_extract_dir() / "UECFOOD256"


# ---------------------------------------------------------------------------
# Staging — per-source helpers
# ---------------------------------------------------------------------------
def get_uec_staging_dir() -> Path:
    """Staging directory for the UEC FOOD-256 → YOLO conversion."""
    return get_staging_dir() / "uec_food"


def get_open_images_staging_dir() -> Path:
    """Staging directory for the Open Images COCO → YOLO conversion."""
    return get_staging_dir() / "open_images"


# ---------------------------------------------------------------------------
# Models, outputs, reports
# ---------------------------------------------------------------------------
def get_models_dir() -> Path:
    """Return the models directory."""
    return get_project_root() / "models"


def get_model_path(model_name: str) -> Path:
    """Return the path to a model artifact."""
    return get_models_dir() / model_name


def get_outputs_dir() -> Path:
    """Return the outputs directory."""
    return get_project_root() / "outputs"


def get_output_path(run_name: str) -> Path:
    """Return the path to a specific output run."""
    return get_outputs_dir() / run_name


def get_reports_dir() -> Path:
    """Return the reports directory."""
    return get_project_root() / "reports"


# ---------------------------------------------------------------------------
# DVC dataset package (pulled + restored)
# ---------------------------------------------------------------------------
def get_dataset_package_dir() -> Path:
    """Directory the DVC dataset package extracts into.

    ``restore_dataset_package`` unpacks ``table_assistant_yolo_package.zip``
    here, producing ``table_assistant_yolo/{images,labels}`` plus a
    ``metadata/`` subtree, and symlinks ``datasets/table_assistant_yolo`` to
    the inner dataset dir.
    """
    return get_datasets_dir() / "table_assistant_yolo_package"


def get_package_manifest_path() -> Path:
    """Path to the populated provenance manifest inside the dataset package.

    This is the authoritative manifest written into the package metadata. The
    repo's ``reports/dataset_manifest.csv`` is an empty header-only stub; use
    this one (available only after pulling + restoring the package).
    """
    return get_dataset_package_dir() / "metadata" / "dataset_manifest.csv"
