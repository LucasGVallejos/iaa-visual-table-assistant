"""
Path utilities for consistent file and directory resolution.

These helpers avoid hardcoded paths and keep project paths centralized.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


def get_config_path(config_name: str) -> Path:
    """Return the path to a configuration file."""
    return get_project_root() / "configs" / config_name


def get_datasets_dir() -> Path:
    """Return the root datasets directory."""
    return get_project_root() / "datasets"


def get_table_assistant_dataset_path() -> Path:
    """Return the prepared YOLO dataset directory for the table assistant."""
    return get_datasets_dir() / "table_assistant_yolo"


def get_yolo_images_path(split: str) -> Path:
    """Return the image directory for a YOLO dataset split."""
    return get_table_assistant_dataset_path() / "images" / split


def get_yolo_labels_path(split: str) -> Path:
    """Return the label directory for a YOLO dataset split."""
    return get_table_assistant_dataset_path() / "labels" / split


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

def get_staging_dir() -> Path:
  """Per-source staging area used during dataset conversion."""
  return get_datasets_dir() / "_staging"

def get_uec_staging_dir() -> Path:
  """Staging directory for the UEC FOOD-256 → YOLO conversion."""
  return get_staging_dir() / "uec_food"

def get_uecfood256_dataset_original_dir() -> Path:
  """Original UEC FOOD-256 dataset directory."""
  return get_datasets_dir() / "UECFOOD256"


def get_reports_dir() -> Path:
  """Return the reports directory."""
  return get_project_root() / "reports"


def get_skipped_images_csv_path() -> Path:
  """Return the path to the skipped images log CSV (shared across sources)."""
  return get_reports_dir() / "skipped_images.csv"
