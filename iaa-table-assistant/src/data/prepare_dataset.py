"""
Prepare the local dataset directory structure for Phase 1.

This module creates the expected YOLO dataset folders for the
visual table assistant project. It does not download or convert data.
"""

from pathlib import Path


DATASET_DIR = Path("datasets/table_assistant_yolo")


def prepare_directories(base_dir: Path = DATASET_DIR) -> dict[str, Path]:
    """
    Create the required YOLO dataset directory structure.

    Expected structure:
    datasets/table_assistant_yolo/
      images/train
      images/val
      images/test
      labels/train
      labels/val
      labels/test
    """
    dirs: dict[str, Path] = {}

    for split in ["train", "val", "test"]:
        image_path = base_dir / "images" / split
        label_path = base_dir / "labels" / split

        image_path.mkdir(parents=True, exist_ok=True)
        label_path.mkdir(parents=True, exist_ok=True)

        dirs[f"images_{split}"] = image_path
        dirs[f"labels_{split}"] = label_path

    return dirs


def main() -> None:
    dirs = prepare_directories()

    print("Created/verified dataset directories:")
    for name, path in dirs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()