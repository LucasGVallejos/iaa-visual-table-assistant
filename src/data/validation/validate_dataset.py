"""
Validate YOLO dataset integrity and format.

Checks for common issues:
- missing image directories;
- missing label directories;
- missing labels for selected images;
- invalid YOLO label rows;
- class IDs outside the configured range;
- bounding box values outside [0, 1].
"""

from pathlib import Path

import yaml


DATASET_DIR = Path("datasets/table_assistant_yolo")
DATA_CONFIG = Path("configs/data_runtime_colab.yaml")


def validate_label_file(label_path: Path, num_classes: int) -> list[str]:
    errors: list[str] = []

    with open(label_path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, 1):
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                errors.append(f"{label_path}:{line_num} - Expected 5 values, got {len(parts)}")
                continue

            try:
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(f"{label_path}:{line_num} - Invalid numeric values")
                continue

            if class_id < 0 or class_id >= num_classes:
                errors.append(f"{label_path}:{line_num} - Class ID {class_id} out of range [0, {num_classes})")

            for index, value in enumerate(values):
                if value < 0.0 or value > 1.0:
                    errors.append(
                        f"{label_path}:{line_num} - Value {value} at position {index + 1} out of [0, 1]"
                    )

    return errors


def validate_split(dataset_dir: Path, split: str, num_classes: int) -> dict:
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    results = {
        "total_images": 0,
        "total_labels": 0,
        "missing_labels": [],
        "errors": [],
    }

    if not images_dir.exists():
        results["errors"].append(f"Images directory not found: {images_dir}")
        return results

    if not labels_dir.exists():
        results["errors"].append(f"Labels directory not found: {labels_dir}")
        return results

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [image for image in images_dir.iterdir() if image.suffix.lower() in image_extensions]
    results["total_images"] = len(images)

    for image in images:
        label_file = labels_dir / image.with_suffix(".txt").name

        if not label_file.exists():
            results["missing_labels"].append(str(image.name))
            continue

        results["total_labels"] += 1
        results["errors"].extend(validate_label_file(label_file, num_classes))

    return results


def load_num_classes(config_path: Path = DATA_CONFIG) -> int:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return int(config["nc"])


def main() -> None:
    num_classes = load_num_classes()

    for split in ["train", "val", "test"]:
        print(f"\nValidating {split}...")
        results = validate_split(DATASET_DIR, split, num_classes)

        print(f"  Images: {results['total_images']}")
        print(f"  Labels: {results['total_labels']}")
        print(f"  Missing labels: {len(results['missing_labels'])}")
        print(f"  Errors: {len(results['errors'])}")

        if results["errors"]:
            print("  First errors:")
            for error in results["errors"][:10]:
                print(f"    - {error}")


if __name__ == "__main__":
    main()