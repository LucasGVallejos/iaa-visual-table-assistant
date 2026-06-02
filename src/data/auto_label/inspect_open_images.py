"""
Render a few Open Images samples from the restored dataset with their current boxes.

A read-only visual sanity check before any auto-labeling: it picks N random
Open Images stems from the package manifest, draws each image's existing YOLO
boxes (from the flat ``labels/<stem>.txt``), and saves the PNGs under
``outputs/auto_label_checks/phase3_existing/``.

Run with::

    python -m src.data.auto_label.inspect_open_images
    python -m src.data.auto_label.inspect_open_images --samples 5 --seed 7
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.data.auto_label.manifest import select_open_images_stems
from src.data.common.convert_to_yolo import load_classes_config
from src.data.common.dataset_io import parse_yolo_label_file
from src.data.validation.visualize_yolo_mapping import build_class_lookups, render_sample
from src.utils.paths import (
    get_outputs_dir,
    get_package_manifest_path,
    get_table_assistant_yolo_dir,
)


DEFAULT_SAMPLES = 3
DEFAULT_RANDOM_SEED = 42


def _output_dir() -> Path:
    return get_outputs_dir() / "auto_label_checks" / "phase3_existing"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.auto_label.inspect_open_images",
        description=(
            "Render random Open Images samples with their current YOLO boxes "
            "as a sanity check before auto-labeling."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=get_package_manifest_path(),
        help="Path to the populated package manifest. Default: %(default)s.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=get_table_assistant_yolo_dir(),
        help=(
            "Flat YOLO dataset dir holding images/ and labels/. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help="How many random Open Images samples to render. Default: %(default)s.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Seed for the random sampling; same seed + same manifest reproduce "
            "the same selection. Default: %(default)s."
        ),
    )
    args = parser.parse_args(argv)

    if args.samples < 1:
        parser.error("--samples must be >= 1")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print("=" * 60)
    print("Inspect Open Images samples (current boxes)")
    print("=" * 60)
    print(f"manifest:    {args.manifest}")
    print(f"dataset_dir: {args.dataset_dir}")
    print(f"samples:     {args.samples}")
    print(f"seed:        {args.seed}")

    class_entries = load_classes_config()
    id_to_name, id_to_color = build_class_lookups(class_entries)

    stems = select_open_images_stems(args.manifest)
    if not stems:
        print(
            "\nNOTE: no Open Images stems found. The manifest looks empty (the "
            "repo stub). Run `dvc pull "
            "datasets/table_assistant_yolo_package.zip.dvc` then "
            "`python -m src.data.preparation.restore_dataset_package` first."
        )
        return

    chosen = random.Random(args.seed).sample(stems, min(args.samples, len(stems)))
    chosen.sort()

    images_dir = args.dataset_dir / "images"
    labels_dir = args.dataset_dir / "labels"
    output_dir = _output_dir()
    print(f"output_dir:  {output_dir}")

    written: list[Path] = []
    for stem in chosen:
        image_path = images_dir / f"{stem}.jpg"
        label_path = labels_dir / f"{stem}.txt"

        if not image_path.exists():
            print(f"  WARN: image missing, skipping: {image_path}")
            continue
        if not label_path.exists():
            print(f"  WARN: label missing, skipping: {label_path}")
            continue

        annotations = parse_yolo_label_file(label_path)
        output_path = output_dir / f"{stem}.png"
        title = (
            f"{stem} — {len(annotations)} "
            f"box{'es' if len(annotations) != 1 else ''} (current)"
        )
        render_sample(image_path, annotations, id_to_name, id_to_color, output_path, title)
        written.append(output_path)
        print(f"  wrote: {output_path}")

    print("\nGenerated samples:")
    if not written:
        print("  (none)")
        return
    for output_path in written:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
