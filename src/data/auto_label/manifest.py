"""
Read the provenance manifest of the restored dataset package.

The auto-labeling pass only touches Open Images frames, so it needs to know
which rows of the dataset came from that source. The source of truth is the
manifest written **inside** the DVC dataset package
(``datasets/table_assistant_yolo_package/metadata/dataset_manifest.csv``),
populated by the preparation pipeline. The repo copy at
``reports/dataset_manifest.csv`` is an empty header-only stub and must never
be used here.

Run with::

    python -m src.data.auto_label.manifest
    python -m src.data.auto_label.manifest --manifest /path/to/dataset_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.data.preparation.prepare_dataset import MANIFEST_HEADER
from src.utils.paths import get_package_manifest_path


OPEN_IMAGES_SOURCE = "open_images"
SAMPLE_NAMES_PRINTED = 5


def load_manifest_rows(manifest_path: Path) -> list[dict]:
    """Read every manifest row as a dict keyed by :data:`MANIFEST_HEADER`.

    Raises :class:`FileNotFoundError` (with recovery instructions) when the
    manifest is absent, and :class:`ValueError` when the header does not match
    the expected columns.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Pull and restore the "
            "dataset package first: run "
            "`dvc pull datasets/table_assistant_yolo_package.zip.dvc` then "
            "`python -m src.data.preparation.restore_dataset_package`."
        )

    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if header != MANIFEST_HEADER:
            raise ValueError(
                f"Unexpected manifest header in {manifest_path}: "
                f"{header} (expected {MANIFEST_HEADER})."
            )
        return list(reader)


def select_open_images_rows(rows: list[dict]) -> list[dict]:
    """Return only the rows whose ``source`` is ``open_images``."""
    return [row for row in rows if row["source"] == OPEN_IMAGES_SOURCE]


def select_open_images_stems(manifest_path: Path | None = None) -> list[str]:
    """Return the sorted, unique filename stems of Open Images rows.

    A stem is ``Path(row["new_name"]).stem`` (the ``.jpg`` suffix stripped),
    which is shared by the image and its ``.txt`` label in the flat layout.
    """
    path = manifest_path if manifest_path is not None else get_package_manifest_path()
    rows = load_manifest_rows(path)
    stems = {Path(row["new_name"]).stem for row in select_open_images_rows(rows)}
    return sorted(stems)


def summarize_manifest(manifest_path: Path) -> dict[str, int]:
    """Return per-source row counts plus a ``"total"`` key."""
    rows = load_manifest_rows(manifest_path)
    counts: dict[str, int] = {}
    for row in rows:
        source = row["source"]
        counts[source] = counts.get(source, 0) + 1
    counts["total"] = len(rows)
    return counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.auto_label.manifest",
        description=(
            "Summarize the dataset package manifest and list Open Images "
            "samples that the auto-labeling pass will densify."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=get_package_manifest_path(),
        help=(
            "Path to the populated package manifest. Default: %(default)s "
            "(the manifest inside the restored dataset package)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print("=" * 60)
    print("Dataset package manifest summary")
    print("=" * 60)
    print(f"manifest: {args.manifest}")

    counts = summarize_manifest(args.manifest)
    total = counts.get("total", 0)

    print("\nRows per source:")
    for source in sorted(k for k in counts if k != "total"):
        print(f"  {source:<13} {counts[source]}")
    print(f"  {'total':<13} {total}")

    if total == 0:
        print(
            "\nNOTE: the manifest has 0 rows. This looks like the empty repo "
            "stub. Run `dvc pull datasets/table_assistant_yolo_package.zip.dvc` "
            "then `python -m src.data.preparation.restore_dataset_package` to "
            "populate it."
        )
        return

    open_images_rows = select_open_images_rows(load_manifest_rows(args.manifest))
    print(f"\nOpen Images rows: {len(open_images_rows)}")
    sample = open_images_rows[:SAMPLE_NAMES_PRINTED]
    if sample:
        print(f"Sample new_name(s) (up to {SAMPLE_NAMES_PRINTED}):")
        for row in sample:
            print(f"  {row['new_name']}")


if __name__ == "__main__":
    main()
