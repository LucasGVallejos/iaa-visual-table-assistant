# Phase 1 — Dataset Preparation Plan

This document is the operational plan for converting the two raw source
datasets into a single YOLO-format dataset ready for training. It captures
the user's decisions, the staged execution order, and the gotchas to watch.

It supersedes ad-hoc instructions in earlier docs for this step. When this
plan completes, the produced dataset becomes the input for the training
pipeline and this document can be moved to an archive.

---

## 1. Goal of this step

Produce a single directory `datasets/table_assistant_yolo/` with a flat
YOLO layout:

```
datasets/table_assistant_yolo/
├── images/   ← all .jpg files, flat (no split subdirs)
└── labels/   ← all .txt files in YOLO format (one per image), flat
```

Labels follow the YOLO convention: one line per object, format
`class_id cx cy w h` with all bbox values normalized to `[0, 1]`. The class
IDs match `configs/data_runtime_colab.yaml`:

```
0=food  1=cup  2=bottle  3=plate  4=spoon  5=fork  6=knife
```

**No train/val/test split is performed in this step.** Splits, balancing,
and any other training-time concerns are deferred. The goal is strictly:
produce the flat YOLO directory plus a manifest.

`prepare_dataset.py` currently scaffolds `images/{train,val,test}` and
`labels/{train,val,test}` subdirs; it will be adapted (or replaced) to
scaffold the flat layout.

---

## 2. Top-level rules

These apply across all stages and override any later instruction.

### DVC confirmation gate

**No DVC-mutating command runs without explicit user confirmation in the
same conversation turn.** This includes:

- `dvc add`, `dvc commit`, `dvc remove`, `dvc unprotect`
- `dvc push`, `dvc pull`, `dvc fetch`
- `dvc gc`, `dvc destroy`, any other write/delete operation
- Any change to `.dvc/config` or `.dvcignore`

Read-only DVC commands (`dvc status`, `dvc list`, `dvc diff`) are fine to
run without confirmation.

A confirmation in a previous conversation does NOT carry over. When in
doubt, ask first.

---

## 3. User decisions (authoritative)

These were settled in conversation. Treat them as fixed unless the user
revisits them.

### Decision 1 — Images without any of our classes
**Discard them.** After applying the class mapping and filtering, if an
image ends up with zero valid YOLO labels, skip the image entirely (no
label file written, no image copied).

### Decision 2 — UEC FOOD-256 volume
**Convert everything.** No balancing or subsampling. All ~32K UEC images
with valid bboxes get converted. Class imbalance is accepted at this
stage; DVC versioning lets us revisit subsampling later without
re-converting.

### Decision 3 — File naming
**Embed sequential number + sorted classes in the filename.** Pattern:

```
<NNNNNNNN>_<class_a>_<class_b>_..._<class_n>.jpg
```

Where:
- `<NNNNNNNN>` is an 8-digit zero-padded sequential number, monotonically
  increasing across the whole dataset. Open Images images are processed
  first, then UEC, so the manifest is human-scannable by source. Counter
  starts at `00000001`. 8 digits leaves room for future source datasets.
- `<class_*>` is the YOLO class name (from `configs/data_runtime_colab.yaml`), included
  once per unique class present in the image's labels. An image with 3
  knife bboxes still gets `_knife` once.
- Classes are ordered by **class ID ascending**: food (0), cup (1),
  bottle (2), plate (3), spoon (4), fork (5), knife (6). This guarantees
  the same content always produces the same filename, regardless of
  bbox order in the source.
- Extension: `.jpg`.

Examples:
- `00000001_food.jpg` — UEC image, single food bbox
- `00000123_food_fork_knife.jpg` — Open Images with food + fork + knife
- `00000877_bottle_cup.jpg` — Open Images with bottle + cup only

The filename embeds *content* (which classes are in the image). The
manifest (Decision 4) embeds *provenance* (where the image came from).
Both are needed.

### Decision 4 — Manifest
A manifest file at `reports/dataset_manifest.csv` MUST map each new name
back to its source and original path:

```csv
new_name,source,original_relative_path,original_id
00000001_food.jpg,uec_food_256,1/100.jpg,
00000877_bottle_cup.jpg,open_images,data/0a1b2c3d.jpg,42
```

Without this manifest, debugging "why does the model fail on image
00004217?" becomes impossible (the filename tells you the *classes*, not
the source).

### Decision 5 — Image format
**RGB JPEG, quality 95, original resolution.** All output images are
re-encoded to RGB JPEG at quality 95, regardless of source format:
- **RGB (3-channel)**: strips any RGBA/grayscale source weirdness at the
  source. Every ML framework expects 3-channel input.
- **JPEG q95**: visually lossless for training purposes. q90 starts
  producing visible artifacts on flat regions; q100 wastes bytes for no
  gain.
- **Original resolution**: do not resize. Train-time augmentation decides
  `imgsz`; downscaling now is irreversible and upscaling is wasted bytes.

Accept the small generation-loss artifact from re-encoding already-JPEG
sources. The benefit (uniform format, guaranteed RGB) outweighs the cost.

### Decision 6 — Workflow
**Build locally, validate locally. DVC step requires explicit user
confirmation per §2.** Concretely:

1. Run the full pipeline (Etapas A–E) on the local machine using the
   already-extracted raw datasets.
2. Run `validate_dataset.py` against the produced flat YOLO directory.
3. Stop. Show the user the final layout, the manifest summary, and the
   validation report. The user then decides whether to proceed to
   Etapa F (DVC).
4. Etapa F runs only after the user explicitly approves the DVC
   operations in that turn.

Do not delete the local raw datasets at any point during this step —
they remain the recovery source if regeneration is needed.

---

## 4. Staged execution order

The work is broken into discrete stages, each producing something
verifiable. Do not skip ahead; each stage's output feeds the next.

### Etapa A — Materialize the mapping in `configs/label_mapping.yaml`

Currently the source-to-target mapping lives in prose in
`docs/architecture.md`. Make it machine-readable:

```yaml
# configs/label_mapping.yaml
# Source class -> YOLO class ID (defined in configs/data_runtime_colab.yaml).
# Source class names that do not appear here are dropped.

open_images:
  Bottle: 2
  "Coffee cup": 1
  "Wine glass": 1
  Bowl: 3
  Plate: 3
  "Mixing bowl": 3
  Spoon: 4
  Fork: 5
  Knife: 6
  "Kitchen knife": 6

uec_food_256:
  # All 256 numeric categories collapse to the food class.
  default: 0
```

In `src/data/convert_to_yolo.py`, rename the existing helper to clarify
the two distinct concepts:
- `load_target_classes(...)` reads `configs/classes.yaml` and returns
  `{class_name: class_id}` for the 7 YOLO classes.
- `load_source_to_target_mapping(...)` reads `configs/label_mapping.yaml`
  and returns `{source_dataset: {source_label: target_class_id}}`.

Update existing callers of `load_class_mapping` to use the new names.
`configs/classes.yaml` stays as-is — it describes target classes; the
new `label_mapping.yaml` describes how sources reach targets.

### Etapa B — Analysis script (read-only)

Create `src/data/analyze_raw_datasets.py`. The script:

- Loads `configs/label_mapping.yaml`.
- Walks the extracted Open Images COCO JSON and counts:
  - Images total.
  - Annotations per source category.
  - Annotations per YOLO target class (after applying the mapping).
  - Images that would be discarded under Decision 1.
  - Distribution of bbox sizes (min/median/max width and height, normalized).
- Walks UEC FOOD-256 and counts:
  - Categories present (should be 256).
  - Images per category.
  - Annotations per category (some images have multiple bboxes).
  - Total images and total annotations.
- Prints a summary table to stdout and writes a JSON report at
  `reports/raw_dataset_analysis.json`.

The JSON report MUST use this exact schema (pinned to prevent ad-hoc
drift across runs):

```json
{
  "open_images": {
    "json_path": "<string>",
    "images_total": 0,
    "images_with_valid_mapping": 0,
    "images_discarded_no_mapping": 0,
    "annotations_total": 0,
    "annotations_by_source_label": { "Bottle": 0, "Coffee cup": 0 },
    "annotations_by_target_class":  { "0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0 },
    "bbox_normalized_size_stats": {
      "w_min": 0.0, "w_p50": 0.0, "w_max": 0.0,
      "h_min": 0.0, "h_p50": 0.0, "h_max": 0.0
    }
  },
  "uec_food_256": {
    "uec_root": "<string>",
    "categories_count": 0,
    "images_total": 0,
    "annotations_total": 0,
    "images_per_category": { "1": 0, "2": 0 }
  }
}
```

This stage produces no dataset output — only information. Run it before
Etapa C as a sanity check.

### Etapa C — Per-source conversion to staging

Create `src/data/convert_open_images_to_yolo.py` and
`src/data/convert_uec_food_to_yolo.py`. Each writes to a per-source
staging directory:

```
datasets/_staging/
├── open_images/
│   ├── images/   ← re-encoded JPG copies (NOT symlinks)
│   └── labels/   ← .txt YOLO labels
└── uec_food/
    ├── images/
    └── labels/
```

Per-source staging avoids filename collisions between sources before the
final merge in Etapa D. Images are real copies (not symlinks) because the
final dataset must be portable.

For Open Images:
- Read the COCO JSON via `setup_colab_raw_datasets.py`'s inspector pattern.
- For each image, group its annotations.
- For each annotation, look up the source category name and map to a YOLO
  class id via `label_mapping.yaml`. Drop annotations with no mapping.
- Convert each surviving bbox via `coco_to_yolo()` from
  `src/data/convert_to_yolo.py`.
- Clip coordinates to `[0, 1]` to absorb rounding artifacts at image edges.
- Skip images that end up with zero valid annotations (Decision 1).
- Filter out degenerate bboxes where `w == 0` or `h == 0` after conversion.
- Image filename in staging: original Open Images filename (renaming to
  the final scheme happens in Etapa D, not here).
- Re-encode every output image to RGB JPEG q95 per Decision 5, even when
  the source is already JPEG.

For UEC FOOD-256:
- Walk each numbered category directory.
- Read `bb_info.txt` (skip the `img x1 y1 x2 y2` header line if present;
  robust approach: try to parse every line, catch `ValueError` on the
  first token, skip).
- For each bbox line, parse `img x1 y1 x2 y2`.
- Read image dimensions with PIL: `Image.open(path).size` returns `(W, H)`.
  Watch the W/H order — PIL gives `(W, H)` while numpy/cv2 give
  `(H, W, C)`. This is a classic silent-bug source.
- Convert via `voc_to_yolo()` (the format is already `[x1, y1, x2, y2]`).
- All UEC labels get class id `0` (food).
- Clip and filter degenerate boxes as above.
- Aggregate per image: an image with multiple bboxes gets a `.txt` with
  multiple lines.
- Image filename in staging: `<3-digit-category>_<imgid>.jpg`
  (e.g., `001_100.jpg`) to disambiguate the cross-category `100.jpg`
  collisions. Re-encode to RGB JPEG q95.

Robustness requirements for both converters:
- Wrap every image-open call in `try/except`. If PIL or the converter
  raises, append a row to `reports/skipped_images.csv` (columns:
  `source,original_path,reason`) and continue. Hard-fail only if > 1% of
  a source is unreadable.
- Idempotent: re-running with the same input wipes the staging output
  cleanly and rewrites. Do not silently merge with a stale staging dir.

After Etapa C, run `src/data/validate_dataset.py` against each staging
dir (parametrized with a custom base path; see Etapa E note).

### Etapa D — Merge staging into final flat layout

Create `src/data/merge_and_rename.py`. This script:

- Walks both staging dirs. **Open Images is processed first, then UEC**,
  so the manifest is human-scannable by source.
- For each (image, label) pair:
  - Read the label file to extract the unique set of class IDs present.
  - Translate class IDs → class names via `configs/data_runtime_colab.yaml`.
  - Compute the new filename per Decision 3 (8-digit seq + classes sorted
    by ID ascending + `.jpg`).
  - Copy the image to `datasets/table_assistant_yolo/images/<new_name>`.
  - Copy the label to `datasets/table_assistant_yolo/labels/<new_name>.txt`
    (label content is unchanged; only the filename stem changes).
  - Append a row to `reports/dataset_manifest.csv` with columns
    `new_name,source,original_relative_path,original_id`.
- Sequence counter starts at 1; increments per output pair.
- Idempotent: wipe `datasets/table_assistant_yolo/images/`,
  `datasets/table_assistant_yolo/labels/`, and the manifest CSV before
  writing. Do not append to a stale manifest.
- Deterministic ordering: within each source, iterate files in sorted
  filename order so two runs produce identical sequence numbers.

### Etapa E — Validate final layout

`src/data/validate_dataset.py` currently iterates `train/val/test`
subdirs. Adapt it (or add a flat-layout wrapper) to validate:

- Every image in `datasets/table_assistant_yolo/images/` has a matching
  `labels/<stem>.txt`.
- Every label has valid YOLO rows: 5 fields, class ID in `[0, 7)`, all
  bbox values in `[0, 1]`.
- No empty `.txt` files (Decision 1 means zero-label images should be
  absent entirely; an empty `.txt` indicates a bug).

Also: write a per-class instance count table to `reports/dataset_notes.md`.
The current table there has `Train|Val|Test` columns; replace those with
a single `Count` column since this plan no longer produces splits.

Spot-check 3–5 random images visually using `visualize_raw_bboxes.py`
adapted to read from the flat layout (small extension).

### Etapa F — DVC versioning (REQUIRES USER CONFIRMATION per §2)

Do not start this stage without an explicit "go" from the user in the
current turn. Etapas A–E produce a valid local dataset; Etapa F is the
distribution and versioning layer on top.

When approved:
- `dvc add datasets/table_assistant_yolo/`
- Commit the resulting `datasets/table_assistant_yolo.dvc` file to git
  (the dataset itself stays gitignored).
- `dvc push` to the configured Google Drive remote.
- Verify the push by inspecting `.dvc/cache/` size and the remote.

### Cleanup (separate script, runs only after the user confirms Etapa E is fine)

Create `src/data/clean_intermediates.py`. Wipes `datasets/_staging/`.
Does NOT touch:
- the final `datasets/table_assistant_yolo/`
- `local_data/` (raw downloaded datasets)
- `reports/` (manifests, analysis JSON, skipped images log)

Run only after the user has confirmed the final layout is correct.

---

## 5. Gotchas

These are the things that bite people doing this for the first time. Watch
them actively while implementing.

- **Image W/H order**: PIL returns `(W, H)` from `Image.size`, numpy/cv2
  return `(H, W, C)` from `.shape`. UEC FOOD-256 conversion requires
  reading actual image dimensions; mixing these up swaps coordinates and
  produces silently invalid labels that pass numerical validation but
  produce a broken model.
- **COCO category IDs are not 0-indexed and not contiguous**. Do not
  assume `category_id == YOLO class id`. Always look up the category
  *name* in the COCO JSON, then map name → YOLO id via
  `label_mapping.yaml`.
- **Open Images label names are case-sensitive**: the JSON has `Bottle`
  with a capital B, not `bottle`. Match exactly.
- **Filename collisions across sources**: pre-renaming, two source
  datasets may both have files named `001.jpg`. Working in per-source
  staging dirs (Etapa C) avoids the collision; the new naming scheme
  in Etapa D finalizes it globally.
- **Degenerate bboxes**: occasionally a source has `w=0` or `h=0` from
  rounding. These pass the `[0,1]` numerical check in
  `validate_dataset.py` but train poorly. Filter them in the converters,
  before writing.
- **Coordinates slightly outside `[0, 1]`**: source data sometimes has
  bboxes that touch or marginally exceed image edges. Clip to `[0, 1]`
  rather than reject.
- **Empty label files vs missing label files**: in YOLO, a present-but-
  empty `.txt` means "this image has no objects" (negative example).
  A missing `.txt` is an error. Decision 1 says we drop those images
  entirely, so neither case should occur in our output; if either
  appears, it's a bug.
- **Re-encoding JPEG quality loss**: re-encoding an already-JPEG image
  to q95 introduces a tiny generation-loss artifact. Accepted as the
  price of uniform format (RGB, single quality setting).
- **Filename determinism**: the merge step's sequence numbers depend on
  iteration order. Sort the staging file list by name before assigning
  numbers so two runs on the same staging dir produce identical
  filenames.
- **Corrupted source images**: wrap PIL opens in try/except. Log to
  `reports/skipped_images.csv` and continue. Don't crash the whole
  pipeline for a single bad file.
- **Idempotency**: every script should be safe to re-run. Wipe outputs
  before re-writing; do not silently append.
- **`configs/data_runtime_colab.yaml` mismatch**: `data_runtime_colab.yaml` currently points at
  `images/train`, `images/val`, `images/test`. The flat layout this plan
  produces does not have those subdirs. Updating `data_runtime_colab.yaml` (or
  producing a split before training) is a training-time concern, not
  dataset-prep. Flag it when starting the training workstream.

---

## 6. What "done" looks like for this step

All of these must be true before considering this step complete:

- [ ] `configs/label_mapping.yaml` exists and matches the table in
  `docs/architecture.md`. _(Etapa A)_
- [ ] `reports/raw_dataset_analysis.json` exists and matches the schema
  pinned in §4 Etapa B. _(Etapa B)_
- [ ] `datasets/table_assistant_yolo/` has the flat layout populated:
  `images/*.jpg` + `labels/*.txt`, one label per image, no split
  subdirs. _(Etapas C + D)_
- [ ] `reports/dataset_manifest.csv` traces every renamed file back to
  its source. _(Etapa D)_
- [ ] `reports/skipped_images.csv` exists (may be empty if nothing was
  skipped). _(Etapas C + D)_
- [ ] `src/data/validate_dataset.py` (adapted for flat layout) reports
  zero errors. _(Etapa E)_
- [ ] Per-class counts in `reports/dataset_notes.md` are filled in
  (no `-` placeholders); the table columns adapted to a single `Count`
  column. _(Etapa E)_
- [ ] User has reviewed the local dataset before any DVC step runs.
  _(Decision 6 + §2 DVC gate)_

Etapa F's DVC artifacts (`.dvc` file, remote push) are explicitly NOT
required for "done"; they are a separate, user-gated step.

---

## 7. Open questions to discuss before writing the corresponding stage

Bring these up with the user when reaching the relevant stage; do not
guess.

- **Etapa E table format**: in `reports/dataset_notes.md`, replace the
  `Train|Val|Test` columns with a single `Count` column, or leave the
  table alone and write counts elsewhere? Decide when running Etapa E.
- **Etapa F DVC operations**: confirm the explicit "go" before each of
  `dvc add` and `dvc push`. The default per §2 is to ask first.
