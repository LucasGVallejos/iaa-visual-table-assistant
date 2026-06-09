# Auto-labeling Open Images (v2)

## Why this exists

The Open Images V7 subset is exported with only the ~10 source object classes
we requested (bottle, bowl, cup variants, cutlery, plate). It has **no `food`
boxes**, even though many tabletop scenes contain food. Rather than re-download
or hand-label, we enrich the existing COCO export by running a strong
pretrained COCO-80 detector over the images and **injecting** its detections as
new annotations — most importantly a new `Food` category, plus reinforcement of
the cutlery/tableware classes.

The result is a second iteration of the subset, referred to as **v2**, that
notebook `01_dataset_prep_colab.ipynb` consumes instead of v1.

This is a pseudo-labeling step. The injected boxes are model predictions, not
human annotations; they trade some label noise for much broader `food`
coverage. The downstream class imbalance and quality trade-offs are tracked in
`reports/dataset_notes.md`.

## Key invariants

- **v1 is read-only.** The raw export at
  `datasets/raw_datasets/open_images_subset/labels.json` is never modified. The
  enriched document is written to a separate
  `open_images_subset_v2/labels.json`, and the enrichment pass refuses to run
  if the output path resolves to the v1 source.
- **Images are reused, not copied.** v2 references the v1 `data/` images by
  their original `file_name`. Only a new `labels.json` is produced until the
  packaging step bundles both into a portable zip.
- **Name-based mapping is the contract.** Injected annotations carry a COCO
  *category name* (e.g. `"Food"`, `"Coffee cup"`), not a YOLO id. Notebook 01's
  `convert_open_images_to_yolo` later maps each annotation name → YOLO id via
  `configs/label_mapping.yaml`. This keeps v2 a valid, self-describing COCO
  document.

## Configuration

`configs/label_mapping.yaml` carries two relevant sections:

- `open_images` — maps the original Open Images source category names to YOLO
  target ids (used by the YOLO converter and to seed de-duplication).
- `coco_pretrained` — maps the pretrained detector's COCO-80 class names
  (lowercase, e.g. `wine glass`) to our 7 YOLO target ids. Names absent here
  are dropped during enrichment.

The reverse decision — which COCO category *name* to write for each target id
when injecting a box — lives in code as `TARGET_TO_COCO_CATEGORY` in
`src/data/auto_label/coco_target_mapping.py`. All those names except `Food`
already exist in the v1 export; `Food` is created by the enrichment pass.

## Modules (`src/data/auto_label/`)

| Module | Role |
|--------|------|
| `coco_io.py` | COCO read/inject/write primitive: `load_coco`, `save_coco` (compact), `ensure_category` (idempotent), `add_annotation`, id helpers. |
| `coco_target_mapping.py` | Pure logic (no torch): loads the `coco_pretrained` mapping, maps detector names → target ids, and provides geometry helpers (`xywh_to_xyxy`, `iou`, `is_duplicate`). |
| `prepare_open_images_input.py` | Extract the raw Open Images zip from Drive, inspect the COCO export, and render a few raw samples to confirm it reads correctly. Read-only. |
| `preview_detections.py` | Run the pretrained detector over a few images and print/render what *would* be injected (kept per target class, dropped names) — no writes. Used to pick `--conf`. |
| `auto_label_open_images.py` | The enrichment pass: run the detector over the subset, de-duplicate against existing boxes, inject surviving detections into a v2 `labels.json`, write a JSON report. |
| `verify_autolabel.py` | Render before/after panels for images that gained boxes, so precision can be eyeballed before a full run. Read-only. |
| `package_open_images_v2.py` | Bundle the v2 `labels.json` + the v1 images it references into one portable zip mirroring the v1 layout, with strict pre/post validation. |

## Workflow

The notebook `notebooks/0.5_auto_label_open_images_colab.ipynb` orchestrates
these steps; the phase numbers in the module docstrings refer to its sections.

1. **Prepare input (phase 3)** — extract and inspect the raw v1 export, render
   a few raw samples.
   ```bash
   python -m src.data.auto_label.prepare_open_images_input --samples 3
   # local, against an already-extracted dir:
   python -m src.data.auto_label.prepare_open_images_input --skip-extract --samples 3
   ```
2. **Preview detections (phase 5)** — eyeball detector quality and tune `--conf`.
   Renders to `outputs/auto_label_checks/phase5_detections/`.
   ```bash
   python -m src.data.auto_label.preview_detections --samples 5 --conf 0.4
   ```
3. **Enrich — dry run, then write (phase 6)** — count what would be added, then
   write the v2 `labels.json`. Writes a report to `reports/auto_label_report.json`.
   ```bash
   python -m src.data.auto_label.auto_label_open_images --limit 50 --dry-run
   python -m src.data.auto_label.auto_label_open_images            # all images
   ```
4. **Verify (phase 6)** — before/after panels for added boxes, to
   `outputs/auto_label_checks/phase6_verify/`.
   ```bash
   python -m src.data.auto_label.verify_autolabel --samples 8
   ```
5. **Package (phase 8)** — bundle v2 into a portable zip mirroring the v1 layout
   (`labels.json` at root + `data/<file_name>` entries).
   ```bash
   python -m src.data.auto_label.package_open_images_v2 --check-only
   python -m src.data.auto_label.package_open_images_v2
   ```
6. **Upload** the resulting zip to Drive under
   `raw_datasets/open_images_subset_v2/`, and the v2 `labels.json` as
   `raw_datasets/labels_v2_full.json` (uploaded as a notebook cell, not a
   script). From there `setup_colab_raw_datasets.py` extracts the v2 zip and
   overrides the in-zip `labels.json` with the full `labels_v2_full.json`
   during raw setup (see `docs/environment-and-data-setup.md`).

## How enrichment decides what to inject

For each image the pass:

1. Collects the v1 boxes that already map to a target class (so a detection on
   top of an existing box is not injected twice).
2. Runs the detector in chunks of `--chunk-size` images (the chunk size is the
   effective inference batch — passing the whole ~11k list at once would try to
   allocate a ~50 GiB tensor and crash).
3. Processes detections by confidence descending. For each: maps the detector
   name → target id (drops unmapped names), clips to image bounds, drops
   degenerate boxes (≤1px), and drops same-class duplicates via IoU
   (`--iou-dedup`, default 0.5).
4. Injects each surviving box with its target category name plus a
   `score` and `source="auto_label"` tag for provenance.

## Notable parameters (`auto_label_open_images.py`)

- `--weights` (default `yolov8x.pt`) — pretrained detector; weights are cached
  under `models/` (gitignored), never the repo root.
- `--conf` (0.4) — detector confidence threshold.
- `--iou-dedup` (0.5) — IoU at/above which a same-class detection duplicates an
  existing box.
- `--chunk-size` (32) — images per `predict()` call; bounds memory.
- `--limit` / `--seed` — process a seeded random sample instead of all images.
- `--dry-run` — compute and report counts without writing v2.

## Outputs

- `datasets/raw_datasets/open_images_subset_v2/labels.json` — enriched COCO doc.
- `datasets/open_images_table_objects_v2_coco.zip` — portable v2 package.
- `reports/auto_label_report.json` — per-class added/skipped/dropped counts.
- `outputs/auto_label_checks/{phase3_raw,phase5_detections,phase6_verify}/` —
  sanity-check renders.
