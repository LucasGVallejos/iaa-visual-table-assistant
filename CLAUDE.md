# CLAUDE.md

This file is the entry point for Claude Code sessions in this repository. It
gives you the persistent context of the project so you can help without
re-discovering everything each time.

If you are reading this, you are operating inside the `iaa-visual-table-assistant`
repository. Before acting on any user request, skim this whole file. Then read
the referenced files only if relevant to the task at hand.

---

## 1. Project: IAA Visual Table Assistant

A vision-based assistant for visually impaired people. The system detects
common tabletop objects (food, cutlery, tableware) from a camera feed and
communicates what is on the table via audio.

This is an academic project ("IAA" = Inteligencia Artificial Aplicada, UTN
Buenos Aires). The current phase 1 is scoped tightly: dataset preparation,
training a YOLO detector, evaluation, experiment tracking with MLflow, dataset
and model versioning with DVC, and ONNX export.

**Phase 1 does NOT include**: real-time webcam capture, voice synthesis, UI,
interactive prototype, user testing. Those belong to phase 2.

The primary author is a senior software engineer with 10+ years in
JavaScript/TypeScript/Node and a Systems Engineering background. He has solid
fundamentals in networking, operating systems, and CS in general, but is new
to deep learning and computer vision specifics. He explicitly wants to
**learn, not be sold a solution**. When you respond:

- Explain the *why* behind decisions, not just the *how*.
- State confidence levels for non-trivial claims (e.g., "confidence: 90%").
- Cite sources for factual or technical claims when reasonable.
- Don't pad answers with hedging or excessive caveats; he prefers direct
  technical communication.
- Spanish is welcome for natural conversation; code, identifiers, comments,
  and commit messages stay in English.

---

## 2. Detection classes (frozen)

The final YOLO class set is defined in `configs/data_runtime_colab.yaml` and should NOT be
changed without explicit user confirmation:

| ID | Class      | Description                              |
|----|------------|------------------------------------------|
| 0  | food       | Any visible food item (generic, no type) |
| 1  | cup  | Cups, glasses, mugs                      |
| 2  | bottle     | Bottles                                  |
| 3  | plate | Plates, bowls                            |
| 4  | spoon      | Spoons                                   |
| 5  | fork       | Forks                                    |
| 6  | knife      | Knives                                   |

The system does NOT attempt to identify food type. All food categories
collapse to the single `food` class.

---

## 3. Source datasets

Two raw sources, already documented in `docs/architecture.md` and
`docs/environment-and-data-setup.md`:

- **Open Images V7** (subset, COCO format) for non-food classes. Downloaded
  locally (or on Colab) with FiftyOne via
  `src/data/raw_setup/download_open_images_subset.py`. A second iteration
  (`open_images_subset_v2`) re-labels the subset through an auto-labeling
  pass under `src/data/auto_label/` (see
  `docs/auto-label-open-images.md`). The full labels for v2 live on Drive as
  `labels_v2_full.json` and override the in-zip labels at setup time.
  Source classes: Bottle, Bowl, Coffee cup, Fork, Kitchen knife, Knife,
  Mixing bowl, Plate, Spoon, Wine glass.
- **UEC FOOD-256** for the `food` class. All 256 categories collapse to
  class id 0.

The mapping from source labels to YOLO class IDs is the machine-readable
`configs/label_mapping.yaml` (source of truth). The prose table in
`docs/architecture.md` mirrors it for human reference.

OCID was originally proposed but is OUT OF SCOPE for phase 1.

---

## 4. Repository layout (what lives where)

```
iaa-visual-table-assistant/
├── CLAUDE.md                       ← this file
├── README.md                       ← user-facing entry
├── configs/                        ← YAML configs (data, classes, training)
├── docs/                           ← architecture, setup, phase plans
├── notebooks/                      ← Colab notebooks (download, auto-label, prep, train, infer)
├── reports/                        ← experiment logs, dataset notes, baselines
├── src/
│   ├── data/
│   │   ├── raw_setup/              ← download, extract, visualize raw datasets
│   │   ├── auto_label/             ← Open Images auto-labeling (v2 generation)
│   │   ├── conversion/             ← per-source COCO/VOC → YOLO converters
│   │   ├── common/                 ← shared YOLO + dataset I/O helpers
│   │   ├── preparation/            ← merge, split, package, restore, notes
│   │   └── validation/             ← YOLO integrity + class distribution checks
│   ├── training/                   ← train, evaluate, MLflow (stubs)
│   ├── inference/                  ← ONNX export/validate, single-image predict (stubs)
│   └── utils/                      ← path and label helpers
├── datasets/                       ← all datasets (gitignored)
│   ├── raw_datasets/               ← extracted raw source datasets
│   ├── _staging/                   ← per-source intermediate YOLO conversions
│   ├── table_assistant_yolo/       ← final trainable flat YOLO dataset
│   └── table_assistant_yolo_package.zip  ← packaged dataset, DVC-tracked
├── models/                         ← trained weights (gitignored)
├── outputs/                        ← run artifacts (gitignored)
└── reports/                        ← skipped-image logs, dataset notes, baselines
```

Important conventions:

- `src/utils/paths.py` centralizes all project-relative paths. Use it instead
  of hardcoding strings. New paths go here.
- `src/data/common/convert_to_yolo.py` already provides `coco_to_yolo`,
  `voc_to_yolo`, `write_yolo_label`, `write_yolo_annotations`,
  `load_class_mapping`, and `load_label_mapping`. Reuse these helpers; do not
  reimplement.
- `src/data/common/dataset_io.py` provides shared staging/skipped-CSV/sample
  writing helpers used by every per-source converter.
- `src/data/validation/validate_dataset.py` validates YOLO label integrity.
  Run it after any dataset generation step.
- Stubs (files that just print "Pending implementation:") are intentional
  placeholders. Replace the body when implementing; preserve module-level
  docstrings.

---

## 5. Execution environments

Two environments, with different responsibilities:

- **Local** (Conda env `iaa-table-assistant`, see `environment.yml` and
  `requirements.txt`): raw dataset acquisition (FiftyOne is heavy and best
  run locally), dataset conversion to YOLO format, dataset zip packaging.
- **Google Colab**:
  - `notebooks/00_download_open_images_colab.ipynb`: download the Open Images
    subset on Colab (FiftyOne is heavy; this offloads it from a constrained
    local machine).
  - `notebooks/0.5_auto_label_open_images_colab.ipynb`: auto-label the Open
    Images subset into the `v2` export (see `src/data/auto_label/`).
  - `notebooks/01_dataset_prep_colab.ipynb`: dataset preparation (extract raw
    zips from Drive, convert per source to YOLO, merge to flat layout, refresh
    notes, package into `table_assistant_yolo_package.zip`).
  - `notebooks/02_training_colab.ipynb`: pull the DVC-tracked package,
    restore + regenerate splits at runtime, smoke + baseline training with
    MLflow on Drive.
  - `notebooks/03_inference_colab.ipynb`: load a trained `best.pt` and run
    predictions on specific images (lookup-by-name or upload).

When writing scripts, declare the intended environment at the top of the
module docstring. Do not write code that silently assumes Colab paths
(`/content/...`) on local runs or vice versa.

Google Drive layout (the project's persistent storage):

```
MyDrive/iaa-table-assistant/
├── raw_datasets/
│   ├── open_images_subset/      ← v1 COCO export zip
│   ├── open_images_subset_v2/   ← v2 auto-labeled export zip
│   ├── labels_v2_full.json      ← full v2 labels, overrides in-zip labels
│   └── uec_food_256/            ← zip from UEC source
├── training_outputs/            ← YOLO run dirs (weights, plots, summaries)
└── mlflow/                      ← MLflow tracking store
```

---

## 6. Tooling and operational rules

- **Python**: 3.10+. Type hints encouraged but not enforced strictly.
- **Module execution**: always invoke project scripts as modules
  (`python -m src.data.foo`), not as files (`python src/data/foo.py`).
  Several modules use absolute imports like `from src.utils.paths import ...`,
  which only resolve when launched via `-m` from the project root.
- **Style**: ruff with line length 100, target py310 (see `pyproject.toml`).
- **Tests**: pytest available as a dev dependency; no test suite exists yet.
  When you add non-trivial logic, add a minimal test under `tests/` (create
  the directory if needed).
- **Notebooks**: keep logic in `src/`. The Colab notebook orchestrates
  (`!python -m src.data...`) but does not contain core logic. Do not paste
  large code blocks into notebook cells.
- **Commits**: descriptive, imperative mood, no emojis. Group related
  changes. Reference the affected module in the message when useful.
- **MLflow**: wired up in `notebooks/02_training_colab.ipynb`. Tracking URI
  points at `MyDrive/iaa-table-assistant/mlflow`, experiment
  `visual-table-assistant`, Ultralytics MLflow integration enabled. Do not
  log to MLflow from data preparation steps.
- **DVC**: active. The single artifact
  `datasets/table_assistant_yolo_package.zip` is tracked (not the unpacked
  folder); the `.dvc` file is committed to git, the zip is pushed to the
  `gdrive_storage` remote. DVC-mutating commands still require explicit user
  confirmation in the same turn.

---

## 7. Common commands

Setup (local, Conda):

```bash
conda env create -f environment.yml
conda activate iaa-table-assistant
pip install -r requirements.txt
```

Lint and format (ruff, line-length 100, target py310; see `pyproject.toml`):

```bash
ruff check .
ruff format .
```

Tests (pytest is a dev dependency; no suite exists yet — see §6):

```bash
pytest
pytest tests/test_foo.py::test_bar     # single test
```

Data pipeline — always run as modules from the project root (see §6):

```bash
python -m src.data.raw_setup.download_open_images_subset   # local/Colab, FiftyOne
python -m src.data.raw_setup.setup_colab_raw_datasets      # Colab only
python -m src.data.raw_setup.visualize_raw_bboxes          # Colab only
python -m src.data.conversion.convert_open_images_to_yolo  # COCO → YOLO staging
python -m src.data.conversion.convert_uec_food_to_yolo     # VOC → YOLO staging
python -m src.data.validation.visualize_yolo_mapping       # per-class staging sanity check
python -m src.data.validation.analyze_class_distribution   # cross-source class stats
python -m src.data.preparation.prepare_dataset             # merge staging → flat layout + manifest
python -m src.data.preparation.update_dataset_notes        # refresh dataset_notes.md
python -m src.data.validation.validate_dataset             # validate final YOLO layout
python -m src.data.preparation.split_dataset               # stratified 60/15/25 split files (training time)
python -m src.data.preparation.restore_dataset_package     # unzip + symlink package (training time)
```

Training and inference (training/eval scripts are stubs; the live flow lives
in the Colab notebooks — see §8):

```bash
python -m src.training.train
python -m src.training.evaluate
python -m src.inference.export_onnx
python -m src.inference.validate_onnx
python -m src.inference.predict_image
```

---

## 8. Where we are now

Phase 1 is well advanced. Status:

- [x] Environment and local conda setup (`environment.yml`, `requirements.txt`)
- [x] Open Images V7 subset downloader (`download_open_images_subset.py`)
- [x] Open Images auto-labeling → `v2` export (`src/data/auto_label/`)
- [x] Colab raw dataset setup (`setup_colab_raw_datasets.py`, with v2 labels override)
- [x] Visual bounding box sanity checks (`visualize_raw_bboxes.py`)
- [x] Conversion helpers (`common/convert_to_yolo.py`, `common/dataset_io.py`)
- [x] Per-source converters (`conversion/convert_{open_images,uec_food}_to_yolo.py`)
- [x] Staging sanity checks (`validation/visualize_yolo_mapping.py`, `analyze_class_distribution.py`)
- [x] Merge into flat layout + manifest (`preparation/prepare_dataset.py`)
- [x] Dataset notes refresh (`preparation/update_dataset_notes.py`)
- [x] Label validation (`validation/validate_dataset.py`)
- [x] Stratified split files (`preparation/split_dataset.py`)
- [x] Dataset packaging + DVC tracking (zip + `.dvc`, pushed to gdrive remote)
- [x] Package restore at training time (`preparation/restore_dataset_package.py`)
- [x] Training notebook: smoke + baseline cells with MLflow (`02_training_colab.ipynb`)
- [x] Inference notebook: load `best.pt`, predict on specific images (`03_inference_colab.ipynb`)
- [ ] First real baseline run + recorded metrics in `reports/experiments.md`
- [ ] `src/training/*` and `src/inference/*` script implementations (logic currently in notebooks)
- [ ] ONNX export and validation

The dataset preparation workstream documented at
`docs/phase1-dataset-preparation-plan.md` is complete (kept as a historical
record). Training and inference are documented at
`docs/phase1-training-and-inference.md`.

---

## 9. Default expectations for Claude Code

When asked to implement or modify something:

1. **Read the relevant existing file(s) first**. Do not assume the codebase
   matches your priors. Several files are stubs; check before writing.
2. **Reuse existing utilities** (`src/utils/paths.py`, `src/utils/labels.py`,
   `src/data/common/convert_to_yolo.py`, `src/data/common/dataset_io.py`).
   Adding parallel helpers is a smell.
3. **Match existing style**: prose docstrings explaining purpose, no
   excessive inline comments, type hints on public functions, ruff-compliant.
4. **Keep changes scoped**. Don't refactor unrelated files in passing.
5. **Confirm before destructive actions**: deleting files, rewriting large
   sections, changing public APIs, modifying configs that affect training.
6. **State assumptions explicitly** when the request is ambiguous, and ask
   before guessing.
7. **Phase 1 boundary**: do not start work on phase 2 features (webcam,
   audio, UI) under any framing. If asked, confirm the user wants to expand
   scope before proceeding.

When asked a conceptual question (e.g., "how does X work in YOLO"):

1. Explain the concept clearly with the user's CS background in mind. He
   knows networking, OS, general CS — he does NOT know ML jargon by default.
2. State confidence levels.
3. Cite sources for non-obvious technical claims.
4. Connect the explanation back to this project's code when possible.

---

## 10. References in this repo

Documents to read on demand:

- `docs/architecture.md` — full phase 1 architecture, source-to-target label
  mapping, components, scope boundaries.
- `docs/environment-and-data-setup.md` — environment setup, Google Drive
  layout, dataset acquisition workflow.
- `docs/auto-label-open-images.md` — Open Images auto-labeling workstream
  (v2 generation, modules, notebooks).
- `docs/phase1-dataset-preparation-plan.md` — dataset prep plan, user
  decisions, staged order. Historical record; the work is complete.
- `docs/phase1-dataset-preparation-flow.md` — visual companion to the plan
  (historical).
- `docs/phase1-training-and-inference.md` — packaging, DVC, split regen,
  MLflow, baseline conventions, inference notebook.
- `reports/dataset_notes.md` — running log of dataset preparation decisions
  and known issues. Update this when making relevant decisions.
- `reports/experiments.md` — experiment log (training runs). Populate after
  each significant training run.
- `reports/baselines/` — per-run JSON summaries written by the training notebook.
- `README.md` — user-facing project overview.
