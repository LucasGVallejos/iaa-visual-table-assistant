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

The final YOLO class set is defined in `configs/data.yaml` and should NOT be
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
  locally with FiftyOne via `src/data/raw_setup/download_open_images_subset.py`.
  Source classes: Bottle, Bowl, Coffee cup, Fork, Kitchen knife, Knife,
  Mixing bowl, Plate, Spoon, Wine glass.
- **UEC FOOD-256** for the `food` class. All 256 categories collapse to
  class id 0.

The mapping from source labels to YOLO class IDs is enumerated in
`docs/architecture.md` ("Source-to-Target Label Mapping"). When the file
`configs/label_mapping.yaml` exists, that file is the source of truth for the
mapping (since it is machine-readable). Until it exists, use the table in
`architecture.md`.

OCID was originally proposed but is OUT OF SCOPE for phase 1.

---

## 4. Repository layout (what lives where)

```
iaa-visual-table-assistant/
├── CLAUDE.md                       ← this file
├── README.md                       ← user-facing entry
├── configs/                        ← YAML configs (data, classes, training)
├── docs/                           ← architecture, setup, phase plans
├── notebooks/                      ← Colab training notebook
├── reports/                        ← experiment logs, dataset notes
├── src/
│   ├── data/
│   │   ├── raw_setup/              ← download, extract, visualize raw datasets
│   │   ├── conversion/             ← per-source COCO/VOC → YOLO converters
│   │   ├── common/                 ← shared YOLO + dataset I/O helpers
│   │   ├── preparation/            ← merge + split into final YOLO dataset
│   │   └── validation/             ← YOLO label/integrity validation
│   ├── training/                   ← train, evaluate, MLflow (stubs)
│   ├── inference/                  ← ONNX export and validation (stubs)
│   └── utils/                      ← path and label helpers
├── datasets/                       ← all datasets (gitignored)
│   ├── raw_datasets/               ← extracted raw source datasets
│   ├── _staging/                   ← per-source intermediate YOLO conversions
│   └── table_assistant_yolo/       ← final trainable YOLO dataset (DVC-tracked later)
├── models/                         ← trained weights (gitignored, DVC-tracked)
├── outputs/                        ← run artifacts (gitignored)
└── reports/                        ← skipped-image logs, dataset notes
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
  - `notebooks/01_dataset_prep_colab.ipynb`: dataset preparation (extract raw
    zips from Drive, convert per source to YOLO, merge, split). Reads zips
    from Drive into `datasets/raw_datasets/` inside the repo for fast local
    I/O during the session.
  - `notebooks/02_training_colab.ipynb`: training, evaluation, ONNX export.
    Pulls the DVC-tracked dataset before running.

When writing scripts, declare the intended environment at the top of the
module docstring. Do not write code that silently assumes Colab paths
(`/content/...`) on local runs or vice versa.

Google Drive layout (the project's persistent storage):

```
MyDrive/iaa-table-assistant/
├── raw_datasets/
│   ├── open_images/      ← zip from local download
│   └── uec_food_256/     ← zip from UEC source
├── prepared_datasets/     ← final YOLO dataset zip (phase 1 output)
├── models/                ← trained weights
└── outputs/               ← evaluation artifacts
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
- **MLflow**: deferred until the training pipeline is wired up. Do not log
  to MLflow from data preparation steps.
- **DVC**: deferred until the dataset reaches "definitive" state. Do not add
  intermediate staging dirs to DVC.

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
python -m src.data.raw_setup.download_open_images_subset   # local, FiftyOne
python -m src.data.raw_setup.setup_colab_raw_datasets      # Colab only
python -m src.data.raw_setup.visualize_raw_bboxes          # Colab only
python -m src.data.preparation.prepare_dataset             # scaffolds datasets/table_assistant_yolo/
python -m src.data.validation.validate_dataset             # validates final YOLO layout
```

Training and inference (currently stubs — see §8 below):

```bash
python -m src.training.train
python -m src.training.evaluate
python -m src.inference.export_onnx
python -m src.inference.validate_onnx
```

---

## 8. Where we are now

Phase 1 is in progress. Status:

- [x] Environment and local conda setup (`environment.yml`, `requirements.txt`)
- [x] Open Images V7 subset downloader (`download_open_images_subset.py`)
- [x] Colab raw dataset setup (`setup_colab_raw_datasets.py`)
- [x] Visual bounding box sanity checks (`visualize_raw_bboxes.py`)
- [x] YOLO directory scaffolding (`prepare_dataset.py`)
- [x] Conversion helpers (`convert_to_yolo.py`)
- [x] Label validation skeleton (`validate_dataset.py`)
- [ ] **Dataset preparation: convert + merge + split** ← in progress
- [ ] Training pipeline
- [ ] Evaluation
- [ ] MLflow integration
- [ ] DVC versioning
- [ ] ONNX export and validation

The current focused workstream is documented in detail at:

@docs/phase1-dataset-preparation-plan.md

That file contains user decisions, staged plan, and gotchas. Read it before
suggesting code for the dataset preparation step.

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
- `docs/phase1-dataset-preparation-plan.md` — current step's plan, user
  decisions, staged execution order.
- `reports/dataset_notes.md` — running log of dataset preparation decisions
  and known issues. Update this when making relevant decisions.
- `reports/experiments.md` — experiment log (training runs). Populate after
  each significant training run.
- `README.md` — user-facing project overview.
