# Phase 1 — Training and Inference

This document covers what happens after the dataset is prepared: packaging it
for distribution, pulling it at training time, regenerating splits, training a
YOLO baseline with MLflow tracking, and running inference on specific images.

Dataset preparation itself is covered by `docs/architecture.md` and the
historical `docs/phase1-dataset-preparation-plan.md`.

## Artifacts and where they live

| Artifact | Location | Persistence |
|----------|----------|-------------|
| Dataset package zip | `datasets/table_assistant_yolo_package.zip` | DVC remote (Google Drive) |
| DVC pointer | `datasets/table_assistant_yolo_package.zip.dvc` | git |
| Split files | `reports/dataset_splits/{train,val,test}.txt` | regenerated per runtime |
| MLflow store | `MyDrive/iaa-table-assistant/mlflow/` | Google Drive |
| Training runs | `MyDrive/iaa-table-assistant/training_outputs/<run>/` | Google Drive |
| Per-run summary | `reports/baselines/baseline_<size>_<timestamp>.json` | git (committed manually) |

## Dataset packaging (notebook 01 tail)

The final flat YOLO dataset is bundled into a single zip so DVC tracks one
artifact instead of tens of thousands of files. ZIP layout:

```
table_assistant_yolo_package/
├── table_assistant_yolo/
│   ├── images/
│   └── labels/
└── metadata/
    ├── dataset_manifest.csv
    ├── dataset_notes.md
    ├── dataset_package_metadata.json
    ├── class_distribution_<latest>.json
    └── skipped_images/{open_images.csv,uec_food_256.csv}
```

Splits, configs, raw and staging datasets, and visual checks are deliberately
excluded. The split files reference absolute paths that would be invalid in a
different runtime, so they are regenerated at training time rather than shipped.

DVC tracking is a local, user-gated step (DVC-mutating commands require explicit
confirmation). After notebook 01 produces the zip:

```bash
dvc add datasets/table_assistant_yolo_package.zip
git add datasets/table_assistant_yolo_package.zip.dvc datasets/.gitignore
git commit -m "track table_assistant_yolo_package.zip with DVC"
dvc push
git push
```

Commit to git: the `.dvc` file and `.gitignore`. Do **not** commit the zip, the
unpacked package dir, or the dataset dir.

## Training (`notebooks/02_training_colab.ipynb`)

Sections, in order:

1. **Repository setup** — clone/update the repo into `/content`.
2. **Dependency install** — `pip install -r requirements.txt`.
3. **Google Drive mount** — creates `mlflow/` and `training_outputs/`.
4. **DVC pull** — credentials come from the Colab Secret
   `GDRIVE_CREDENTIALS_DATA` (cached DVC gdrive credentials JSON, no interactive
   OAuth). Pulls only `datasets/table_assistant_yolo_package.zip.dvc`.
5. **Extract package** — `python -m src.data.preparation.restore_dataset_package`
   unzips the package and exposes it at `datasets/table_assistant_yolo/` via a
   symlink, so scripts and `configs/data_runtime_colab.yaml` keep addressing the
   canonical path.
6. **Generate splits** — `python -m src.data.preparation.split_dataset` writes
   `reports/dataset_splits/{train,val,test}.txt` with absolute paths for the
   current runtime. Stratified by rarest class per image, seed 42, 60/15/25.
7. **MLflow setup** — tracking URI on Drive (`mlflow/`), experiment
   `visual-table-assistant`, Ultralytics MLflow integration enabled.
8. **GPU check** — `nvidia-smi` + a CUDA-availability warning.
9. **Smoke training** — `yolo26n`, 5 epochs, to confirm the whole pipeline
   (data config → split files → loader → training loop → MLflow) works.
10. **Baseline training** — the real run.

### Baseline conventions

- `MODEL_SIZE` is the single knob to alternate runs across `n/s/m/l/x`; it
  drives both the weights name (`yolo26<size>.pt`) and the run name
  (`baseline_yolo26<size>_001`).
- `cache="disk"`, `seed=42`, `deterministic=True`, `patience` set for early
  stopping. The model family stays YOLO26 across smoke and baseline so the smoke
  validates the actual loader path.
- **No class weighting / focal loss** in the baseline: the dataset has a
  documented food/knife imbalance (~32:1) and the goal is to measure the
  uncorrected impact first.
- `MLFLOW_RUN` is set to the run name so MLflow runs are named, not auto-generated.

### Final evaluation and summary

After training, `best.pt` is re-validated on `val` (sanity at the chosen
checkpoint) and on `test` (the 25% holdout — the headline baseline metric). A
summary JSON capturing config + val/test metrics + per-class `mAP@50-95` is
written twice:

- `<save_dir>/baseline_summary.json` on Drive (auto-persisted), and
- `reports/baselines/baseline_<size>_<YYYYMMDD_HHMMSS>.json` in the repo clone.

The repo clone is ephemeral, so the second file lands in git only after you copy
it (same content) from Drive into your local checkout and commit it. No git push
from Colab.

## Inference (`notebooks/03_inference_colab.ipynb`)

Loads a trained `best.pt` and runs predictions on specific images. It does not
train, split, or evaluate.

Two input modes:

- **Lookup by name** — query a dataset filename or fragment
  (`00012345_cup_plate.jpg`, or `_knife.jpg`) against
  `datasets/table_assistant_yolo/images/`. Requires restoring the dataset once
  per session (`USE_DATASET_LOOKUP=True`, DVC pull + restore). The
  `<seq>_<class>.jpg` naming makes fragment queries a cheap audit tool for rare
  classes. Ground-truth boxes are overlaid (dashed) on predictions (solid).
- **Upload** — point at an explicit path to an uploaded image. No dataset
  restore needed; no ground truth available.

Configuration knobs: `CONF` (confidence threshold), `SHOW_GT` (ground-truth
overlay, on by default), `SAVE` (write rendered figures to
`outputs/inference/<run>/<timestamp>/`, off by default). Visualization is
matplotlib inline so figures persist in the notebook. A batch cell renders
several queries at once.

The notebook's `MODEL_SIZE` / `RUN_NAME` default to the baseline naming
convention and resolve `best.pt` under
`training_outputs/<run>/weights/best.pt` on Drive.

## Status and next steps

- The training and inference flows live in the notebooks; the `src/training/*`
  and `src/inference/*` modules are still stubs to be filled as the notebook
  logic stabilizes.
- A first real baseline run and its recorded metrics in
  `reports/experiments.md` are pending.
- ONNX export and validation (`src/inference/export_onnx.py`,
  `validate_onnx.py`) are planned.
