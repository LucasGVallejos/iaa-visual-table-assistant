# Environment and Data Setup

## Overview

This document describes how to prepare the local development environment, organize
Google Drive for persistent storage, acquire raw datasets, and set up Google Colab
for data inspection.

## 1. Local Environment Setup

The local environment is used primarily for data acquisition, especially downloading
Open Images V7 via FiftyOne.

```bash
conda activate iaa-table-assistant
pip install -r requirements.txt
```

FiftyOne requires a working Python environment with network access. The download
scripts run locally and produce zip files that are then uploaded manually to
Google Drive.

## 2. Open Images Acquisition

Open Images V7 is downloaded using FiftyOne, either locally or on Colab
(`notebooks/00_download_open_images_colab.ipynb`, useful when the local
machine is RAM-constrained):

```bash
python -m src.data.raw_setup.download_open_images_subset
```

The script:

- Downloads selected Open Images V7 classes with detection annotations (bounding boxes).
- Validates requested class names against the Open Images catalog before downloading.
- Downloads up to `MAX_SAMPLES_PER_CLASS` samples per class.
- Filters detections to keep only the requested classes (removes unrelated annotations).
- Deduplicates and merges per-class downloads into a single dataset.
- Exports the combined dataset to COCO Detection format.
- Creates a zip file for upload to Google Drive.

Expected output:

```
datasets/raw_datasets/
  open_images_subset/         # Exported COCO dataset (images + labels.json)
  open_images_subset.zip      # Portable zip for Google Drive
```

`datasets/` is ignored by Git.

### Open Images v2 (auto-labeled)

A second iteration improves the labels by running an auto-labeling pass over
the same images (`src/data/auto_label/`,
`notebooks/0.5_auto_label_open_images_colab.ipynb`). It produces
`open_images_subset_v2` plus a full `labels_v2_full.json`. During raw setup,
`labels_v2_full.json` overrides the partial `labels.json` shipped inside the
v2 zip. See `docs/auto-label-open-images.md` for the full workflow.

## 3. Current Open Images Source Classes

The following Open Images classes are currently configured for download:

| Open Images class | Target model class |
|---|---|
| Bottle | bottle |
| Coffee cup, Wine glass | cup |
| Bowl, Plate, Mixing bowl | plate |
| Spoon | spoon |
| Fork | fork |
| Knife, Kitchen knife | knife |

The full list of source classes in the download script:

- Bottle
- Bowl
- Coffee cup
- Fork
- Kitchen knife
- Knife
- Mixing bowl
- Plate
- Spoon
- Wine glass

**The mapping** from Open Images class names to the project's model classes **will be
applied later during YOLO dataset preparation**.

At this stage, the raw COCO export
preserves the original Open Images labels.

## 4. UEC FOOD-256 Raw Dataset

UEC FOOD-256 is used as the source for the generic `food` class. All 256 food
categories will be mapped to a single `food` class during YOLO conversion.

Dataset structure:

- `category.txt` maps numeric food IDs to food names.
- Each category has a numbered folder (e.g., `1/`, `2/`, ..., `256/`).
- Each folder contains JPEG images and a `bb_info.txt` file.
- `bb_info.txt` format (one line per bounding box):
  ```
  img x1 y1 x2 y2
  ```
- Bounding box coordinates are absolute pixel values.
- Some images have multiple bounding boxes.

## 5. Google Drive Structure

Google Drive is used as persistent storage for raw dataset zips and, later, for
prepared datasets and model weights.

Expected layout:

```
MyDrive/iaa-table-assistant/
  raw_datasets/
    open_images_subset/
      open_images_subset.zip
    open_images_subset_v2/
      open_images_subset_v2.zip
    labels_v2_full.json
    uec_food_256/
      UECFOOD256.zip
  training_outputs/
  mlflow/
```

- Google Drive stores persistent zip files that survive Colab session resets.
- Colab extracts zips into `datasets/raw_datasets/` inside the repo for faster
  local access during the session.
- `training_outputs/` and `mlflow/` persist YOLO run artifacts and MLflow
  tracking across sessions.
- Training should not read large image datasets directly from Drive if avoidable,
  since Drive I/O is significantly slower than local Colab storage.

## 6. Colab Raw Dataset Setup

Colab is used as an orchestrator. Most logic lives in versioned scripts under
`src/data/`, not inline in the notebook.

Workflow:

1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```

2. Clone the repo and install dependencies (handled in notebook section 1).

3. Run the setup script to extract and inspect raw datasets:
   ```bash
   python -m src.data.raw_setup.setup_colab_raw_datasets
   ```
   This script:
   - Finds zip files in the Drive `raw_datasets/` directories.
   - Extracts them into `datasets/raw_datasets/` inside the repo.
   - Inspects the Open Images COCO JSON (images, annotations, categories).
   - Inspects the UEC FOOD-256 structure (category.txt, bb_info.txt files).

4. Run the visual sanity check:
   ```bash
   python -m src.data.raw_setup.visualize_raw_bboxes
   ```
   This script:
   - Creates one Open Images bounding box sample image.
   - Creates one UEC FOOD-256 bounding box sample image.
   - Saves both under `outputs/bbox_checks/` inside the repo.

5. Display the generated images in the notebook:
   ```python
   from IPython.display import Image, display
   display(Image("outputs/bbox_checks/open_images_sample.png"))
   display(Image("outputs/bbox_checks/uec_food_sample.png"))
   ```

## 7. Visual Sanity Checks

Visual bounding box checks confirmed that:

- Open Images COCO bounding boxes are interpreted correctly (COCO `[x, y, w, h]`
  converted to `[x1, y1, x2, y2]` for drawing).
- UEC FOOD-256 `bb_info.txt` boxes are interpreted correctly (already in
  `[x1, y1, x2, y2]` absolute pixel format).

Generated check images:

```
outputs/bbox_checks/
  open_images_sample.png
  uec_food_sample.png
```
