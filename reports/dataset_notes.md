# Dataset Notes

## Overview

This document records the dataset sources, target classes, preprocessing decisions and known risks for the visual table assistant project.

## Selected Sources

- Open Images V7: tableware and everyday-use objects.
- UEC FOOD-256: food detection with bounding boxes.

OCID was evaluated but not used in the final Phase 1 dataset.

## Target Classes

| ID | Class | Description |
|---:|-------|-------------|
| 0 | food | Any visible food item on the table. The specific type of food is not identified. |
| 1 | cup | Drinking containers such as cups, glasses or mugs. |
| 2 | bottle | Bottles or similar vertical liquid containers. |
| 3 | plate | Plates, bowls or food containers. |
| 4 | spoon | Spoon used as tableware. |
| 5 | fork | Fork used as tableware. |
| 6 | knife | Knife used as tableware or cutting utensil. |

## Preprocessing Decisions

- Class filtering strategy: fixed 7-class target taxonomy defined in `configs/classes.yaml`.
- Source datasets: Open Images V7 for table objects and UEC FOOD-256 for food. OCID was evaluated but excluded from the final Phase 1 dataset.
- Label mapping: Open Images source labels are mapped through `configs/label_mapping.yaml`; every UEC FOOD-256 category maps to the `food` class.
- Conversion: raw annotations are converted to YOLO normalized `class_id cx cy w h` labels.
- Dataset layout: final dataset uses a flat YOLO layout with a single `images/` directory and a single `labels/` directory.
- Split strategy: split files are generated in `02_training_colab.ipynb` using rarest-class-per-image stratification, seed 42, ratios 60/15/25.
- Class balancing: no balancing is applied before the first baseline. Class imbalance will be evaluated from per-class metrics after the first training run.

## Open Images Auto-Labeling (v2)

### What was done

The original Open Images COCO export (v1) only carries the 10 source labels and
no `food` class, so many visible tableware and food items are unannotated. A
pretrained `yolov8x` COCO-80 detector was run over the v1 images to densify the
labels: its detections were mapped to our 7 target classes by category name,
de-duplicated against the existing v1 boxes, and injected as new annotations.
The result was exported as an enriched v2 COCO document
(`open_images_subset_v2/labels.json`). The v1 export — both its `labels.json`
and its `data/` images — is left untouched and remains the rollback source.

### Parameters

- Detector confidence threshold: `conf = 0.4`.
- Same-class IoU de-duplication against existing boxes: `iou_dedup = 0.5`.
- Inference image size: `imgsz = 640`.

### Results

- 25,734 boxes added across 7,047 of 11,251 images (62.6%).
- Per-class additions: cup 9,433, plate 3,524, bottle 3,315, food 2,632,
  spoon 2,581, knife 2,404, fork 1,845.
- 19,418 detections skipped as duplicates of existing v1 boxes.
- All original v1 boxes are preserved unchanged.

### Provenance

- Each injected annotation carries a detector `score` and a
  `source: "auto_label"` tag, so auto-labeled boxes can be told apart from the
  original export at any later step.
- The full run report is at `reports/auto_label_report.json`.

### Known limitations

- COCO has no flat `plate` class; the `bowl` class is used as a surrogate for
  `plate`, which can miss flat dishes.
- COCO `food` is only 10 specific food classes (e.g. banana, pizza, sandwich),
  so generic plated dishes are under-detected — UEC FOOD-256 remains the main
  source for the `food` class.
- `person`, `chair` and `dining table` detections are deliberately dropped (no
  target mapping), so they never enter the dataset.

### Export discrepancy note

The LOCAL v1 zip's `labels.json` lists 13,023 image entries while its `data/`
holds only 11,251 files. The DRIVE v1 zip (which produced v2 on Colab) is
self-consistent at 11,251 images, and v2 derives from that Drive variant. The
packaging step therefore hard-validates that every v2-referenced image exists
locally and fails otherwise, directing the packaging to run in Colab from the
Drive-extracted v1.

## Known Issues

- Class imbalance: `food` is the dominant class and `knife` is the least represented one.
- Some UEC FOOD-256 images may contain visible tableware that is not annotated as a separate object.
- Some Open Images scenes may contain visible objects that the source dataset does not annotate.
- Minority classes (`knife`, `fork`, `spoon`) may produce noisier per-class metrics and should be reviewed after baseline training.