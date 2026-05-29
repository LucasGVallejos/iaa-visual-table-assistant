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

## Known Issues

- Class imbalance: `food` is the dominant class and `knife` is the least represented one.
- Some UEC FOOD-256 images may contain visible tableware that is not annotated as a separate object.
- Some Open Images scenes may contain visible objects that the source dataset does not annotate.
- Minority classes (`knife`, `fork`, `spoon`) may produce noisier per-class metrics and should be reviewed after baseline training.