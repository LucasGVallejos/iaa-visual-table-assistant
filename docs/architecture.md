# Architecture — Phase 1

## Overview

This project is a table assistant for visually impaired people.

Phase 1 focuses on building the training and export pipeline for the detection model. Real-time camera and audio synthesis belong to Phase 2.

## Phase 1 Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌───────────────┐
│  Images +    │───▶│    Data      │────▶│  YOLO Dataset │
│  annotations │     │  Pipeline    │     │  train/val/   │
│  (raw)       │     │              │     │  test         │
└──────────────┘     └──────────────┘     └──────┬────────┘
                                                 │
                           ┌─────────────────────┤
                           ▼                     ▼
                    ┌──────────────┐     ┌──────────────┐
                    │  DVC         │     │  Training    │
                    │  (data       │     │  YOLOv8      │
                    │   versioning)│     └──────┬───────┘
                    └──────────────┘            │
                                                ▼
                    ┌──────────────┐     ┌──────────────┐
                    │  MLflow      │◀───│  Evaluation   │
                    │  (tracking)  │     │  mAP/P/R     │
                    └──────────────┘     └──────┬───────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │  ONNX Export │
                                         └──────────────┘
```

## Detection Classes

| ID | Class       | Description                    |
|----|-------------|--------------------------------|
| 0  | food        | Food items on the table        |
| 1  | cup_glass   | Cups and glasses               |
| 2  | bottle      | Bottles                        |
| 3  | plate_bowl  | Plates and bowls               |
| 4  | spoon       | Spoons                         |
| 5  | fork        | Forks                          |
| 6  | knife       | Knives                         |

## Components (Phase 1)

### Data Pipeline (`src/data/`)
- Directory structure preparation
- Annotation conversion to YOLO format
- Train/val/test splitting
- Dataset integrity validation

### Training (`src/training/`)
- YOLOv8 training with YAML-based configuration
- Evaluation with standard metrics (mAP50, mAP50-95, precision, recall)
- Experiment tracking with MLflow

### Export (`src/inference/`)
- Trained model export to ONNX
- Exported model validation

### Utilities (`src/utils/`)
- Project-relative path resolution
- YOLO label reading and class counting

## Configuration

- `configs/data.yaml` — Dataset paths and class definitions
- `configs/classes.yaml` — Class metadata (names, colors)
- `configs/train_baseline.yaml` — Training hyperparameters

## Model

- **Architecture**: YOLOv8
- **Task**: Object detection
- **Classes**: 7 (food, cup_glass, bottle, plate_bowl, spoon, fork, knife)
- **Input**: 640×640 RGB
- **Export**: ONNX

## Out of Scope (Phase 2+)

- Real-time video capture (webcam)
- Voice / audio synthesis
- User interface / interactive prototype
