# IAA Table Assistant

Table assistant for visually impaired people, based on object detection with YOLOv8.

## Description

The goal of this project is to develop a system capable of identifying common objects on a table (food, cutlery, tableware) to assist visually impaired people. The system is built in incremental phases.

## Current status: Phase 1

Phase 1 focuses exclusively on:

- Dataset preparation and validation
- Training a YOLOv8 object detection model
- Metric evaluation (mAP, precision, recall)
- Experiment tracking with MLflow
- Data versioning with DVC
- Model export to ONNX format

Phase 1 **does not include** real-time camera, audio synthesis, or interactive prototype.

## Model classes

| ID | Class |
|----|-------|
| 0  | food |
| 1  | cup_glass |
| 2  | bottle |
| 3  | plate_bowl |
| 4  | spoon |
| 5  | fork |
| 6  | knife |

## Project structure

```
iaa-table-assistant/
├── configs/          # Class, dataset, and hyperparameter configuration
├── notebooks/        # Experimentation notebooks (Colab)
├── src/
│   ├── data/         # Data pipeline: prepare, convert, split, validate
│   ├── training/     # Model training and evaluation
│   ├── inference/    # ONNX export and validation (pending)
│   └── utils/        # Shared utilities
├── datasets/         # Images and labels (not versioned in git)
├── models/           # Trained model weights
├── outputs/          # Training artifacts
├── reports/          # Experiment logs and dataset notes
└── docs/             # Project documentation
```

## Setup

```bash
pip install -r requirements.txt
```

## License

MIT
