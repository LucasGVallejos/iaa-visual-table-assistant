# Experiment Log

This file tracks training experiments, hyperparameters and results.

MLflow is the main experiment tracking tool; runs persist to
`MyDrive/iaa-table-assistant/mlflow`. This document is a human-readable summary
of the most relevant runs. Per-run JSON summaries live under
`reports/baselines/`.

Headline metrics below are on the **test** split (the 25% holdout untouched
during training and validation). All runs use the same dataset package
(Open Images v2 auto-labeled + UEC FOOD-256, flat YOLO layout), stratified
60/15/25 split with seed 42.

## Experiments

| Run | Dataset | Model | Epochs | Batch | ImgSz | Precision | Recall | mAP50 | mAP50-95 | Notes |
|-----|---------|-------|--------|-------|-------|-----------|--------|-------|----------|-------|
| baseline_yolo26m_001 | table_assistant_yolo (v2) | yolo26m.pt | 30 | 16 | 640 | 0.652 | 0.565 | 0.618 | 0.454 | First baseline. No class balancing. patience=7, cache=disk, seed=42. |
| baseline_yolo26l_001 | table_assistant_yolo (v2) | yolo26l.pt | 30 | 16 | 640 | 0.663 | 0.588 | 0.637 | 0.474 | Larger backbone. Best so far across every aggregate metric. Same config as m. |

## Per-class mAP@50-95 (test split)

| Class | yolo26m_001 | yolo26l_001 |
|-------|------------:|------------:|
| food | 0.594 | 0.613 |
| cup | 0.617 | 0.656 |
| bottle | 0.457 | 0.483 |
| plate | 0.368 | 0.419 |
| spoon | 0.401 | 0.384 |
| fork | 0.350 | 0.386 |
| knife | 0.391 | 0.377 |

## Observations

- **Larger backbone helps overall.** yolo26l beats yolo26m on every aggregate
  metric (test mAP50-95 0.474 vs 0.454, mAP50 0.637 vs 0.618, precision and
  recall both up). The gain is modest (~2 mAP50-95 points) for a substantially
  heavier model.
- **val/test are consistent.** test mAP50-95 is marginally higher than val for
  both runs (l: 0.474 vs 0.467; m: 0.454 vs 0.433), so no sign of val overfitting
  or a leaky split.
- **Class spread tracks the documented imbalance.** The majority classes (food,
  cup) sit highest (~0.6); the rare tableware classes (plate, spoon, fork,
  knife) trail at ~0.37–0.42. This is the expected fingerprint of the ~32:1
  food/knife imbalance and is the main lever for the next iteration.
- **Bigger model is not uniformly better per class.** Going m→l, spoon (0.401→
  0.384) and knife (0.391→0.377) regressed slightly while plate (0.368→0.419)
  and bottle (0.457→0.483) improved. The minority classes remain noisy, as
  expected from their low support.

## Next steps

- Address class imbalance (class weights / focal loss / minority-class
  oversampling) and re-measure the rare classes.
- Consider longer training (patience=7 at 30 epochs may stop early); compare
  against a longer run before committing to a backbone size.
- Record any new run here and drop its JSON under `reports/baselines/`.
