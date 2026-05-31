# Phase 1 — Dataset Preparation Flow

Visual companion to [`phase1-dataset-preparation-plan.md`](./phase1-dataset-preparation-plan.md).
High-level data flow from raw inputs to the prepared YOLO dataset. For
decisions, gotchas, and per-stage implementation detail, see the plan.

## Diagram

```mermaid
flowchart TB
    subgraph Now["What we have now"]
        direction LR
        OI["Open Images V7<br/>COCO format export<br/>(local_data/)"]
        UEC["UEC FOOD-256<br/>per-category folders<br/>(local_data/)"]
        CFG["configs/<br/>data_runtime_colab.yaml, classes.yaml"]
    end

    subgraph Pipeline["Pipeline — runs locally"]
        direction TB
        A["Etapa A — Mapping<br/>writes configs/label_mapping.yaml"]
        B["Etapa B — Analyze<br/>read-only counts per source/class<br/>writes reports/raw_dataset_analysis.json"]
        C["Etapa C — Convert per source<br/>COCO / bb_info.txt → YOLO labels<br/>re-encode every image to RGB JPEG q95"]
        STAGE[("datasets/_staging/<br/>open_images/ + uec_food/<br/>intermediate (deleted at end)")]
        D["Etapa D — Merge<br/>flatten + rename to<br/>NNNNNNNN_classes.jpg<br/>writes reports/dataset_manifest.csv"]
        E["Etapa E — Validate<br/>label integrity, class IDs, bbox bounds<br/>fills class counts in dataset_notes.md"]
        A --> B --> C --> STAGE --> D --> E
    end

    subgraph After["What we will have after"]
        direction LR
        OUT[("datasets/table_assistant_yolo/<br/>images/*.jpg + labels/*.txt<br/>flat YOLO layout, no splits")]
        REP[("reports/<br/>dataset_manifest.csv<br/>raw_dataset_analysis.json<br/>skipped_images.csv<br/>dataset_notes.md (filled)")]
    end

    F["Etapa F — DVC versioning<br/>dvc add + dvc push to Drive<br/>USER-GATED per §2 of plan"]
    CLEAN["Cleanup script<br/>clean_intermediates.py<br/>wipes datasets/_staging/"]

    Now --> Pipeline
    Pipeline --> After
    After -.->|user confirms| F
    E -.->|user confirms| CLEAN

    style STAGE fill:#0969da,color:#fff,stroke:#033d8b,stroke-width:2px
    style OUT fill:#1a7f37,color:#fff,stroke:#0d4524,stroke-width:2px
    style REP fill:#1a7f37,color:#fff,stroke:#0d4524,stroke-width:2px
    style F fill:#bc4c00,color:#fff,stroke:#762c00,stroke-width:2px,stroke-dasharray: 5 5
    style CLEAN fill:#bc4c00,color:#fff,stroke:#762c00,stroke-width:2px,stroke-dasharray: 5 5
```

## Reading the diagram

- **What we have now** — the two raw datasets already on disk plus the
  existing target-class configs. `configs/label_mapping.yaml` is the one
  config that doesn't exist yet; Etapa A creates it.
- **Pipeline** — five etapas running sequentially on the local machine.
  The intermediate `datasets/_staging/` (blue cylinder) exists only
  between Etapa C and Etapa D and is deleted by the cleanup script at
  the end.
- **What we will have after** — the flat YOLO dataset (`images/` +
  `labels/`, no split subdirs) plus four files in `reports/` covering
  provenance, analysis, errors, and class counts (green cylinders).
- **Dashed orange nodes** — intentionally outside the main flow.
  Etapa F (DVC) and the cleanup script both require explicit user
  confirmation in the same conversation turn per §2 of the plan.

## Etapas at a glance

| Etapa | Action                 | Key output                                   |
| ----- | ---------------------- | -------------------------------------------- |
| A     | Materialize mapping    | `configs/label_mapping.yaml`                 |
| B     | Analyze raw datasets   | `reports/raw_dataset_analysis.json`          |
| C     | Convert per source     | `datasets/_staging/{open_images,uec_food}/`  |
| D     | Merge + rename         | flat layout + `reports/dataset_manifest.csv` |
| E     | Validate               | filled class counts in `dataset_notes.md`    |
| F     | DVC versioning (gated) | `.dvc` file + remote push                    |

## Notes

- The pipeline runs end-to-end on the local machine. No Colab step is
  involved in dataset preparation; Colab enters the picture later, at
  training time.
- Re-running any etapa is safe: each stage wipes its own output before
  writing, so no stale state accumulates.
- Open Images is processed before UEC throughout, so the manifest is
  human-scannable by source.
