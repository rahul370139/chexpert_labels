# CheXpert Label Evaluation with CheXagent

This repository contains scripts for evaluating CheXagent model performance on CheXpert label prediction using a hybrid ensemble approach.

## Overview

The project implements a probability-driven hybrid pipeline that combines:
- **Binary Disease Classification**: 13 binary calls per image (Yes/No for each CheXpert label)
- **Disease Identification**: Narrative text analysis for verification and rescue
- **Per-Class Threshold Tuning**: Data-driven threshold optimization for each disease

## Key Features

- **Smart Ensemble Pipeline**: Combines binary classification and disease identification
- **Precision-First Threshold Tuning**: Optimizes thresholds per disease with medical AI considerations
- **Comprehensive Evaluation**: Detailed per-disease and overall performance metrics
- **Ground Truth Manifest Creation**: Balanced dataset sampling from curriculum data

## Repository Structure

```
├── smart_ensemble.py                    # Main hybrid ensemble prediction script
├── evaluate_results.py                  # Comprehensive evaluation and threshold tuning
├── calculate_performance_detailed.py    # Detailed per-disease performance report
├── create_ground_truth_manifest.py     # Generate balanced ground truth datasets
├── threshold_tuner.py                   # CLI for principled threshold tuning
├── threshold_tuner_impl.py             # Core threshold tuning logic (F-beta, min-precision)
├── prepare_for_threshold_tuning.py     # Data preparation for threshold tuning
├── infer_with_chexagent_class.py       # Basic CheXagent inference
├── config/
│   └── label_thresholds.json           # Per-disease thresholds (tuned)
├── data/
│   ├── evaluation_manifest_1000.csv    # Ground truth for 1000 images
│   └── image_list_*.txt                # Image path lists
└── README.md                            # This file
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running Evaluation

1. **Create ground truth manifest**:
```bash
python create_ground_truth_manifest.py \
    --input curriculum_train_final_clean.jsonl \
    --output data/evaluation_manifest_1000.csv \
    --n_samples 1000
```

2. **Run smart ensemble predictions**:
```bash
python smart_ensemble.py \
    --images data/image_list_1000_absolute.txt \
    --out_csv predictions_1000.csv \
    --device mps  # or cuda, cpu
```

3. **Evaluate and tune thresholds**:
```bash
python evaluate_results.py \
    --predictions predictions_1000.csv \
    --ground_truth data/evaluation_manifest_1000.csv \
    --precision_target 0.70
```

4. **Get detailed performance report**:
```bash
python calculate_performance_detailed.py
```

## Threshold Tuning

The system uses a two-stage approach:

1. **Principled Tuning** (F-beta or min-precision): Data-driven optimization from precision-recall curves
2. **Legacy Fallback** (Precision-first): Used when principled tuning produces uniform thresholds due to score clustering

See [README_thresholds.md](README_thresholds.md) for detailed documentation.

## Key Results

With per-class optimized thresholds (as of latest evaluation):

- **Macro F1-Score**: 0.352 (CHEXPERT14)
- **Micro F1-Score**: 0.418 (CHEXPERT14)
- **Top Performers**: Support Devices (F1=0.740), Pleural Effusion (F1=0.702), Edema (F1=0.563)

## Documentation

- [README_thresholds.md](README_thresholds.md): Threshold tuning utility documentation
- [THRESHOLD_TUNING_INTEGRATION.md](THRESHOLD_TUNING_INTEGRATION.md): Integration details

## License

See LICENSE file for details.
