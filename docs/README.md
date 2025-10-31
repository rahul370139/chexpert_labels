# CheXpert Label Evaluation with CheXagent

This repository contains scripts for evaluating CheXpert label performance with both the CheXagent hybrid ensemble and a new TorchXRayVision DenseNet pipeline that produces continuous probabilities for calibration.

## Overview

The project now offers two complementary pipelines:

1. **CheXagent Hybrid Ensemble** – Text-driven reasoning with binary prompts, disease identification rescue, and precision-first gating.
2. **TorchXRayVision DenseNet** – Vision-only continuous probabilities (no score clustering) that plug directly into our calibration, threshold tuning, and evaluation tooling.

## Key Features

- **Smart Ensemble Pipeline**: Combines binary classification and disease identification (CheXagent hybrid)
- **TorchXRayVision Continuous Scores**: DenseNet-121 probabilities trained on CheXpert, ready for calibration
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
├── prepare_for_threshold_tuning.py     # Data preparation for threshold tuning (CheXagent hybrid)
├── txr_infer.py                        # TorchXRayVision DenseNet inference (continuous probabilities)
├── run_txr_pipeline.py                 # End-to-end TXR calibration + evaluation workflow
├── prepare_predictions_for_calibration.py  # Convert prob tables into y_true_/y_pred_ format
├── evaluate_prob_predictions.py        # Evaluate probability tables with thresholds
├── blend_probabilities.py              # Weighted ensemble of calibrated probability tables
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

### Running Evaluation (CheXagent Hybrid)

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

### Continuous Probability Pipeline (TorchXRayVision)

1. **Run DenseNet inference (continuous probabilities)**:
```bash
python txr_infer.py \
    --images data/image_list_1000_absolute.txt \
    --out_csv txr_predictions_1000.csv \
    --device mps
```

2. **Patient-wise split (no leakage)**:
```bash
python patient_wise_split.py \
    --manifest data/evaluation_manifest_phaseA_matched.csv \
    --predictions txr_predictions_1000.csv \
    --train_ratio 0.7
```

3. **Full TXR calibration workflow**:
```bash
python run_txr_pipeline.py \
    --images data/image_list_1000_absolute.txt \
    --ground_truth data/evaluation_manifest_phaseA_matched.csv \
    --device mps \
    --output_dir txr_pipeline
```

Artifacts (written to `txr_pipeline/`):
- `txr_predictions.csv`: raw DenseNet probabilities
- `train_txr_calibrated.csv`, `test_txr_calibrated.csv`: calibrated probabilities
- `thresholds_txr.json`: per-label thresholds (default F-beta β=0.3)
- `test_txr_metrics.csv`: held-out evaluation metrics

4. **Optional ensembling with CheXagent calibrated outputs**:
```bash
python blend_probabilities.py \
    --predictions txr_pipeline/test_txr_calibrated.csv,0.6 \
    --predictions chex_calibrated/test_chex_calibrated.csv,0.4 \
    --out_csv ensemble_test_calibrated.csv

python evaluate_prob_predictions.py \
    --predictions ensemble_test_calibrated.csv \
    --ground_truth data/ground_truth_test_30.csv \
    --thresholds txr_pipeline/thresholds_txr.json \
    --score_prefix y_cal_
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
