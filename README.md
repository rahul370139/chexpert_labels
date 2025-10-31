# CheXpert Label Evaluation

Ensemble pipeline combining CheXagent (text reasoning) + TorchXRayVision (continuous probabilities) for CheXpert label prediction with calibration and threshold optimization.

## 📁 Repository Structure

```
├── src/                           # All source code
│   ├── inference/                # Core inference scripts
│   │   ├── smart_ensemble.py    # CheXagent hybrid ensemble
│   │   ├── txr_infer.py         # TorchXRayVision DenseNet
│   │   └── infer_with_chexagent_class.py
│   ├── calibration/              # Probability calibration
│   │   ├── fit_label_calibrators.py
│   │   └── apply_label_calibrators.py
│   ├── thresholds/               # Threshold optimization
│   │   ├── threshold_tuner.py
│   │   └── threshold_tuner_impl.py
│   ├── evaluation/               # Evaluation scripts
│   │   ├── evaluate_prob_predictions.py
│   │   ├── evaluate_against_phaseA.py
│   │   └── evaluate_results.py
│   ├── data_prep/                # Data preparation utilities
│   │   ├── patient_wise_split.py
│   │   ├── prepare_for_threshold_tuning.py
│   │   ├── prepare_predictions_for_calibration.py
│   │   └── create_ground_truth_*.py
│   ├── pipelines/                # End-to-end orchestrators
│   │   ├── run_txr_pipeline.py
│   │   └── run_full_ensemble_pipeline.py
│   └── utils/                    # Utility scripts
│       ├── blend_probabilities.py
│       └── redecide_with_precision_gating.py
├── config/                       # Configuration files
│   ├── label_thresholds.json    # Per-disease thresholds
│   └── platt_params.json        # Calibration parameters
├── data/                         # Data files
│   ├── evaluation_manifest_*.csv
│   └── image_list_*.txt
├── results/                      # Output files (CSVs, logs)
├── docs/                         # Documentation
└── scripts/archive/              # Old/test scripts
```

## 🚀 Quick Start

### Full Ensemble Pipeline (TXR + CheXagent)

```bash
# From project root
python src/pipelines/run_full_ensemble_pipeline.py
```

### Individual Components

```bash
# TXR inference
python src/inference/txr_infer.py \
  --images data/image_list.txt \
  --out_csv results/txr_preds.csv \
  --device mps

# CheXagent inference
python src/inference/smart_ensemble.py \
  --images data/image_list.txt \
  --out_csv results/chex_preds.csv \
  --device mps \
  --thresholds config/label_thresholds.json

# Blend probabilities
python src/utils/blend_probabilities.py \
  --predictions results/txr_preds.csv,0.6 \
  --predictions results/chex_preds.csv,0.4 \
  --out_csv results/ensemble_preds.csv
```

## 🔑 Key Features

- **Continuous Probabilities**: TXR provides smooth scores (0.0-1.0), not just 0.20/0.80
- **Calibration**: Platt scaling per-label for calibrated probabilities
- **Precision-First Tuning**: F-beta optimization with medical AI considerations
- **Ensemble Blending**: Per-label logistic regression combining TXR + CheXagent
- **NaN Handling**: CheXagent fills gaps where TXR doesn't output (Pleural Other, Support Devices, No Finding)

## 📚 Documentation

See `docs/` for detailed workflows:
- `CALIBRATION_WORKFLOW.md` - Calibration pipeline guide
- `README_thresholds.md` - Threshold tuning documentation
