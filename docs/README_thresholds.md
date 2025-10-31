
# CheXpert Threshold Tuner

This utility picks per-label thresholds from validation predictions to optimize your desired objective.

## Input CSV format
One row per study/image. For each label L in CHEXPERT14, include:
- `y_true_L` (0 or 1)
- `y_pred_L` (probability or calibrated score)

## Modes
1. **F-beta (default beta=0.5)** — precision-weighted:
   ```bash
   python threshold_tuner.py --csv val_preds.csv --mode fbeta --beta 0.5
   ```
2. **Min-precision constraint** — maximize recall subject to macro-precision >= target:
   ```bash
   python threshold_tuner.py --csv val_preds.csv --mode min_precision --min_macro_precision 0.60
   ```

Outputs:
- `thresholds.json` — per-label thresholds
- `thresholds_summary.csv` — per-label operating points
- console prints macro/micro P/R/F1 for the chosen thresholds
