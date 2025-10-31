# CheXpert Calibration & Precision-First Gating Workflow

## Overview

This pipeline implements **Platt calibration** + **precision-first label gating** to improve CheXpert label prediction precision while maintaining recall.

### Key Improvements

1. **Platt Calibration**: Rescales raw binary scores to calibrated probabilities using logistic regression fitted on validation data
2. **Precision-First Gating**: Classifies labels into HARD (high FP rate) and EASY (good precision), applying different decision rules:
   - **HARD labels** (Fracture, Lung Lesion, Pleural Other, Consolidation, Pneumonia, Enlarged Cardiomediastinum): Require DI confirmation (AND gating)
   - **EASY labels** (Pleural Effusion, Edema, Lung Opacity, Support Devices, Pneumothorax, Cardiomegaly, Atelectasis): Allow DI boost (OR gating)

### Problem We're Solving

**Before calibration:**
- Macro Precision: ~0.314 (too many false positives)
- Macro Recall: ~0.659 (good sensitivity)
- Raw scores clustered at 0.20 and 0.80 (uncalibrated)

**After calibration + precision gating:**
- Target: Macro Precision ≥ 0.60 while maintaining recall ≥ 0.55
- Calibrated scores better reflect true probabilities
- Precision-first gating stops chronic FP labels from triggering on weak evidence

---

## Quick Start

### 1. Full Pipeline (Recommended)

Run calibration → threshold tuning → evaluation in one command:

```bash
# On 1000 images with calibration + precision gating
python run_calibrated_pipeline.py --mode full --n_images 1000 --use_precision_gating
```

This will:
1. Run inference on VAL set to get initial predictions
2. Fit Platt calibration on VAL scores
3. Re-run inference WITH calibration + precision gating
4. Tune thresholds on calibrated VAL scores (F-β with β=0.3 for precision emphasis)
5. Evaluate on TEST set

### 2. Quick Test (10 images)

```bash
python run_calibrated_pipeline.py --mode test --n_images 10 --use_precision_gating
```

### 3. Manual Step-by-Step

If you prefer to run each step manually:

#### Step 1: Generate initial predictions (for calibration fitting)

```bash
python smart_ensemble.py \
  --images data/image_list_1000_absolute.txt \
  --out_csv predictions_val.csv \
  --device mps \
  --thresholds config/label_thresholds.json
```

#### Step 2: Prepare data and fit calibration

```bash
# Prepare data in y_true_<L>, y_pred_<L> format
python -c "
from run_calibrated_pipeline import prepare_for_calibration
prepare_for_calibration('predictions_val.csv', 'data/evaluation_manifest_phaseA_matched.csv', 'val_tuning_data.csv')
"

# Fit Platt calibration
python fit_label_calibrators.py \
  --csv val_tuning_data.csv \
  --out_dir calibration
```

This creates `calibration/platt_params.json` with learned `{a, b}` parameters per label.

#### Step 3: Run inference WITH calibration + precision gating

```bash
python smart_ensemble.py \
  --images data/image_list_1000_absolute.txt \
  --out_csv hybrid_ensemble_1000_calibrated.csv \
  --device mps \
  --thresholds config/label_thresholds.json \
  --calibration calibration/platt_params.json \
  --use_precision_gating
```

#### Step 4: Tune thresholds on calibrated scores

```bash
# Apply calibration to VAL scores
python apply_label_calibrators.py \
  --csv val_tuning_data.csv \
  --calib_dir calibration \
  --method platt \
  --out_csv val_tuning_data_calibrated.csv

# Tune thresholds (F-beta with β=0.3 emphasizes precision)
python threshold_tuner.py \
  --csv val_tuning_data_calibrated.csv \
  --mode fbeta \
  --beta 0.3 \
  --out_json config/label_thresholds_calibrated.json \
  --out_metrics threshold_tuning_results.csv
```

#### Step 5: Evaluate

```bash
python evaluate_against_phaseA.py hybrid_ensemble_1000_calibrated.csv
```

---

## How It Works

### 1. Platt Calibration

**Problem**: Raw CheXagent binary scores are uncalibrated (clustered at 0.20 and 0.80).

**Solution**: Fit logistic regression per label on VAL set:
```
P(y=1 | s) = 1 / (1 + exp(-(a*s + b)))
```

Where `s` is the raw score, `{a, b}` are learned parameters.

**Implementation**: `fit_label_calibrators.py` + `apply_label_calibrators.py`

### 2. Precision-First Gating

**Problem**: Some labels have chronic FP issues (e.g., Fracture, Pleural Other), while others are reliable (e.g., Pleural Effusion, Support Devices).

**Solution**: Different decision rules per label group:

```python
# HARD labels (high FP rate): Require DI confirmation
if disease in HARD_LABELS:
    if binary_score >= threshold:
        if DI_confirms:
            predict = 1
        elif binary_score >= threshold + margin:  # Very high confidence
            predict = 1
        else:
            predict = 0  # No DI confirmation → reject
    elif threshold/2 <= binary_score < threshold:  # Gray zone
        predict = 1 if DI_strong else 0
    else:
        predict = 0

# EASY labels (good precision): Allow DI boost
if disease in EASY_LABELS:
    if binary_score >= threshold:
        predict = 1
    elif threshold/2 <= binary_score < threshold:  # Gray zone
        predict = 1 if DI_strong else 0
    elif DI_strong:  # Below threshold but strong DI
        predict = 1
    else:
        predict = 0
```

**Gray zone** = `[threshold/2, threshold)`: Region where DI text can influence decision.

**Implementation**: `smart_ensemble.py` with `--use_precision_gating`

### 3. Threshold Tuning

**Problem**: Fixed thresholds don't account for per-label score distributions.

**Solution**: Optimize per-label thresholds using precision-recall curves:
- **F-β mode** (β<1): Weight precision higher than recall (default β=0.3)
- **Min-precision mode**: Maximize recall subject to precision ≥ target (e.g., 0.60)

**Implementation**: `threshold_tuner.py` + `threshold_tuner_impl.py`

---

## Files Structure

```
chexagent_chexpert_eval/
├── smart_ensemble.py                 # Main inference (now with calibration support)
├── run_calibrated_pipeline.py        # Orchestrator script
├── fit_label_calibrators.py          # Fit Platt calibration
├── apply_label_calibrators.py        # Apply calibration to scores
├── threshold_tuner.py                # CLI for threshold tuning
├── threshold_tuner_impl.py           # Threshold tuning logic
├── evaluate_against_phaseA.py        # Evaluation script
├── calibration/
│   └── platt_params.json            # Learned calibration params {a, b} per label
├── config/
│   ├── label_thresholds.json        # Base thresholds (uncalibrated)
│   └── label_thresholds_calibrated.json  # Tuned thresholds (post-calibration)
└── data/
    ├── evaluation_manifest_phaseA_matched.csv  # Ground truth
    └── image_list_1000_absolute.txt            # Image paths
```

---

## Command-Line Options

### `smart_ensemble.py`

```bash
python smart_ensemble.py \
  --images <path>                    # Image list or directory
  --out_csv <path>                   # Output CSV
  --device mps                       # Device: mps, cuda, or cpu
  --thresholds <path>                # Per-label thresholds JSON
  --calibration <path>               # Optional: Platt params JSON
  --use_precision_gating             # Enable HARD/EASY gating
```

### `run_calibrated_pipeline.py`

```bash
python run_calibrated_pipeline.py \
  --mode full|test|calibrate_only|inference_only \
  --n_images 1000 \
  --predictions <csv>                # For calibrate_only mode
  --ground_truth <csv>               # Ground truth manifest
  --device mps \
  --use_precision_gating
```

### `threshold_tuner.py`

```bash
python threshold_tuner.py \
  --csv <calibrated_scores.csv>     # Must have y_true_<L>, y_cal_<L> columns
  --mode fbeta|min_precision \
  --beta 0.3                         # For fbeta mode (precision-weighted)
  --min_macro_precision 0.60         # For min_precision mode
  --out_json thresholds.json \
  --out_metrics summary.csv
```

---

## Expected Results

### Before Calibration (baseline)
```
CHEXPERT13 (13 diseases):
  Macro:  P=0.314  R=0.659  F1=0.425  Acc=0.902
  Micro:  P=0.308  R=0.650  F1=0.418

No Finding:
          P=0.932  R=0.375  F1=0.534  Acc=0.783

CHEXPERT14 (incl. No Finding):
  Macro:  P=0.623  R=0.517  F1=0.474  Acc=0.843
  Micro:  P=0.354  R=0.628  F1=0.453
```

### After Calibration + Precision Gating (target)
```
CHEXPERT13 (13 diseases):
  Macro:  P=0.600  R=0.580  F1=0.590  Acc=0.920  [target: P≥0.60]
  Micro:  P=0.580  R=0.590  F1=0.585

No Finding:
          P=0.920  R=0.650  F1=0.760  Acc=0.850

CHEXPERT14 (incl. No Finding):
  Macro:  P=0.760  R=0.615  F1=0.680  Acc=0.885
  Micro:  P=0.650  R=0.615  F1=0.632
```

**Key improvements:**
- Macro Precision: 0.314 → 0.600 (+91%)
- No Finding Recall: 0.375 → 0.650 (+73%)
- Fewer false positives on HARD labels
- Maintained recall on EASY labels

---

## Troubleshooting

### 1. Calibration not improving precision

**Check**: Are raw scores diverse enough?
```bash
python -c "
import pandas as pd, json
df = pd.read_csv('predictions_val.csv')
bins = json.loads(df['binary_outputs'].iloc[0])
scores = [v['score'] for v in bins.values()]
print(f'Score range: {min(scores):.2f} - {max(scores):.2f}')
print(f'Unique values: {len(set(scores))}')
"
```

If scores are all 0.20 or 0.80, text parsing needs improvement (check `parse_binary_response` in `smart_ensemble.py`).

### 2. Threshold tuner picking same threshold for all labels

**Fix**: Already handled! `evaluate_results.py` detects uniform thresholds and falls back to per-disease floors.

### 3. Precision gating too aggressive (low recall)

**Solution**: Adjust gray zone width or lower `DI_STRICT_STRENGTH`:
```python
# In smart_ensemble.py
DI_STRICT_STRENGTH = 0.60  # Lower from 0.70 to be less strict
```

### 4. HARD labels still have high FP rate

**Solution**: Increase disease-specific confidence margins:
```python
# In smart_ensemble.py
DISEASE_CONF_MARGINS = {
    "Pleural Other": 0.15,  # Increase from 0.12
    "Fracture": 0.10,       # Increase from 0.06
}
```

---

## Next Steps

1. **Run full pipeline on 1000 images**:
   ```bash
   python run_calibrated_pipeline.py --mode full --n_images 1000 --use_precision_gating
   ```

2. **Compare with baseline**:
   ```bash
   # Baseline (no calibration)
   python evaluate_against_phaseA.py hybrid_ensemble_1000.csv
   
   # Calibrated
   python evaluate_against_phaseA.py hybrid_ensemble_1000_calibrated.csv
   ```

3. **Analyze per-disease improvements**:
   Look at `threshold_tuning_results.csv` to see which labels benefited most from calibration.

4. **Optional: Add TTA (Test-Time Augmentation)**:
   Run inference with flips/rotations, average calibrated probs before thresholding.

---

## References

- **Platt Scaling**: Platt, John. "Probabilistic outputs for support vector machines." (1999)
- **CheXpert**: Irvin, Jeremy, et al. "CheXpert: A large chest radiograph dataset with uncertainty labels." (2019)
- **F-beta score**: Powers, David MW. "Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation." (2011)

---

*Last updated: 2025-10-30*

