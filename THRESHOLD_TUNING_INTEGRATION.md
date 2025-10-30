# Principled Threshold Tuning Integration

## Overview

We've integrated the principled threshold tuning approach into our evaluation workflow. Instead of using hard-coded thresholds or simple grid search, we now use data-driven threshold selection based on precision-recall optimization.

## Key Features

### 1. **Two Tuning Modes**

- **F-beta mode** (default: `beta=0.5`): Optimizes F-beta score where β<1 emphasizes precision over recall
  - Perfect for medical AI where false positives are costly
  - Formula: F_β = (1+β²) * (P*R) / (β²*P + R)
  
- **Min-precision mode**: Maximizes recall subject to a minimum macro-precision constraint
  - Ensures precision ≥ target (e.g., 0.60) while maximizing recall
  - Falls back to F1-optimization if target can't be met

### 2. **Integrated Workflow**

The `evaluate_results.py` script now automatically:

1. **Prepares tuning data**: Converts predictions + ground truth → `tuning_data.csv` format
   - Extracts binary scores from `binary_outputs` column
   - Matches with ground truth labels
   - Handles "No Finding" properly (locked after other labels)

2. **Tunes thresholds**: Uses `threshold_tuner_impl.py` to find optimal per-label thresholds

3. **Saves results**:
   - `config/label_thresholds_tuned.json` - tuned thresholds
   - `thresholds_tuning_summary.csv` - per-label metrics at chosen thresholds
   - Auto-updates `config/label_thresholds.json` with tuned values

4. **Provides comparison**: Shows both principled and legacy recommendations

## Usage

### Basic Usage (Default: F-beta with β=0.5)

```bash
python evaluate_results.py \
    --predictions hybrid_ensemble_1000.csv \
    --ground_truth data/evaluation_manifest_1000.csv \
    --name "Evaluation Run"
```

### Customize Tuning Mode

Edit `evaluate_results.py` main function to change:

```python
# F-beta mode (precision-emphasized)
tuned_thresholds, tuning_results = recommend_thresholds_principled(
    tuning_csv, mode="fbeta", beta=0.5, min_precision=0.60
)

# Min-precision mode
tuned_thresholds, tuning_results = recommend_thresholds_principled(
    tuning_csv, mode="min_precision", beta=0.5, min_precision=0.65
)
```

### Standalone Threshold Tuner

You can also use the threshold tuner directly:

```bash
# First, prepare the data
python prepare_for_threshold_tuning.py \
    --predictions hybrid_ensemble_1000.csv \
    --ground_truth data/evaluation_manifest_1000.csv \
    --output tuning_data.csv

# Then tune thresholds
python threshold_tuner.py \
    --csv tuning_data.csv \
    --mode fbeta \
    --beta 0.5 \
    --out_json config/label_thresholds.json \
    --out_metrics thresholds_summary.csv
```

## Files

- **`threshold_tuner.py`**: CLI interface for threshold tuning
- **`threshold_tuner_impl.py`**: Core implementation with F-beta and min-precision logic
- **`prepare_for_threshold_tuning.py`**: Prepares data in correct format
- **`evaluate_results.py`**: Integrated evaluation with automatic threshold tuning

## Output Files

1. **`config/label_thresholds.json`**: Final tuned thresholds (used by `smart_ensemble.py`)
2. **`config/label_thresholds_tuned.json`**: Intermediate file with all 14 labels
3. **`thresholds_tuning_summary.csv`**: Per-label metrics at chosen thresholds
4. **`tuning_data.csv`**: Formatted data for tuning (y_true_L, scenario_L columns)

## Key Benefits

✅ **Data-driven**: Thresholds chosen from actual performance data, not gut feeling  
✅ **Objective-driven**: Optimizes for your specific goal (precision vs recall trade-off)  
✅ **No hard-coding**: Removes arbitrary threshold choices  
✅ **Reproducible**: Same data + same parameters = same thresholds  
✅ **Transparent**: Shows PR curves and chosen operating points

## Next Steps

1. Run evaluation on your data to generate tuned thresholds
2. The tuned thresholds are automatically saved to `config/label_thresholds.json`
3. These thresholds will be used by `smart_ensemble.py` in the next run
4. Iterate: Run → Tune → Evaluate → Adjust (if needed)

