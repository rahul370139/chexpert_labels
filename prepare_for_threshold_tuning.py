"""
Prepare prediction and ground truth data for threshold tuning.

Converts our prediction CSV (with binary_outputs) and ground truth CSV
to the format expected by threshold_tuner.py (y_true_L, y_pred_L columns).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

CHEXPERT13 = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

CHEXPERT14 = CHEXPERT13 + ["No Finding"]


def prepare_tuning_data(
    predictions_csv: Path,
    ground_truth_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """
    Prepare data for threshold tuning.
    
    Args:
        predictions_csv: CSV with predictions (must have binary_outputs column)
        ground_truth_csv: CSV with ground truth labels
        output_csv: Path to save the prepared CSV
        
    Returns:
        DataFrame in the format expected by threshold_tuner
    """
    print("📊 Preparing data for threshold tuning...")
    
    # Load predictions
    pred_df = pd.read_csv(predictions_csv)
    print(f"✅ Loaded {len(pred_df)} predictions")
    
    # Load ground truth
    gt_df = pd.read_csv(ground_truth_csv)
    print(f"✅ Loaded {len(gt_df)} ground truth samples")
    
    # Extract filename for matching
    pred_df["filename"] = pred_df["image"].apply(lambda x: Path(x).name)
    gt_df["filename"] = gt_df["image"].apply(lambda x: Path(x).name)
    
    # Merge on filename
    merged = pd.merge(pred_df, gt_df, on="filename", suffixes=("_pred", "_gt"))
    print(f"✅ Matched {len(merged)} images")
    
    # Prepare output DataFrame
    output_data = {"filename": merged["filename"].values}
    
    # Extract scores from binary_outputs and ground truth labels
    if "binary_outputs" not in pred_df.columns:
        raise ValueError("predictions_csv must contain 'binary_outputs' column with scores")
    
    binary_dicts = merged["binary_outputs"].fillna("{}").apply(json.loads)
    
    for label in CHEXPERT13:
        # Extract predicted scores
        scores = binary_dicts.apply(
            lambda record: record.get(label, {}).get("score", np.nan)
        )
        output_data[f"y_pred_{label}"] = scores.values
        
        # Extract ground truth
        gt_col = f"{label}_gt"
        if gt_col not in merged.columns:
            # Try alternative naming
            gt_col = label
            if gt_col not in merged.columns:
                print(f"⚠️  Warning: Ground truth column for {label} not found, using zeros")
                output_data[f"y_true_{label}"] = np.zeros(len(merged))
                continue
        
        output_data[f"y_true_{label}"] = merged[gt_col].values.astype(int)
    
    # Handle "No Finding"
    # No Finding = 1 if all other labels are 0
    no_finding_pred = (
        merged[[f"{label}_pred" for label in CHEXPERT13]].sum(axis=1) == 0
    ).astype(int)
    no_finding_gt = merged.get("No Finding_gt", merged.get("No Finding", 0))
    if isinstance(no_finding_gt, (pd.Series, np.ndarray)):
        no_finding_gt = no_finding_gt.values.astype(int)
    else:
        no_finding_gt = np.zeros(len(merged), dtype=int)
    
    output_data["y_pred_No Finding"] = no_finding_pred.astype(float)
    output_data["y_true_No Finding"] = no_finding_gt
    
    # Create DataFrame
    output_df = pd.DataFrame(output_data)
    
    # Save
    output_df.to_csv(output_csv, index=False)
    print(f"💾 Saved prepared data to {output_csv}")
    print(f"   Columns: {list(output_df.columns)}")
    
    return output_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for threshold tuning")
    parser.add_argument("--predictions", required=True, help="Predictions CSV")
    parser.add_argument("--ground_truth", required=True, help="Ground truth CSV")
    parser.add_argument("--output", default="tuning_data.csv", help="Output CSV")
    
    args = parser.parse_args()
    
    prepare_tuning_data(
        Path(args.predictions),
        Path(args.ground_truth),
        Path(args.output),
    )

