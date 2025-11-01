#!/usr/bin/env python3
"""
Evaluate predictions against three-class ground truth (-1 uncertain, 0 negative, 1 positive).

Compares predictions (0/1) with ground truth (-1/0/1) and reports:
- Performance on certain labels only (GT = 0 or 1, ignoring -1)
- Performance on all labels (treating -1 as 0 for comparison)
- Agreement analysis (where GT has -1)
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np

# Import shared labels
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.common.labels import CHEXPERT13, CHEXPERT14


def evaluate_three_class(
    predictions_csv: Path,
    ground_truth_csv: Path,
    output_csv: Path,
) -> None:
    """
    Evaluate predictions against three-class ground truth.
    
    Args:
        predictions_csv: Predictions CSV (0/1 predictions)
        ground_truth_csv: Ground truth CSV with -1/0/1 labels
        output_csv: Output evaluation CSV
    """
    print("=" * 80)
    print("THREE-CLASS EVALUATION (-1/0/1)")
    print("=" * 80)
    
    # Load data
    print(f"\n📂 Loading predictions: {predictions_csv}")
    pred_df = pd.read_csv(predictions_csv)
    print(f"   Rows: {len(pred_df)}")
    
    print(f"\n📂 Loading ground truth: {ground_truth_csv}")
    gt_df = pd.read_csv(ground_truth_csv)
    print(f"   Rows: {len(gt_df)}")
    
    # Ensure filename column
    if "filename" not in pred_df.columns:
        if "image" in pred_df.columns:
            pred_df["filename"] = pred_df["image"].apply(lambda x: Path(x).name if pd.notna(x) else "")
    
    if "filename" not in gt_df.columns:
        if "image" in gt_df.columns:
            gt_df["filename"] = gt_df["image"].apply(lambda x: Path(x).name if pd.notna(x) else "")
    
    # Merge on filename
    merged = pred_df.merge(gt_df, on="filename", how="inner", suffixes=("_pred", "_gt"))
    print(f"\n✅ Matched {len(merged)} images")
    
    if len(merged) == 0:
        print("❌ No matches found")
        sys.exit(1)
    
    # Evaluate per label
    results = []
    
    for label in CHEXPERT13:
        # Get columns
        pred_col = label if label in merged.columns else f"{label}_pred"
        gt_col = label if f"{label}_pred" not in merged.columns else label
        
        if pred_col not in merged.columns or gt_col not in merged.columns:
            print(f"⚠️  Skipping {label}: missing columns")
            continue
        
        y_pred = merged[pred_col].astype(int).values
        y_true_raw = merged[gt_col].astype(int).values
        
        # Count uncertain (-1) in ground truth
        uncertain_count = (y_true_raw == -1).sum()
        certain_count = len(y_true_raw) - uncertain_count
        
        # Mode 1: Only evaluate on certain labels (GT = 0 or 1, ignore -1)
        certain_mask = y_true_raw != -1
        if certain_mask.sum() > 0:
            y_true_certain = y_true_raw[certain_mask]
            y_pred_certain = y_pred[certain_mask]
            
            p_certain = precision_score(y_true_certain, y_pred_certain, zero_division=0)
            r_certain = recall_score(y_true_certain, y_pred_certain, zero_division=0)
            f1_certain = f1_score(y_true_certain, y_pred_certain, zero_division=0)
            acc_certain = accuracy_score(y_true_certain, y_pred_certain)
        else:
            p_certain = r_certain = f1_certain = acc_certain = np.nan
        
        # Mode 2: Treat -1 as 0 for comparison (binary evaluation)
        y_true_binary = (y_true_raw == 1).astype(int)
        p_binary = precision_score(y_true_binary, y_pred, zero_division=0)
        r_binary = recall_score(y_true_binary, y_pred, zero_division=0)
        f1_binary = f1_score(y_true_binary, y_pred, zero_division=0)
        acc_binary = accuracy_score(y_true_binary, y_pred)
        
        results.append({
            "label": label,
            "uncertain_count": uncertain_count,
            "certain_count": certain_count,
            "precision_certain_only": p_certain,
            "recall_certain_only": r_certain,
            "f1_certain_only": f1_certain,
            "accuracy_certain_only": acc_certain,
            "precision_binary_mode": p_binary,
            "recall_binary_mode": r_binary,
            "f1_binary_mode": f1_binary,
            "accuracy_binary_mode": acc_binary,
        })
        
        print(f"\n  {label}:")
        print(f"    Uncertain in GT: {uncertain_count} ({uncertain_count/len(merged)*100:.1f}%)")
        print(f"    Certain only - P: {p_certain:.3f}, R: {r_certain:.3f}, F1: {f1_certain:.3f}")
        print(f"    Binary mode (-1→0) - P: {p_binary:.3f}, R: {r_binary:.3f}, F1: {f1_binary:.3f}")
    
    # Overall metrics
    results_df = pd.DataFrame(results)
    
    macro_p_certain = results_df["precision_certain_only"].mean()
    macro_r_certain = results_df["recall_certain_only"].mean()
    macro_f1_certain = results_df["f1_certain_only"].mean()
    
    macro_p_binary = results_df["precision_binary_mode"].mean()
    macro_r_binary = results_df["recall_binary_mode"].mean()
    macro_f1_binary = results_df["f1_binary_mode"].mean()
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Certain-only evaluation:")
    print(f"     Macro P: {macro_p_certain:.3f}, R: {macro_r_certain:.3f}, F1: {macro_f1_certain:.3f}")
    print(f"   Binary mode (uncertain→0):")
    print(f"     Macro P: {macro_p_binary:.3f}, R: {macro_r_binary:.3f}, F1: {macro_f1_binary:.3f}")
    
    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    
    # Save summary
    summary_csv = output_csv.with_name(output_csv.stem + "_summary.csv")
    summary_df = pd.DataFrame([{
        "metric": "macro_precision_certain_only",
        "value": macro_p_certain,
    }, {
        "metric": "macro_recall_certain_only",
        "value": macro_r_certain,
    }, {
        "metric": "macro_f1_certain_only",
        "value": macro_f1_certain,
    }, {
        "metric": "macro_precision_binary_mode",
        "value": macro_p_binary,
    }, {
        "metric": "macro_recall_binary_mode",
        "value": macro_r_binary,
    }, {
        "metric": "macro_f1_binary_mode",
        "value": macro_f1_binary,
    }])
    summary_df.to_csv(summary_csv, index=False)
    
    print(f"\n✅ Results saved to: {output_csv}")
    print(f"✅ Summary saved to: {summary_csv}")
    
    print("\n" + "=" * 80)
    print("THREE-CLASS EVALUATION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions against three-class ground truth")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Predictions CSV (0/1)"
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        required=True,
        help="Ground truth CSV with -1/0/1 labels"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs_5k/final/three_class_evaluation.csv"),
        help="Output evaluation CSV"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    predictions_csv = (project_root / args.predictions).resolve()
    ground_truth_csv = (project_root / args.ground_truth).resolve()
    output_csv = (project_root / args.output).resolve()
    
    if not predictions_csv.exists():
        print(f"❌ Predictions CSV not found: {predictions_csv}")
        sys.exit(1)
    
    if not ground_truth_csv.exists():
        print(f"❌ Ground truth CSV not found: {ground_truth_csv}")
        sys.exit(1)
    
    evaluate_three_class(predictions_csv, ground_truth_csv, output_csv)


if __name__ == "__main__":
    main()

