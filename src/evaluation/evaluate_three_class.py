#!/usr/bin/env python3
"""
Comprehensive evaluation: Binary (0/1) vs Three-Class (-1/0/1) predictions.

This is the CANONICAL evaluation script for comparing prediction modes against 3-class ground truth.

Features:
1. Supports both binary predictions (0/1) and 3-class predictions (-1/0/1)
2. Evaluates in both modes:
   - Binary mode: treats -1 as 0 (allows comparison with uncertain GT)
   - Certain-only mode: ignores -1 cases (evaluates only on certain GT)
3. Provides comparison and recommendation on which mode performs better
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
        # Determine merged column names after suffixing
        pred_col = f"{label}_pred" if f"{label}_pred" in merged.columns else (label if label in merged.columns else None)
        gt_col = f"{label}_gt" if f"{label}_gt" in merged.columns else (label if label in merged.columns else None)

        if pred_col is None or gt_col is None or pred_col not in merged.columns or gt_col not in merged.columns:
            print(f"⚠️  Skipping {label}: missing columns (pred='{pred_col}', gt='{gt_col}')")
            continue
        
        y_pred = merged[pred_col]
        y_true_raw = merged[gt_col]
        
        # Handle NaN (blanks) and -1 (uncertain) in ground truth
        # Convert to numeric, preserving -1, 0, 1, and NaN
        y_true_raw = pd.to_numeric(y_true_raw, errors='coerce')
        y_pred = pd.to_numeric(y_pred, errors='coerce')
        
        # Convert to numpy arrays
        y_true_raw = y_true_raw.values
        y_pred = y_pred.values
        
        # Detect if predictions are 3-class (-1/0/1) or binary (0/1)
        pred_not_nan = ~np.isnan(y_pred)
        has_three_class_preds = pred_not_nan.any() and (y_pred[pred_not_nan] == -1).any()
        
        # For binary mode evaluation: convert to 0/1 (treat -1 as 0)
        y_pred_bin = np.where(y_pred > 0, 1, 0).astype(int)
        
        # Count blanks (NaN) and uncertain (-1) in ground truth
        blank_count = np.isnan(y_true_raw).sum()
        uncertain_count = ((y_true_raw == -1) & ~np.isnan(y_true_raw)).sum()
        certain_count = len(y_true_raw) - blank_count - uncertain_count
        
        # Mode 1: Only evaluate on certain labels (GT = 0 or 1, ignore -1 and blanks)
        # Mask out: -1 (uncertain), NaN (blanks), and NaN predictions
        certain_mask_gt = (~np.isnan(y_true_raw)) & (y_true_raw != -1)
        if has_three_class_preds:
            certain_mask_pred = ~np.isnan(y_pred) & (y_pred != -1)
        else:
            certain_mask_pred = ~np.isnan(y_pred)
        
        certain_mask = certain_mask_gt & certain_mask_pred
        if certain_mask.sum() > 0:
            y_true_certain = y_true_raw[certain_mask].astype(int)
            if has_three_class_preds:
                y_pred_certain = y_pred[certain_mask].astype(int)  # Use 3-class predictions directly
            else:
                y_pred_certain = y_pred_bin[certain_mask]  # Binary predictions
            
            p_certain = precision_score(y_true_certain, y_pred_certain, zero_division=0)
            r_certain = recall_score(y_true_certain, y_pred_certain, zero_division=0)
            f1_certain = f1_score(y_true_certain, y_pred_certain, zero_division=0)
            acc_certain = accuracy_score(y_true_certain, y_pred_certain)
        else:
            p_certain = r_certain = f1_certain = acc_certain = np.nan
        
        # Mode 2: Treat -1 as 0 for comparison (binary evaluation)
        # Mask out blanks (NaN) - don't evaluate on them
        binary_mask = ~np.isnan(y_true_raw)
        if binary_mask.sum() > 0:
            y_true_binary = np.where(y_true_raw[binary_mask] == 1, 1, 0).astype(int)
            y_pred_binary = y_pred_bin[binary_mask]
            
            p_binary = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            r_binary = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1_binary = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            acc_binary = accuracy_score(y_true_binary, y_pred_binary)
        else:
            p_binary = r_binary = f1_binary = acc_binary = np.nan
        
        # Count uncertain predictions and blanks
        pred_uncertain = ((~np.isnan(y_pred)) & (y_pred == -1)).sum() if has_three_class_preds else 0
        
        results.append({
            "label": label,
            "blank_gt_count": int(blank_count),
            "uncertain_gt_count": int(uncertain_count),
            "uncertain_pred_count": int(pred_uncertain),
            "certain_count": int(certain_count),
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
        print(f"    Blanks in GT: {blank_count} ({blank_count/len(merged)*100:.1f}%)")
        print(f"    Uncertain (-1) in GT: {uncertain_count} ({uncertain_count/len(merged)*100:.1f}%)")
        print(f"    Certain (0/1) in GT: {certain_count} ({certain_count/len(merged)*100:.1f}%)")
        if has_three_class_preds:
            print(f"    Uncertain (-1) in Pred: {pred_uncertain} ({pred_uncertain/len(merged)*100:.1f}%)")
        print(f"    Certain-only (masked) - P: {p_certain:.3f}, R: {r_certain:.3f}, F1: {f1_certain:.3f}, Acc: {acc_certain:.3f}")
        print(f"    Binary mode (-1→0, masked) - P: {p_binary:.3f}, R: {r_binary:.3f}, F1: {f1_binary:.3f}, Acc: {acc_binary:.3f}")
    
    # Overall metrics
    results_df = pd.DataFrame(results)
    
    # Compute macro averages (ignore NaN)
    macro_p_certain = results_df["precision_certain_only"].dropna().mean()
    macro_r_certain = results_df["recall_certain_only"].dropna().mean()
    macro_f1_certain = results_df["f1_certain_only"].dropna().mean()
    macro_acc_certain = results_df["accuracy_certain_only"].dropna().mean()
    
    macro_p_binary = results_df["precision_binary_mode"].dropna().mean()
    macro_r_binary = results_df["recall_binary_mode"].dropna().mean()
    macro_f1_binary = results_df["f1_binary_mode"].dropna().mean()
    macro_acc_binary = results_df["accuracy_binary_mode"].dropna().mean()
    
    print(f"\n📊 Overall Macro Metrics:")
    print(f"   Certain-only evaluation (masked -1 and blanks):")
    print(f"     Precision: {macro_p_certain:.3f}, Recall: {macro_r_certain:.3f}, F1: {macro_f1_certain:.3f}, Accuracy: {macro_acc_certain:.3f}")
    print(f"   Binary mode (-1→0, masked blanks):")
    print(f"     Precision: {macro_p_binary:.3f}, Recall: {macro_r_binary:.3f}, F1: {macro_f1_binary:.3f}, Accuracy: {macro_acc_binary:.3f}")
    
    # Recommendation based on best accuracy, precision, recall, F1
    print(f"\n💡 Recommendation (Best Performance):")
    metrics_certain = {
        "precision": macro_p_certain,
        "recall": macro_r_certain,
        "f1": macro_f1_certain,
        "accuracy": macro_acc_certain,
    }
    metrics_binary = {
        "precision": macro_p_binary,
        "recall": macro_r_binary,
        "f1": macro_f1_binary,
        "accuracy": macro_acc_binary,
    }
    
    wins_certain = sum(1 for m in ["precision", "recall", "f1", "accuracy"] 
                       if metrics_certain[m] > metrics_binary[m])
    wins_binary = 4 - wins_certain
    
    if wins_certain > wins_binary:
        print(f"   ✅ USE CERTAIN-ONLY MODE (masks -1 and blanks)")
        print(f"   - Wins {wins_certain}/4 metrics: P={macro_p_certain:.3f} vs {macro_p_binary:.3f}, "
              f"R={macro_r_certain:.3f} vs {macro_r_binary:.3f}, "
              f"F1={macro_f1_certain:.3f} vs {macro_f1_binary:.3f}, "
              f"Acc={macro_acc_certain:.3f} vs {macro_acc_binary:.3f}")
        print(f"   - Better clinical interpretation (uncertainty/blanks excluded from evaluation)")
    elif wins_binary > wins_certain:
        print(f"   ✅ USE BINARY MODE (-1→0, masks blanks)")
        print(f"   - Wins {wins_binary}/4 metrics: P={macro_p_binary:.3f} vs {macro_p_certain:.3f}, "
              f"R={macro_r_binary:.3f} vs {macro_r_certain:.3f}, "
              f"F1={macro_f1_binary:.3f} vs {macro_f1_certain:.3f}, "
              f"Acc={macro_acc_binary:.3f} vs {macro_acc_certain:.3f}")
        print(f"   - Simpler interpretation (uncertain treated as negative)")
    else:
        print(f"   ⚖️  BOTH MODES SIMILAR (tied {wins_certain}-{wins_binary})")
        print(f"   - Use binary for simplicity")
        print(f"   - Use certain-only if uncertainty interpretation is important")
    
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
