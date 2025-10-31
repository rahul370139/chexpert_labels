#!/usr/bin/env python3
"""
Compare results before/after calibration + precision gating.

Usage:
    python compare_calibration_results.py \
        --baseline hybrid_ensemble_1000.csv \
        --calibrated hybrid_ensemble_1000_calibrated.csv \
        --ground_truth data/evaluation_manifest_phaseA_matched.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

CHEXPERT13 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]


def evaluate_predictions(pred_csv, gt_csv, name="Results"):
    """Evaluate predictions against ground truth."""
    pred_df = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv)
    
    # Match on filename
    pred_df['filename'] = pred_df['image'].apply(lambda x: Path(x).name)
    gt_df['filename'] = gt_df['image'].apply(lambda x: Path(x).name)
    merged = pd.merge(pred_df, gt_df, on='filename', suffixes=('_pred', '_gt'))
    
    n_matched = len(merged)
    if n_matched == 0:
        print(f"❌ No matches for {name}")
        return None
    
    # Calculate per-label metrics
    rows = []
    all_y_true, all_y_pred = [], []
    
    for disease in CHEXPERT13:
        pred_col = f"{disease}_pred"
        gt_col = f"{disease}_gt"
        
        if pred_col not in merged.columns or gt_col not in merged.columns:
            continue
        
        y_pred = merged[pred_col].values.astype(int)
        y_true = merged[gt_col].values.astype(int)
        
        if len(np.unique(y_true)) >= 2:  # At least 2 classes
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            
            rows.append({
                "Disease": disease,
                "Precision": p,
                "Recall": r,
                "F1": f1,
                "Accuracy": acc,
                "N_Pos_GT": int(y_true.sum()),
                "N_Pos_Pred": int(y_pred.sum())
            })
        
        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())
    
    results_df = pd.DataFrame(rows)
    
    # Macro metrics (13 diseases)
    macro_p = results_df["Precision"].mean()
    macro_r = results_df["Recall"].mean()
    macro_f1 = results_df["F1"].mean()
    macro_acc = results_df["Accuracy"].mean()
    
    # Micro metrics (13 diseases)
    micro_p = precision_score(all_y_true, all_y_pred, zero_division=0)
    micro_r = recall_score(all_y_true, all_y_pred, zero_division=0)
    micro_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    
    # No Finding
    nf_pred = (merged[[f"{d}_pred" for d in CHEXPERT13 if f"{d}_pred" in merged.columns]].sum(axis=1) == 0).astype(int)
    nf_gt = (merged[[f"{d}_gt" for d in CHEXPERT13 if f"{d}_gt" in merged.columns]].sum(axis=1) == 0).astype(int)
    
    nf_p = precision_score(nf_gt, nf_pred, zero_division=0)
    nf_r = recall_score(nf_gt, nf_pred, zero_division=0)
    nf_f1 = f1_score(nf_gt, nf_pred, zero_division=0)
    nf_acc = accuracy_score(nf_gt, nf_pred)
    
    # Overall (14 including No Finding)
    overall_p = (macro_p + nf_p) / 2
    overall_r = (macro_r + nf_r) / 2
    overall_f1 = (macro_f1 + nf_f1) / 2
    overall_acc = (macro_acc + nf_acc) / 2
    
    return {
        "name": name,
        "n_images": n_matched,
        "per_disease": results_df,
        "macro_13": {"P": macro_p, "R": macro_r, "F1": macro_f1, "Acc": macro_acc},
        "micro_13": {"P": micro_p, "R": micro_r, "F1": micro_f1},
        "no_finding": {"P": nf_p, "R": nf_r, "F1": nf_f1, "Acc": nf_acc},
        "overall_14": {"P": overall_p, "R": overall_r, "F1": overall_f1, "Acc": overall_acc}
    }


def print_comparison(baseline, calibrated):
    """Print side-by-side comparison."""
    print("\n" + "="*80)
    print("CALIBRATION IMPACT ANALYSIS")
    print("="*80)
    
    print(f"\n📊 Dataset: {baseline['n_images']} images matched\n")
    
    # Overall metrics
    print("OVERALL (CHEXPERT13 - 13 diseases):")
    print("-" * 60)
    print(f"{'Metric':<15} {'Baseline':>12} {'Calibrated':>12} {'Δ':>12}")
    print("-" * 60)
    
    for metric in ["P", "R", "F1", "Acc"]:
        base_val = baseline['macro_13'][metric]
        calib_val = calibrated['macro_13'][metric]
        delta = calib_val - base_val
        delta_pct = (delta / base_val * 100) if base_val > 0 else 0
        
        print(f"Macro {metric:<8} {base_val:>12.3f} {calib_val:>12.3f} {delta:>+12.3f} ({delta_pct:+.1f}%)")
    
    print()
    for metric in ["P", "R", "F1"]:
        base_val = baseline['micro_13'][metric]
        calib_val = calibrated['micro_13'][metric]
        delta = calib_val - base_val
        delta_pct = (delta / base_val * 100) if base_val > 0 else 0
        
        print(f"Micro {metric:<8} {base_val:>12.3f} {calib_val:>12.3f} {delta:>+12.3f} ({delta_pct:+.1f}%)")
    
    print("\n" + "-" * 60)
    print("NO FINDING:")
    print("-" * 60)
    
    for metric in ["P", "R", "F1", "Acc"]:
        base_val = baseline['no_finding'][metric]
        calib_val = calibrated['no_finding'][metric]
        delta = calib_val - base_val
        delta_pct = (delta / base_val * 100) if base_val > 0 else 0
        
        print(f"{metric:<15} {base_val:>12.3f} {calib_val:>12.3f} {delta:>+12.3f} ({delta_pct:+.1f}%)")
    
    # Per-disease comparison (top improvements and regressions)
    print("\n" + "="*80)
    print("PER-DISEASE PRECISION IMPROVEMENTS")
    print("="*80)
    
    base_diseases = baseline['per_disease'].set_index('Disease')
    calib_diseases = calibrated['per_disease'].set_index('Disease')
    
    precision_deltas = []
    for disease in base_diseases.index:
        if disease in calib_diseases.index:
            base_p = base_diseases.loc[disease, 'Precision']
            calib_p = calib_diseases.loc[disease, 'Precision']
            delta = calib_p - base_p
            precision_deltas.append((disease, base_p, calib_p, delta))
    
    precision_deltas.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\n{'Disease':<30} {'Baseline P':>12} {'Calibrated P':>12} {'Δ':>12}")
    print("-" * 70)
    for disease, base_p, calib_p, delta in precision_deltas[:10]:
        print(f"{disease:<30} {base_p:>12.3f} {calib_p:>12.3f} {delta:>+12.3f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Calculate key metrics
    base_macro_p = baseline['macro_13']['P']
    calib_macro_p = calibrated['macro_13']['P']
    p_improvement = ((calib_macro_p - base_macro_p) / base_macro_p * 100) if base_macro_p > 0 else 0
    
    base_macro_r = baseline['macro_13']['R']
    calib_macro_r = calibrated['macro_13']['R']
    r_change = ((calib_macro_r - base_macro_r) / base_macro_r * 100) if base_macro_r > 0 else 0
    
    print(f"\n✅ Macro Precision: {base_macro_p:.3f} → {calib_macro_p:.3f} ({p_improvement:+.1f}%)")
    print(f"✅ Macro Recall: {base_macro_r:.3f} → {calib_macro_r:.3f} ({r_change:+.1f}%)")
    print(f"✅ No Finding Recall: {baseline['no_finding']['R']:.3f} → {calibrated['no_finding']['R']:.3f}")
    
    if calib_macro_p >= 0.60:
        print(f"\n🎯 TARGET ACHIEVED: Macro Precision ≥ 0.60")
    else:
        print(f"\n⚠️  TARGET NOT MET: Macro Precision = {calib_macro_p:.3f} (goal: ≥ 0.60)")
    
    if calib_macro_r >= 0.55:
        print(f"🎯 TARGET ACHIEVED: Macro Recall ≥ 0.55")
    else:
        print(f"⚠️  Macro Recall = {calib_macro_r:.3f} (goal: ≥ 0.55)")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs calibrated results")
    parser.add_argument("--baseline", required=True, help="Baseline predictions CSV")
    parser.add_argument("--calibrated", required=True, help="Calibrated predictions CSV")
    parser.add_argument("--ground_truth", required=True, help="Ground truth CSV")
    
    args = parser.parse_args()
    
    print("\n📋 Evaluating baseline...")
    baseline = evaluate_predictions(args.baseline, args.ground_truth, "Baseline")
    
    print("\n📋 Evaluating calibrated...")
    calibrated = evaluate_predictions(args.calibrated, args.ground_truth, "Calibrated")
    
    if baseline and calibrated:
        print_comparison(baseline, calibrated)
    else:
        print("❌ Could not complete comparison")


if __name__ == "__main__":
    main()

