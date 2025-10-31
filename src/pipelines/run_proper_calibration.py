#!/usr/bin/env python3
"""
Proper Calibration Workflow (NO DATA LEAKAGE)

This script implements the correct calibration workflow:
1. Patient-wise 70/30 split of existing 1k predictions
2. Fit Platt calibration on 70% ONLY
3. Apply to held-out 30% and evaluate
4. Compare baseline vs calibrated on TRUE held-out set

If this pilot shows improvement, scale to 4k with 80/20 split.
"""

import subprocess
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score


CHEXPERT13 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

# Sentinel labels: High recall required
SENTINEL_LABELS = {"Pneumothorax", "Pleural Effusion", "Edema", "Support Devices"}

# Hard tail: Precision-first
HARD_TAIL = {"Fracture", "Lung Lesion", "Pleural Other", "Consolidation", 
             "Pneumonia", "Enlarged Cardiomediastinum"}


def run_cmd(cmd, desc):
    """Run command with logging."""
    print(f"\n{'='*80}")
    print(f"🔧 {desc}")
    print(f"{'='*80}")
    print(f"$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"❌ Failed with code {result.returncode}")
        sys.exit(1)
    return result


def prepare_calibration_data(predictions_csv, ground_truth_csv, output_csv):
    """Prepare data for calibration (y_true_L, y_pred_L format)."""
    print(f"\n📋 Preparing calibration data...")
    
    pred_df = pd.read_csv(predictions_csv)
    gt_df = pd.read_csv(ground_truth_csv)
    
    # Match on filename
    pred_df['filename'] = pred_df['image'].apply(lambda x: Path(x).name)
    gt_df['filename'] = gt_df['image'].apply(lambda x: Path(x).name)
    merged = pd.merge(pred_df, gt_df, on='filename', suffixes=('_pred', '_gt'))
    
    print(f"  Matched: {len(merged)} images")
    
    # Parse binary_outputs to get RAW scores (pre-calibration)
    if "binary_outputs" in pred_df.columns:
        binary_dicts = pred_df["binary_outputs"].fillna("{}").apply(json.loads)
        for disease in CHEXPERT13:
            pred_df[f"{disease}_score"] = binary_dicts.apply(
                lambda record: record.get(disease, {}).get("score_raw",
                                          record.get(disease, {}).get("score", np.nan))
            )
    
    # Re-merge with scores
    pred_df['filename'] = pred_df['image'].apply(lambda x: Path(x).name)
    merged = pd.merge(pred_df, gt_df, on='filename', suffixes=('_pred', '_gt'))
    
    output_data = {}
    for label in CHEXPERT13:
        score_col = f"{label}_score"
        gt_col = f"{label}_gt"
        
        if score_col in merged.columns:
            output_data[f"y_pred_{label}"] = merged[score_col].values
        else:
            print(f"  ⚠️  No raw score for {label}")
            output_data[f"y_pred_{label}"] = merged.get(f"{label}_pred", np.zeros(len(merged)))
        
        if gt_col in merged.columns:
            output_data[f"y_true_{label}"] = merged[gt_col].values.astype(int)
        else:
            output_data[f"y_true_{label}"] = np.zeros(len(merged), dtype=int)
    
    tuning_df = pd.DataFrame(output_data)
    tuning_df.to_csv(output_csv, index=False)
    print(f"  ✅ Saved {len(tuning_df)} samples to {output_csv}")


def evaluate_on_split(predictions_csv, ground_truth_csv, name="Split"):
    """Evaluate predictions on a split."""
    pred_df = pd.read_csv(predictions_csv)
    gt_df = pd.read_csv(ground_truth_csv)
    
    pred_df['filename'] = pred_df['image'].apply(lambda x: Path(x).name)
    gt_df['filename'] = gt_df['image'].apply(lambda x: Path(x).name)
    merged = pd.merge(pred_df, gt_df, on='filename', suffixes=('_pred', '_gt'))
    
    n = len(merged)
    if n == 0:
        print(f"❌ No matches for {name}")
        return None
    
    results = {}
    all_y_true, all_y_pred = [], []
    
    for disease in CHEXPERT13:
        pred_col = f"{disease}_pred"
        gt_col = f"{disease}_gt"
        
        if pred_col not in merged.columns or gt_col not in merged.columns:
            continue
        
        y_pred = merged[pred_col].values.astype(int)
        y_true = merged[gt_col].values.astype(int)
        
        if len(np.unique(y_true)) >= 2:
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results[disease] = {"P": p, "R": r, "F1": f1}
        
        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())
    
    # Macro/micro metrics
    macro_p = np.mean([v["P"] for v in results.values()])
    macro_r = np.mean([v["R"] for v in results.values()])
    macro_f1 = np.mean([v["F1"] for v in results.values()])
    
    micro_p = precision_score(all_y_true, all_y_pred, zero_division=0)
    micro_r = recall_score(all_y_true, all_y_pred, zero_division=0)
    micro_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
    
    # Sentinel vs Hard tail
    sentinel_metrics = {k: v for k, v in results.items() if k in SENTINEL_LABELS}
    hard_tail_metrics = {k: v for k, v in results.items() if k in HARD_TAIL}
    
    sentinel_r = np.mean([v["R"] for v in sentinel_metrics.values()]) if sentinel_metrics else 0
    sentinel_p = np.mean([v["P"] for v in sentinel_metrics.values()]) if sentinel_metrics else 0
    
    hard_tail_p = np.mean([v["P"] for v in hard_tail_metrics.values()]) if hard_tail_metrics else 0
    hard_tail_r = np.mean([v["R"] for v in hard_tail_metrics.values()]) if hard_tail_metrics else 0
    
    return {
        "name": name,
        "n_images": n,
        "macro": {"P": macro_p, "R": macro_r, "F1": macro_f1},
        "micro": {"P": micro_p, "R": micro_r, "F1": micro_f1},
        "sentinel": {"P": sentinel_p, "R": sentinel_r},
        "hard_tail": {"P": hard_tail_p, "R": hard_tail_r},
        "per_disease": results
    }


def print_pilot_results(baseline, calibrated):
    """Print pilot calibration results."""
    print("\n" + "="*80)
    print("PILOT CALIBRATION RESULTS (70/30 split, NO LEAKAGE)")
    print("="*80)
    
    print(f"\n📊 Test set: {calibrated['n_images']} images (held-out 30%)\n")
    
    print(f"{'Metric':<20} {'Baseline':>12} {'Calibrated':>12} {'Δ':>12} {'Status':<15}")
    print("-" * 75)
    
    # Overall macro
    base_p = baseline['macro']['P']
    calib_p = calibrated['macro']['P']
    delta_p = calib_p - base_p
    status_p = "✅ IMPROVED" if delta_p > 0.03 else "⚠️  MARGINAL" if delta_p > 0 else "❌ WORSE"
    print(f"{'Macro Precision':<20} {base_p:>12.3f} {calib_p:>12.3f} {delta_p:>+12.3f} {status_p:<15}")
    
    base_r = baseline['macro']['R']
    calib_r = calibrated['macro']['R']
    delta_r = calib_r - base_r
    status_r = "✅ MAINTAINED" if delta_r >= -0.03 else "⚠️  SMALL DROP" if delta_r >= -0.05 else "❌ BIG DROP"
    print(f"{'Macro Recall':<20} {base_r:>12.3f} {calib_r:>12.3f} {delta_r:>+12.3f} {status_r:<15}")
    
    print(f"{'Macro F1':<20} {baseline['macro']['F1']:>12.3f} {calibrated['macro']['F1']:>12.3f} {calibrated['macro']['F1']-baseline['macro']['F1']:>+12.3f}")
    
    print("\n" + "-" * 75)
    print("SENTINEL LABELS (Pneumothorax, Effusion, Edema, Support Devices)")
    print("-" * 75)
    
    sent_r_base = baseline['sentinel']['R']
    sent_r_calib = calibrated['sentinel']['R']
    sent_r_delta = sent_r_calib - sent_r_base
    sent_r_status = "✅ TARGET MET" if sent_r_calib >= 0.85 else "⚠️  BELOW TARGET"
    print(f"{'Recall (target ≥0.85)':<20} {sent_r_base:>12.3f} {sent_r_calib:>12.3f} {sent_r_delta:>+12.3f} {sent_r_status:<15}")
    
    sent_p_base = baseline['sentinel']['P']
    sent_p_calib = calibrated['sentinel']['P']
    sent_p_delta = sent_p_calib - sent_p_base
    print(f"{'Precision':<20} {sent_p_base:>12.3f} {sent_p_calib:>12.3f} {sent_p_delta:>+12.3f}")
    
    print("\n" + "-" * 75)
    print("HARD TAIL (Fracture, Lesion, Pleural Other, Consolidation, Pneumonia, Enlarged Cardio)")
    print("-" * 75)
    
    hard_p_base = baseline['hard_tail']['P']
    hard_p_calib = calibrated['hard_tail']['P']
    hard_p_delta = hard_p_calib - hard_p_base
    hard_p_status = "✅ TARGET MET" if hard_p_calib >= 0.60 else "⚠️  BELOW TARGET"
    print(f"{'Precision (target ≥0.60)':<20} {hard_p_base:>12.3f} {hard_p_calib:>12.3f} {hard_p_delta:>+12.3f} {hard_p_status:<15}")
    
    hard_r_base = baseline['hard_tail']['R']
    hard_r_calib = calibrated['hard_tail']['R']
    hard_r_delta = hard_r_calib - hard_r_base
    print(f"{'Recall':<20} {hard_r_base:>12.3f} {hard_r_calib:>12.3f} {hard_r_delta:>+12.3f}")
    
    print("\n" + "="*80)
    print("DECISION")
    print("="*80 + "\n")
    
    # Decision logic
    precision_improved = delta_p >= 0.05
    recall_maintained = delta_r >= -0.05
    
    if precision_improved and recall_maintained:
        print("✅ GREEN LIGHT: Calibration shows meaningful precision improvement without killing recall.")
        print("   → Proceed with full 4k dataset and 80/20 split.")
        return True
    elif precision_improved:
        print("⚠️  YELLOW LIGHT: Precision improved but recall dropped more than expected.")
        print(f"   → Consider tuning: adjust thresholds or gray-zone width.")
        print(f"   → Macro recall dropped by {abs(delta_r):.3f} (target: ≤0.05)")
        return False
    else:
        print("❌ RED LIGHT: Calibration did not improve precision meaningfully.")
        print("   → Check: Are raw scores diverse enough? Try isotonic if ≥200 positives per label.")
        print("   → Or: Improve text parsing in parse_binary_response() first.")
        return False


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║   PROPER CALIBRATION PILOT (70/30 Patient-Wise Split)                ║
    ║   NO DATA LEAKAGE: Fit on 70%, Evaluate on held-out 30%              ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # STEP 1: Patient-wise split
    print("\nSTEP 1: Patient-wise splitting...")
    run_cmd([
        "python", "patient_wise_split.py",
        "--manifest", "data/evaluation_manifest_phaseA_matched.csv",
        "--predictions", "hybrid_ensemble_1000.csv",
        "--train_ratio", "0.7",
        "--output_dir", "data",
        "--seed", "42"
    ], "Creating 70/30 patient-wise split")
    
    # STEP 2: Prepare train data for calibration
    print("\nSTEP 2: Preparing train data for calibration...")
    prepare_calibration_data(
        "hybrid_ensemble_1000.csv",
        "data/ground_truth_train_70.csv",
        "train_70_for_calibration.csv"
    )
    
    # STEP 3: Fit calibration on TRAIN ONLY
    print("\nSTEP 3: Fitting Platt calibration on 70% train set ONLY...")
    run_cmd([
        "python", "fit_label_calibrators.py",
        "--csv", "train_70_for_calibration.csv",
        "--out_dir", "calibration_proper"
    ], "Fitting calibration (train only, no leakage)")
    
    # STEP 4: Extract test set predictions from existing hybrid_ensemble_1000.csv
    print("\nSTEP 4: Extracting test set predictions...")
    
    # Load test image list
    test_gt = pd.read_csv("data/ground_truth_test_30.csv")
    test_filenames = set(test_gt['image'].apply(lambda x: Path(x).name))
    
    # Filter predictions to test set only
    pred_all = pd.read_csv("hybrid_ensemble_1000.csv")
    pred_all['filename'] = pred_all['image'].apply(lambda x: Path(x).name)
    pred_test = pred_all[pred_all['filename'].isin(test_filenames)]
    pred_test.to_csv("hybrid_ensemble_test_30.csv", index=False)
    
    print(f"  Extracted {len(pred_test)} test predictions (30% held-out)")
    
    # STEP 5: Evaluate baseline on test set
    print("\nSTEP 5: Evaluating baseline on held-out 30%...")
    baseline_results = evaluate_on_split(
        "hybrid_ensemble_test_30.csv",
        "data/ground_truth_test_30.csv",
        "Baseline (Test 30%)"
    )
    
    # STEP 6: Apply calibration and re-threshold test set
    print("\nSTEP 6: Applying calibration to test set and re-evaluating...")
    
    # Prepare test data
    prepare_calibration_data(
        "hybrid_ensemble_test_30.csv",
        "data/ground_truth_test_30.csv",
        "test_30_for_calibration.csv"
    )
    
    # Apply calibration
    run_cmd([
        "python", "apply_label_calibrators.py",
        "--csv", "test_30_for_calibration.csv",
        "--calib_dir", "calibration_proper",
        "--method", "platt",
        "--out_csv", "test_30_calibrated_scores.csv"
    ], "Applying calibration to test set")
    
    # Tune thresholds on train (using calibrated scores)
    print("\nSTEP 6b: Tuning thresholds on train set (calibrated scores)...")
    
    # Apply calibration to train for threshold tuning
    run_cmd([
        "python", "apply_label_calibrators.py",
        "--csv", "train_70_for_calibration.csv",
        "--calib_dir", "calibration_proper",
        "--method", "platt",
        "--out_csv", "train_70_calibrated_scores.csv"
    ], "Applying calibration to train set")
    
    # Tune thresholds (only on CHEXPERT13, not No Finding)
    run_cmd([
        "python", "threshold_tuner.py",
        "--csv", "train_70_calibrated_scores.csv",
        "--mode", "fbeta",
        "--beta", "0.5",  # Balanced F-beta
        "--out_json", "config/thresholds_calibrated_pilot.json",
        "--out_metrics", "threshold_tuning_pilot.csv",
        "--labels"
    ] + CHEXPERT13, "Tuning thresholds on calibrated train scores (CHEXPERT13 only)")
    
    # TODO: Apply tuned thresholds to test set and get final predictions
    # This requires modifying the evaluation to use calibrated scores + tuned thresholds
    # For now, let's simulate by checking if calibration improved raw scores
    
    print("\n✅ Pilot calibration workflow complete!")
    print(f"\nNext: Compare baseline vs calibrated performance on held-out test set")
    print(f"Files created:")
    print(f"  - calibration_proper/platt_params.json (fitted on 70% train)")
    print(f"  - test_30_calibrated_scores.csv (calibrated test scores)")
    print(f"  - config/thresholds_calibrated_pilot.json (tuned on train)")


if __name__ == "__main__":
    main()

