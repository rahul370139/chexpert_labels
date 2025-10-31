#!/usr/bin/env python3
"""
Master orchestrator for TXR + CheXagent ensemble pipeline.

Runs end-to-end with error handling:
1. Monitor CheXagent completion
2. Prepare CheXagent probability tables
3. Blend TXR + CheXagent (per-label logistic regression)
4. Calibrate blended probabilities
5. Tune thresholds on calibrated blend
6. Evaluate on held-out test set

Usage:
    python run_full_ensemble_pipeline.py
"""

import subprocess
import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import warnings

CHEXPERT14 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]

CHEXPERT13 = CHEXPERT14[:-1]


def run_cmd(cmd, desc, check=True, timeout=None):
    """Run command with logging."""
    print(f"\n{'='*80}")
    print(f"🔧 {desc}")
    print(f"{'='*80}")
    if isinstance(cmd, str):
        cmd = cmd.split()
    print(f"$ {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and "Warning" not in result.stderr:
            print(result.stderr, file=sys.stderr)
        return result
    except subprocess.TimeoutExpired:
        print(f"❌ Command timed out after {timeout}s")
        if check:
            sys.exit(1)
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with code {e.returncode}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        if check:
            sys.exit(1)
        return None


def wait_for_chexagent(log_path, timeout_hours=4):
    """Wait for CheXagent run to complete."""
    print(f"\n⏳ Waiting for CheXagent hybrid run to complete...")
    print(f"   Log: {log_path}")
    
    start_time = time.time()
    timeout_seconds = timeout_hours * 3600
    
    while True:
        if not Path(log_path).exists():
            print(f"⚠️  Log file not found: {log_path}")
            time.sleep(60)
            continue
        
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        # Check for completion
        if any("🎉 Smart ensemble processing complete!" in line for line in lines):
            # Get total images processed
            for line in reversed(lines):
                if "Processed" in line and "images" in line:
                    print(f"✅ CheXagent completed: {line.strip()}")
                    return True
        
        # Check for progress
        processing_lines = [l for l in lines if "Processing" in l]
        if processing_lines:
            last_line = processing_lines[-1]
            print(f"   Progress: {last_line.strip()}")
        
        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            print(f"⏰ Timeout after {timeout_hours} hours")
            return False
        
        time.sleep(120)  # Check every 2 minutes


def prepare_chex_probabilities(pred_csv, gt_csv, output_csv):
    """Convert CheXagent predictions to probability format for blending."""
    print(f"\n📊 Preparing CheXagent probabilities...")
    
    pred_df = pd.read_csv(pred_csv)
    gt_df = pd.read_csv(gt_csv)
    
    # Match on filename
    pred_df['filename'] = pred_df['image'].apply(lambda x: Path(x).name)
    gt_df['filename'] = gt_df['image'].apply(lambda x: Path(x).name)
    
    # Load train/test split
    train_files = set(Path("data_full/train_images_70.txt").read_text().strip().split('\n'))
    test_files = set(Path("data_full/test_images_30.txt").read_text().strip().split('\n'))
    
    # Parse binary_outputs to get scores
    import json
    output_data = []
    
    for _, row in pred_df.iterrows():
        filename = row['filename']
        record = {'filename': filename}
        
        # Ground truth
        gt_row = gt_df[gt_df['filename'] == filename]
        if len(gt_row) > 0:
            for label in CHEXPERT14:
                gt_col = f"{label}_gt" if f"{label}_gt" in gt_row.columns else label
                if gt_col in gt_row.columns:
                    record[f"y_true_{label}"] = int(gt_row[gt_col].iloc[0])
                else:
                    record[f"y_true_{label}"] = 0
        
        # Predictions (scores from binary_outputs)
        binary_outputs = json.loads(row.get('binary_outputs', '{}'))
        for label in CHEXPERT13:
            if label in binary_outputs:
                score = binary_outputs[label].get('score', binary_outputs[label].get('score_raw', 0.5))
                record[f"y_pred_{label}"] = float(score)
            else:
                record[f"y_pred_{label}"] = 0.5
        
        # No Finding (inverse of other findings)
        other_scores = [record.get(f"y_pred_{l}", 0.5) for l in CHEXPERT13]
        max_other = max(other_scores) if other_scores else 0.5
        record["y_pred_No Finding"] = max(0.0, 1.0 - max_other)
        
        output_data.append(record)
    
    df = pd.DataFrame(output_data)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} records to {output_csv}")
    
    return df


def blend_probabilities(txr_train_csv, txr_test_csv, chex_train_df, chex_test_df, output_dir):
    """Blend TXR + CheXagent probabilities using per-label logistic regression."""
    print(f"\n🔀 Blending TXR + CheXagent probabilities...")
    
    txr_train = pd.read_csv(txr_train_csv).set_index('filename')
    txr_test = pd.read_csv(txr_test_csv).set_index('filename')
    chex_train = chex_train_df.set_index('filename')
    chex_test = chex_test_df.set_index('filename')
    
    # Ensure consistent index
    common_train = set(txr_train.index) & set(chex_train.index)
    common_test = set(txr_test.index) & set(chex_test.index)
    
    print(f"   Train: {len(common_train)} images")
    print(f"   Test: {len(common_test)} images")
    
    blend_train = txr_train.loc[list(common_train)][[f"y_true_{l}" for l in CHEXPERT14]].copy()
    blend_test = txr_test.loc[list(common_test)][[f"y_true_{l}" for l in CHEXPERT14]].copy()
    
    for label in CHEXPERT14:
        txr_col = f"y_pred_{label}"
        chex_col = f"y_pred_{label}"
        
        # Get probabilities
        txr_train_vals = txr_train.loc[list(common_train)][txr_col].fillna(0.5).values
        txr_test_vals = txr_test.loc[list(common_test)][txr_col].fillna(0.5).values
        chex_train_vals = chex_train.loc[list(common_train)][chex_col].fillna(0.5).values
        chex_test_vals = chex_test.loc[list(common_test)][chex_col].fillna(0.5).values
        
        # Create feature matrix
        X_train = np.column_stack([txr_train_vals, chex_train_vals])
        X_test = np.column_stack([txr_test_vals, chex_test_vals])
        y_train = blend_train[f"y_true_{label}"].values
        
        # Check if both features are constant
        if np.allclose(X_train, X_train[0]):
            # Fallback to TXR if constant
            print(f"   {label}: Constant features, using TXR only")
            blend_train[f"y_pred_{label}"] = txr_train_vals
            blend_test[f"y_pred_{label}"] = txr_test_vals
        else:
            # Fit logistic regression
            clf = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                try:
                    clf.fit(X_train, y_train)
                    blend_train[f"y_pred_{label}"] = clf.predict_proba(X_train)[:, 1]
                    blend_test[f"y_pred_{label}"] = clf.predict_proba(X_test)[:, 1]
                    print(f"   {label}: Blended (coef={clf.coef_[0]})")
                except Exception as e:
                    print(f"   {label}: Blend failed ({e}), using TXR")
                    blend_train[f"y_pred_{label}"] = txr_train_vals
                    blend_test[f"y_pred_{label}"] = txr_test_vals
    
    # Reset index and save
    blend_train.reset_index(inplace=True)
    blend_test.reset_index(inplace=True)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    blend_train.to_csv(f"{output_dir}/train_blend_probs.csv", index=False)
    blend_test.to_csv(f"{output_dir}/test_blend_probs.csv", index=False)
    
    print(f"✅ Saved blended probabilities to {output_dir}/")
    return blend_train, blend_test


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║   FULL ENSEMBLE PIPELINE: TXR + CheXagent                           ║
    ║   Monitoring, Blending, Calibration, Threshold Tuning, Evaluation   ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # STEP 1: Wait for CheXagent to complete
    chex_log = Path("chex_full/hybrid_full.log")
    chex_csv = Path("chex_full/hybrid_full.csv")
    
    if not chex_csv.exists():
        print("⏳ CheXagent run in progress, waiting for completion...")
        if wait_for_chexagent(str(chex_log), timeout_hours=4):
            print("✅ CheXagent completed!")
        else:
            print("⚠️  CheXagent not complete, but continuing with available data...")
    
    # STEP 2: Prepare CheXagent probabilities
    if not chex_csv.exists():
        print("❌ CheXagent output not found, cannot proceed with blending")
        sys.exit(1)
    
    chex_probs_csv = Path("chex_full/chex_probs.csv")
    if not chex_probs_csv.exists():
        chex_df = prepare_chex_probabilities(
            chex_csv,
            "data/evaluation_manifest_phaseA_full.csv",
            str(chex_probs_csv)
        )
    else:
        print("✅ CheXagent probabilities already prepared")
        chex_df = pd.read_csv(chex_probs_csv)
    
    # Split CheXagent data
    train_files = set(Path("data_full/train_images_70.txt").read_text().strip().split('\n'))
    test_files = set(Path("data_full/test_images_30.txt").read_text().strip().split('\n'))
    
    chex_df['filename'] = chex_df['filename'].apply(lambda x: Path(x).name if Path(x).name else x)
    chex_train_df = chex_df[chex_df['filename'].isin(train_files)].copy()
    chex_test_df = chex_df[chex_df['filename'].isin(test_files)].copy()
    
    # STEP 3: Blend TXR + CheXagent
    output_dir = "ensemble_pipeline"
    blend_train, blend_test = blend_probabilities(
        "txr_full/train_calibrated.csv",
        "txr_full/test_calibrated.csv",
        chex_train_df,
        chex_test_df,
        output_dir
    )
    
    # STEP 4: Calibrate blended probabilities
    print(f"\n📊 Calibrating blended probabilities...")
    run_cmd([
        "python", "fit_label_calibrators.py",
        "--csv", f"{output_dir}/train_blend_probs.csv",
        "--out_dir", f"{output_dir}/calibration_blend"
    ], "Fitting calibration on blended train set")
    
    run_cmd([
        "python", "apply_label_calibrators.py",
        "--csv", f"{output_dir}/train_blend_probs.csv",
        "--calib_dir", f"{output_dir}/calibration_blend",
        "--out_csv", f"{output_dir}/train_blend_calibrated.csv"
    ], "Applying calibration to train blend")
    
    run_cmd([
        "python", "apply_label_calibrators.py",
        "--csv", f"{output_dir}/test_blend_probs.csv",
        "--calib_dir", f"{output_dir}/calibration_blend",
        "--out_csv", f"{output_dir}/test_blend_calibrated.csv"
    ], "Applying calibration to test blend")
    
    # STEP 5: Tune thresholds
    run_cmd([
        "python", "threshold_tuner.py",
        "--csv", f"{output_dir}/train_blend_calibrated.csv",
        "--mode", "fbeta",
        "--beta", "0.3",
        "--out_json", f"{output_dir}/thresholds_blend.json",
        "--out_metrics", f"{output_dir}/thresholds_blend_metrics.csv"
    ], "Tuning thresholds on calibrated blend (F-beta, β=0.3)")
    
    # STEP 6: Evaluate on test set
    run_cmd([
        "python", "evaluate_prob_predictions.py",
        "--predictions", f"{output_dir}/test_blend_calibrated.csv",
        "--ground_truth", "data_full/ground_truth_test_30.csv",
        "--thresholds", f"{output_dir}/thresholds_blend.json",
        "--score_prefix", "y_cal_",
        "--out_metrics", f"{output_dir}/test_blend_metrics.csv"
    ], "Evaluating blended model on held-out test set")
    
    print(f"\n✅ FULL PIPELINE COMPLETE!")
    print(f"\n📊 Final metrics: {output_dir}/test_blend_metrics.csv")
    print(f"📊 Thresholds: {output_dir}/thresholds_blend.json")
    print(f"📊 Per-label metrics: {output_dir}/thresholds_blend_metrics.csv")


if __name__ == "__main__":
    main()

