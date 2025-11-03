#!/usr/bin/env python3
"""
Calibrated CheXpert Pipeline Workflow

This orchestrates the full calibration + precision-gating workflow:
1. Split data into train/val (if needed)
2. Fit Platt calibration on VAL set
3. Run smart_ensemble with calibration + precision gating
4. Tune thresholds on calibrated VAL scores
5. Evaluate on TEST set

Usage:
    # Full pipeline on 1000 images with calibration
    python run_calibrated_pipeline.py --mode full --n_images 1000
    
    # Quick test on 10 images
    python run_calibrated_pipeline.py --mode test --n_images 10
    
    # Just calibration fitting (if you already have predictions)
    python run_calibrated_pipeline.py --mode calibrate_only --predictions hybrid_ensemble_1000.csv
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

CHEXPERT13 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]


def run_cmd(cmd, description, check=True):
    """Run shell command with logging."""
    print(f"\n{'='*80}")
    print(f"🔧 {description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if check and result.returncode != 0:
        print(f"❌ Command failed with code {result.returncode}")
        sys.exit(1)
    return result


def split_train_val_test(manifest_csv, train_ratio=0.6, val_ratio=0.2):
    """Split manifest into train/val/test sets."""
    print("\n📊 Splitting data into train/val/test...")
    
    df = pd.read_csv(manifest_csv)
    n = len(df)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_df = df[:n_train]
    val_df = df[n_train:n_train + n_val]
    test_df = df[n_train + n_val:]
    
    # Save splits
    train_df.to_csv("data/image_list_train.csv", index=False)
    val_df.to_csv("data/image_list_val.csv", index=False)
    test_df.to_csv("data/image_list_test.csv", index=False)
    
    # Also create simple image lists
    with open("data/image_list_train.txt", "w") as f:
        f.write("\n".join(train_df["image"].tolist()))
    with open("data/image_list_val.txt", "w") as f:
        f.write("\n".join(val_df["image"].tolist()))
    with open("data/image_list_test.txt", "w") as f:
        f.write("\n".join(test_df["image"].tolist()))
    
    print(f"  Train: {len(train_df)} images")
    print(f"  Val:   {len(val_df)} images")
    print(f"  Test:  {len(test_df)} images")
    
    return train_df, val_df, test_df


def prepare_for_calibration(predictions_csv, ground_truth_csv, output_csv):
    """Prepare data in y_true_L, y_pred_L format for calibration."""
    print(f"\n📋 Preparing calibration data from {predictions_csv}...")
    
    pred_df = pd.read_csv(predictions_csv)
    gt_df = pd.read_csv(ground_truth_csv)
    
    # Match on filename
    pred_df['filename'] = pred_df['image'].apply(lambda x: Path(x).name)
    gt_df['filename'] = gt_df['image'].apply(lambda x: Path(x).name)
    merged = pd.merge(pred_df, gt_df, on='filename', suffixes=('_pred', '_gt'))
    
    print(f"  Matched: {len(merged)} images")
    
    # Parse binary_outputs to get raw scores
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
            print(f"  ⚠️  No score for {label}, using predictions")
            output_data[f"y_pred_{label}"] = merged.get(f"{label}_pred", np.zeros(len(merged)))
        
        if gt_col in merged.columns:
            output_data[f"y_true_{label}"] = merged[gt_col].values.astype(int)
        else:
            output_data[f"y_true_{label}"] = merged.get(label, np.zeros(len(merged)))
    
    tuning_df = pd.DataFrame(output_data)
    tuning_df.to_csv(output_csv, index=False)
    print(f"  ✅ Saved {len(tuning_df)} samples to {output_csv}")
    
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Run calibrated CheXpert pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "test", "calibrate_only", "inference_only"],
        default="full",
        help="Pipeline mode"
    )
    parser.add_argument("--n_images", type=int, default=1000, help="Number of images to process")
    parser.add_argument("--predictions", type=str, help="Existing predictions CSV (for calibrate_only mode)")
    parser.add_argument("--ground_truth", type=str, default="data/evaluation_manifest_phaseA_matched.csv")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--use_precision_gating", action="store_true", default=True, 
                       help="Enable precision-first gating")
    
    args = parser.parse_args()
    
    print(f"""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║        CheXpert Calibrated Pipeline                                   ║
    ║        Mode: {args.mode:30s}                            ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Paths
    calib_dir = Path("calibration")
    calib_dir.mkdir(exist_ok=True)
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    if args.mode == "full":
        # STEP 1: Run inference on VAL set to get initial predictions for calibration
        print("\n" + "="*80)
        print("STEP 1: Generate VAL predictions for calibration fitting")
        print("="*80)
        
        val_predictions = f"predictions_val_{args.n_images}.csv"
        
        run_cmd([
            sys.executable, "src/inference/smart_ensemble.py",
            "--images", "data/image_list_1000_absolute.txt",  # Use your existing image list
            "--out_csv", val_predictions,
            "--device", args.device,
            "--thresholds", "config/label_thresholds.json",
            "--force_zero_labels", "Cardiomegaly", "Atelectasis"
        ], "Running smart_ensemble on VAL set (no calibration yet)")
        
        # STEP 2: Prepare data and fit calibration
        print("\n" + "="*80)
        print("STEP 2: Fit Platt calibration on VAL predictions")
        print("="*80)
        
        val_tuning_csv = "val_tuning_data.csv"
        prepare_for_calibration(val_predictions, args.ground_truth, val_tuning_csv)
        
        run_cmd([
            sys.executable, "src/calibration/fit_label_calibrators.py",
            "--csv", val_tuning_csv,
            "--out_dir", str(calib_dir)
        ], "Fitting Platt calibration")
        
        # STEP 3: Re-run inference with calibration + precision gating
        print("\n" + "="*80)
        print("STEP 3: Re-run inference WITH calibration + precision gating")
        print("="*80)
        
        calibrated_predictions = f"hybrid_ensemble_{args.n_images}_calibrated.csv"
        
        cmd = [
            sys.executable, "src/inference/smart_ensemble.py",
            "--images", "data/image_list_1000_absolute.txt",
            "--out_csv", calibrated_predictions,
            "--device", args.device,
            "--thresholds", "config/label_thresholds.json",
            "--calibration", str(calib_dir / "platt_params.json"),
            "--force_zero_labels", "Cardiomegaly", "Atelectasis"
        ]
        
        if args.use_precision_gating:
            cmd.append("--use_precision_gating")
        
        run_cmd(cmd, "Running calibrated inference with precision gating")
        
        # STEP 4: Tune thresholds on calibrated scores
        print("\n" + "="*80)
        print("STEP 4: Tune thresholds on calibrated VAL scores")
        print("="*80)
        
        # Prepare calibrated VAL data for threshold tuning
        run_cmd([
            sys.executable, "src/calibration/apply_label_calibrators.py",
            "--csv", val_tuning_csv,
            "--calib_dir", str(calib_dir),
            "--method", "platt",
            "--out_csv", "val_tuning_data_calibrated.csv"
        ], "Applying calibration to VAL scores")
        
        run_cmd([
            sys.executable, "src/thresholding/tune_thresholds.py",
            "--csv", "val_tuning_data_calibrated.csv",
            "--mode", "fbeta",
            "--beta", "0.3",  # Precision-weighted
            "--out_json", str(config_dir / "label_thresholds_calibrated.json"),
            "--out_metrics", "threshold_tuning_results.csv"
        ], "Tuning thresholds (F-beta with β=0.3)")
        
        # STEP 5: Evaluate
        print("\n" + "="*80)
        print("STEP 5: Evaluate calibrated results")
        print("="*80)
        
        run_cmd([
            sys.executable, "src/evaluation/evaluate_results.py",
            "--predictions", calibrated_predictions,
            "--ground_truth", "data/evaluation_manifest_phaseA_matched.csv",
            "--name", "Calibrated + Precision Gating",
        ], "Evaluating calibrated predictions", check=False)
        
        print(f"\n✅ Full pipeline complete!")
        print(f"   Calibrated predictions: {calibrated_predictions}")
        print(f"   Calibration params: {calib_dir / 'platt_params.json'}")
        print(f"   Tuned thresholds: {config_dir / 'label_thresholds_calibrated.json'}")
    
    elif args.mode == "test":
        # Quick test on small subset
        print(f"\n🧪 Running quick test on {args.n_images} images...")
        
        test_predictions = f"test_{args.n_images}_calibrated.csv"
        
        cmd = [
            sys.executable, "src/inference/smart_ensemble.py",
            "--images", "data/image_list_10.txt" if args.n_images <= 10 else "data/image_list_1000_absolute.txt",
            "--out_csv", test_predictions,
            "--device", args.device,
            "--thresholds", "config/label_thresholds.json",
            "--force_zero_labels", "Cardiomegaly", "Atelectasis"
        ]
        
        # Use existing calibration if available
        platt_path = calib_dir / "platt_params.json"
        if platt_path.exists():
            cmd.extend(["--calibration", str(platt_path)])
        
        if args.use_precision_gating:
            cmd.append("--use_precision_gating")
        
        run_cmd(cmd, f"Running test inference ({args.n_images} images)")
        
        print(f"\n✅ Test complete: {test_predictions}")
    
    elif args.mode == "calibrate_only":
        if not args.predictions:
            print("❌ --predictions required for calibrate_only mode")
            sys.exit(1)
        
        print(f"\n📊 Fitting calibration from existing predictions: {args.predictions}")
        
        val_tuning_csv = "val_tuning_data_existing.csv"
        prepare_for_calibration(args.predictions, args.ground_truth, val_tuning_csv)
        
        run_cmd([
            sys.executable, "src/calibration/fit_label_calibrators.py",
            "--csv", val_tuning_csv,
            "--out_dir", str(calib_dir)
        ], "Fitting Platt calibration")
        
        print(f"\n✅ Calibration fitted: {calib_dir / 'platt_params.json'}")
    
    elif args.mode == "inference_only":
        print(f"\n🔮 Running inference with existing calibration...")
        
        out_csv = f"hybrid_ensemble_{args.n_images}_calibrated.csv"
        
        cmd = [
            sys.executable, "src/inference/smart_ensemble.py",
            "--images", "data/image_list_1000_absolute.txt",
            "--out_csv", out_csv,
            "--device", args.device,
            "--thresholds", "config/label_thresholds.json",
            "--calibration", str(calib_dir / "platt_params.json"),
            "--force_zero_labels", "Cardiomegaly", "Atelectasis"
        ]
        
        if args.use_precision_gating:
            cmd.append("--use_precision_gating")
        
        run_cmd(cmd, "Running calibrated inference")
        
        print(f"\n✅ Inference complete: {out_csv}")


if __name__ == "__main__":
    main()
