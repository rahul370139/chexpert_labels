#!/usr/bin/env python3
"""
End-to-end calibration pipeline using TorchXRayVision probabilities.

Steps:
 1. Run txr_infer.py to obtain continuous probabilities
 2. Perform patient-wise split (70/30 by default)
 3. Prepare calibration inputs for train/test splits
 4. Fit Platt calibrators on the train split
 5. Apply calibrators to train + test predictions
 6. Tune thresholds on calibrated train probabilities
 7. Evaluate on held-out test split

Optional: blend calibrated TXR probabilities with CheXagent calibrated outputs
          using blend_probabilities.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CHEXPERT14 = [
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
    "No Finding",
]


def run(cmd, description):
    print("\n" + "=" * 90)
    print(f"🔧 {description}")
    print("=" * 90)
    print(" ".join(str(c) for c in cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Command failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="TorchXRayVision calibration pipeline")
    parser.add_argument("--images", required=True, help="Image list (.txt/.csv) or directory for inference")
    parser.add_argument("--ground_truth", default="data/evaluation_manifest_phaseA_matched.csv")
    parser.add_argument("--output_dir", default="txr_pipeline")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3, help="F-beta for threshold tuning (precision emphasis if <1)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txr_predictions = output_dir / "txr_predictions.csv"
    train_gt = Path("data/ground_truth_train_70.csv")
    test_gt = Path("data/ground_truth_test_30.csv")

    print("\n" + "=" * 90)
    print("TorchXRayVision Calibration Pipeline")
    print("=" * 90)

    # 1. Run TXR inference
    run(
        [
            sys.executable,
            "txr_infer.py",
            "--images",
            args.images,
            "--out_csv",
            str(txr_predictions),
            "--device",
            args.device,
        ],
        "Running TorchXRayVision inference",
    )

    # 2. Patient-wise split (avoids leakage)
    run(
        [
            sys.executable,
            "patient_wise_split.py",
            "--manifest",
            args.ground_truth,
            "--predictions",
            str(txr_predictions),
            "--train_ratio",
            str(args.train_ratio),
            "--output_dir",
            "data",
        ],
        "Patient-wise train/test split",
    )

    # 3. Prepare calibration inputs
    train_scores = output_dir / "train_txr_scores.csv"
    test_scores = output_dir / "test_txr_scores.csv"

    run(
        [
            sys.executable,
            "prepare_predictions_for_calibration.py",
            "--predictions",
            str(txr_predictions),
            "--ground_truth",
            str(train_gt),
            "--output",
            str(train_scores),
        ],
        "Preparing train calibration input",
    )

    run(
        [
            sys.executable,
            "prepare_predictions_for_calibration.py",
            "--predictions",
            str(txr_predictions),
            "--ground_truth",
            str(test_gt),
            "--output",
            str(test_scores),
        ],
        "Preparing test calibration input",
    )

    # 4. Fit Platt calibration on train split
    calib_dir = output_dir / "calibration_txr"
    run(
        [
            sys.executable,
            "fit_label_calibrators.py",
            "--csv",
            str(train_scores),
            "--out_dir",
            str(calib_dir),
        ],
        "Fitting Platt calibrators (train split)",
    )

    # 5. Apply calibration to train and test splits
    train_cal = output_dir / "train_txr_calibrated.csv"
    test_cal = output_dir / "test_txr_calibrated.csv"

    run(
        [
            sys.executable,
            "apply_label_calibrators.py",
            "--csv",
            str(train_scores),
            "--calib_dir",
            str(calib_dir),
            "--out_csv",
            str(train_cal),
        ],
        "Applying calibrators to train split",
    )

    run(
        [
            sys.executable,
            "apply_label_calibrators.py",
            "--csv",
            str(test_scores),
            "--calib_dir",
            str(calib_dir),
            "--out_csv",
            str(test_cal),
        ],
        "Applying calibrators to test split",
    )

    # 6. Threshold tuning on calibrated train probabilities
    thresholds_json = output_dir / "thresholds_txr.json"
    thresholds_metrics = output_dir / "thresholds_txr_metrics.csv"

    run(
        [
            sys.executable,
            "threshold_tuner.py",
            "--csv",
            str(train_cal),
            "--mode",
            "fbeta",
            "--beta",
            str(args.beta),
            "--out_json",
            str(thresholds_json),
            "--out_metrics",
            str(thresholds_metrics),
        ],
        "Tuning thresholds on calibrated train probabilities",
    )

    # 7. Evaluate on held-out test split
    metrics_path = output_dir / "test_txr_metrics.csv"
    run(
        [
            sys.executable,
            "evaluate_prob_predictions.py",
            "--predictions",
            str(test_cal),
            "--ground_truth",
            str(test_gt),
            "--thresholds",
            str(thresholds_json),
            "--score_prefix",
            "y_cal_",
            "--out_metrics",
            str(metrics_path),
        ],
        "Evaluating calibrated TXR probabilities on held-out test split",
    )

    print("\n✅ TorchXRayVision pipeline complete!")
    print(f"   Inference predictions: {txr_predictions}")
    print(f"   Calibration directory: {calib_dir}")
    print(f"   Train calibrated probabilities: {train_cal}")
    print(f"   Test calibrated probabilities:  {test_cal}")
    print(f"   Thresholds JSON: {thresholds_json}")
    print(f"   Test metrics CSV: {metrics_path}")
    print("\nSuggested next steps:")
    print("  - Blend TXR probabilities with CheXagent calibrated outputs using blend_probabilities.py")
    print("  - Rerun evaluation via evaluate_prob_predictions.py on the blended results")


if __name__ == "__main__":
    main()
