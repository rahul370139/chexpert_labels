#!/usr/bin/env python3
"""
Patient-wise data splitting to prevent leakage.

MIMIC-CXR images have patient IDs (subject_id) embedded in paths like:
  /path/to/files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.dcm

We need to split by patient (subject_id), not by image, so all images from one patient
stay in the same split.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json


def extract_patient_id(image_path):
    """Extract patient ID from MIMIC-CXR path."""
    path = Path(image_path)
    parts = path.parts
    
    # Look for pattern: p10/p10000032/s50414267/image.dcm
    for i, part in enumerate(parts):
        if part.startswith('p') and len(part) > 3 and i + 1 < len(parts):
            # This should be the patient directory (e.g., p10000032)
            patient_dir = parts[i + 1]
            if patient_dir.startswith('p') and len(patient_dir) > 3:
                return patient_dir
    
    # Fallback: try to find in path string
    path_str = str(image_path)
    import re
    match = re.search(r'/p\d+/(p\d{8})/s\d+/', path_str)
    if match:
        return match.group(1)
    
    # Last resort: use filename as proxy (not ideal but prevents crash)
    return path.stem


def patient_wise_split(manifest_csv, predictions_csv, train_ratio=0.7, random_seed=42):
    """
    Split data patient-wise into train/test.
    
    Args:
        manifest_csv: Ground truth CSV with image paths
        predictions_csv: Predictions CSV (to ensure we only split images we have predictions for)
        train_ratio: Fraction for training (default 0.7 for 70/30 split)
        random_seed: Random seed for reproducibility
    
    Returns:
        train_df, test_df: DataFrames with train/test splits
    """
    print(f"\n{'='*80}")
    print(f"PATIENT-WISE SPLITTING (avoiding data leakage)")
    print(f"{'='*80}\n")
    
    # Load data
    gt_df = pd.read_csv(manifest_csv)
    pred_df = pd.read_csv(predictions_csv)
    
    print(f"Ground truth: {len(gt_df)} images")
    print(f"Predictions: {len(pred_df)} images")
    
    # Match on filename to get only images with predictions
    gt_df['filename'] = gt_df['image'].apply(lambda x: Path(x).name)
    pred_df['filename'] = pred_df['image'].apply(lambda x: Path(x).name)
    
    # Merge to get matched dataset
    merged = pd.merge(gt_df, pred_df[['filename']], on='filename', how='inner')
    print(f"Matched: {len(merged)} images")
    
    # Extract patient IDs
    merged['patient_id'] = merged['image'].apply(extract_patient_id)
    
    # Get unique patients
    patients = merged['patient_id'].unique()
    n_patients = len(patients)
    print(f"Unique patients: {n_patients}")
    
    # Shuffle and split patients
    np.random.seed(random_seed)
    shuffled_patients = np.random.permutation(patients)
    
    n_train = int(n_patients * train_ratio)
    train_patients = set(shuffled_patients[:n_train])
    test_patients = set(shuffled_patients[n_train:])
    
    # Split images by patient
    train_df = merged[merged['patient_id'].isin(train_patients)].copy()
    test_df = merged[merged['patient_id'].isin(test_patients)].copy()
    
    print(f"\n📊 Split summary:")
    print(f"  Train: {len(train_patients)} patients, {len(train_df)} images ({len(train_df)/len(merged)*100:.1f}%)")
    print(f"  Test:  {len(test_patients)} patients, {len(test_df)} images ({len(test_df)/len(merged)*100:.1f}%)")
    
    # Check for leakage
    overlap = train_patients.intersection(test_patients)
    if overlap:
        print(f"  ⚠️  WARNING: {len(overlap)} patients appear in both splits!")
    else:
        print(f"  ✅ No patient leakage detected")
    
    # Show disease distribution in splits
    CHEXPERT13 = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
        "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
        "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
    ]
    
    print(f"\n📈 Disease distribution:")
    print(f"{'Disease':<30} {'Train Pos':>10} {'Test Pos':>10} {'Train %':>10} {'Test %':>10}")
    print("-" * 75)
    
    for disease in CHEXPERT13:
        if disease in train_df.columns and disease in test_df.columns:
            train_pos = train_df[disease].sum()
            test_pos = test_df[disease].sum()
            train_pct = train_pos / len(train_df) * 100
            test_pct = test_pos / len(test_df) * 100
            print(f"{disease:<30} {train_pos:>10} {test_pos:>10} {train_pct:>9.1f}% {test_pct:>9.1f}%")
    
    return train_df, test_df


def save_split(train_df, test_df, output_dir="data"):
    """Save train/test splits to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save image lists (for predictions)
    train_images = train_df['image'].tolist()
    test_images = test_df['image'].tolist()
    
    with open(output_dir / "train_images_70.txt", "w") as f:
        f.write("\n".join(train_images))
    
    with open(output_dir / "test_images_30.txt", "w") as f:
        f.write("\n".join(test_images))
    
    # Save ground truth CSVs
    train_df.to_csv(output_dir / "ground_truth_train_70.csv", index=False)
    test_df.to_csv(output_dir / "ground_truth_test_30.csv", index=False)
    
    # Save patient ID lists (for verification)
    train_patients = sorted(train_df['patient_id'].unique())
    test_patients = sorted(test_df['patient_id'].unique())
    
    with open(output_dir / "train_patients.json", "w") as f:
        json.dump(train_patients, f, indent=2)
    
    with open(output_dir / "test_patients.json", "w") as f:
        json.dump(test_patients, f, indent=2)
    
    print(f"\n💾 Saved split files to {output_dir}/")
    print(f"  - train_images_70.txt ({len(train_images)} images)")
    print(f"  - test_images_30.txt ({len(test_images)} images)")
    print(f"  - ground_truth_train_70.csv")
    print(f"  - ground_truth_test_30.csv")
    print(f"  - train_patients.json ({len(train_patients)} patients)")
    print(f"  - test_patients.json ({len(test_patients)} patients)")


def main():
    parser = argparse.ArgumentParser(description="Patient-wise train/test split")
    parser.add_argument(
        "--manifest",
        default="data/evaluation_manifest_phaseA_matched.csv",
        help="Ground truth manifest CSV"
    )
    parser.add_argument(
        "--predictions",
        default="hybrid_ensemble_1000.csv",
        help="Predictions CSV (to match images)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Train split ratio (default 0.7 for 70/30)"
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Output directory for split files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    train_df, test_df = patient_wise_split(
        args.manifest,
        args.predictions,
        args.train_ratio,
        args.seed
    )
    
    save_split(train_df, test_df, args.output_dir)
    
    print(f"\n✅ Patient-wise split complete!")
    print(f"\nNext steps:")
    print(f"  1. Fit calibration on train set:")
    print(f"     python fit_label_calibrators.py --csv <train_predictions> --out_dir calibration_70")
    print(f"  2. Evaluate on held-out test set:")
    print(f"     python evaluate_against_phaseA.py <test_predictions>")


if __name__ == "__main__":
    main()

