#!/usr/bin/env python3
"""
Patient-wise stratified splitting with positive-count balancing.

Goal: ensure each CheXpert label retains approximately the desired number of positives
in train vs validation while keeping patients disjoint AND achieving exact image ratio (80:20).

The algorithm:
 1. Aggregate labels per patient (positive if any study for that patient has label==1).
 2. Target IMAGE ratio (80:20), not patient ratio.
 3. Greedily assign patients to minimize error in both image count AND positive label distribution.
 4. Fine-tune to hit exact image ratio within 2% tolerance.
 5. Fix zero-positive labels by swapping patients.

Usage:
    python src/data_prep/patient_wise_split.py \
        --manifest data/evaluation_manifest_phaseA_full_abs.csv \
        --train_ratio 0.8 \
        --output_dir data/splits_80_20
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

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


def extract_patient_id(image_path: str) -> str:
    path = Path(image_path)
    parts = path.parts
    for idx, part in enumerate(parts):
        if part.startswith("p") and len(part) > 3 and idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if candidate.startswith("p") and len(candidate) > 3:
                return candidate
    # fallback: look for "/p1234/p12345678/"
    import re

    match = re.search(r"/p\d+/(p\d{6,})/", str(path))
    if match:
        return match.group(1)
    return path.stem


def load_manifest(manifest_csv: Path, labels: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    if "image" not in df.columns:
        raise ValueError(f"{manifest_csv} must contain an 'image' column with file paths.")
    missing = [lab for lab in labels if lab not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing label columns: {missing}")
    df["filename"] = df["image"].apply(lambda p: Path(str(p)).name)
    df["patient_id"] = df["image"].apply(extract_patient_id)
    return df


def aggregate_patient_labels(df: pd.DataFrame, labels: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Aggregate patient labels and return both vectors and image counts."""
    patient_groups = df.groupby("patient_id")
    patient_vectors: Dict[str, np.ndarray] = {}
    patient_image_counts: Dict[str, int] = {}
    for patient, group in patient_groups:
        label_matrix = group[labels].values
        positives = (label_matrix == 1).any(axis=0).astype(int)
        patient_vectors[patient] = positives
        patient_image_counts[patient] = len(group)
    return patient_vectors, patient_image_counts


def stratified_patient_split(
    patient_vectors: Dict[str, np.ndarray],
    patient_image_counts: Dict[str, int],
    train_ratio: float,
    random_seed: int,
    min_positives_per_label: int = 15,
) -> Tuple[set[str], set[str]]:
    """
    Smart stratified patient-wise split that:
    1. Targets EXACT 80:20 IMAGE split (not patient split)
    2. Prioritizes patients with MORE images for train (to achieve image balance)
    3. Ensures each label has minimum positives in train AND test
    4. Maintains patient-wise separation (zero leakage)
    """
    rng = np.random.default_rng(random_seed)
    patients = list(patient_vectors.keys())
    
    # Calculate total images and target image counts
    total_images = sum(patient_image_counts.get(p, 1) for p in patients)
    train_target_images = int(round(total_images * train_ratio))
    test_target_images = total_images - train_target_images
    
    print(f"   Target: {train_target_images:,} train images, {test_target_images:,} test images")

    label_matrix = np.stack([patient_vectors[p] for p in patients])
    total_pos = label_matrix.sum(axis=0).astype(float)
    desired_train = total_pos * train_ratio
    desired_test = total_pos - desired_train

    # Order by: (1) image count DESC, (2) positives count DESC, (3) random tie-break
    # This prioritizes high-image-count patients for train set
    image_counts = np.array([patient_image_counts.get(p, 1) for p in patients])
    positives_counts = label_matrix.sum(axis=1)
    
    # Combined score: image_count * 1000 + positives (prioritize high-image patients)
    combined_score = image_counts * 1000 + positives_counts
    order = np.argsort(-combined_score + rng.random(len(patients)) * 1e-6)

    train_patients: set[str] = set()
    test_patients: set[str] = set()
    train_counts = np.zeros(len(total_pos), dtype=float)
    test_counts = np.zeros(len(total_pos), dtype=float)
    train_image_count = 0
    test_image_count = 0

    # Phase 1: Initial assignment (target IMAGE ratio)
    for idx in order:
        patient = patients[idx]
        vector = patient_vectors[patient]
        image_count = patient_image_counts.get(patient, 1)

        # Check if image targets are met (with tolerance)
        train_image_full = train_image_count >= train_target_images
        test_image_full = test_image_count >= test_target_images
        
        # If both full (shouldn't happen, but safety check)
        if train_image_full and test_image_full:
            # Assign to smaller split
            if test_image_count < train_image_count:
                test_patients.add(patient)
                test_counts += vector
                test_image_count += image_count
            else:
                train_patients.add(patient)
                train_counts += vector
                train_image_count += image_count
            continue
        
        # If train full, force to test
        if train_image_full:
            test_patients.add(patient)
            test_counts += vector
            test_image_count += image_count
            continue
        
        # If test full, force to train
        if test_image_full:
            train_patients.add(patient)
            train_counts += vector
            train_image_count += image_count
            continue

        # Compute error considering:
        # 1. Image count balance (weighted 70%)
        # 2. Positive label balance (weighted 30%)
        train_image_after = train_image_count + image_count
        test_image_after = test_image_count + image_count
        
        # Normalized image count error
        image_err_train = abs(train_image_after - train_target_images) / max(1, train_target_images)
        image_err_test = abs(test_image_after - test_target_images) / max(1, test_target_images)
        
        # Normalized positive label balance error
        pos_err_train = np.sum((train_counts + vector - desired_train) ** 2) / max(1, np.sum(desired_train) ** 2 + 1e-6)
        pos_err_test = np.sum((test_counts + vector - desired_test) ** 2) / max(1, np.sum(desired_test) ** 2 + 1e-6)

        # Combined error (image balance weighted 70%, label balance 30%)
        err_train = 0.7 * image_err_train + 0.3 * pos_err_train
        err_test = 0.7 * image_err_test + 0.3 * pos_err_test

        if err_train <= err_test:
            train_patients.add(patient)
            train_counts += vector
            train_image_count += image_count
        else:
            test_patients.add(patient)
            test_counts += vector
            test_image_count += image_count

    # Phase 2: Fine-tune to hit exact image ratio
    image_ratio_tolerance = 0.02  # 2% tolerance
    current_train_ratio = train_image_count / total_images if total_images > 0 else 0
    
    if abs(current_train_ratio - train_ratio) > image_ratio_tolerance:
        print(f"\n🔧 Fine-tuning to hit exact {train_ratio*100:.0f}:{100-train_ratio*100:.0f} image ratio...")
        print(f"   Current: {train_image_count:,}/{total_images:,} = {current_train_ratio*100:.1f}% train")
        
        # If train is too small, move high-image patients from test to train
        if current_train_ratio < train_ratio - image_ratio_tolerance:
            deficit = train_target_images - train_image_count
            candidates = sorted(
                [(p, patient_image_counts.get(p, 1)) for p in test_patients],
                key=lambda x: x[1],
                reverse=True  # Start with largest
            )
            for patient, img_count in candidates:
                if img_count <= deficit + 200:  # Allow small overshoot
                    train_patients.add(patient)
                    test_patients.remove(patient)
                    train_counts += patient_vectors[patient]
                    test_counts -= patient_vectors[patient]
                    train_image_count += img_count
                    test_image_count -= img_count
                    deficit -= img_count
                    if deficit <= 0:
                        break
        
        # If train is too large, move high-image patients from train to test
        elif current_train_ratio > train_ratio + image_ratio_tolerance:
            surplus = train_image_count - train_target_images
            candidates = sorted(
                [(p, patient_image_counts.get(p, 1)) for p in train_patients],
                key=lambda x: x[1],
                reverse=True  # Start with largest
            )
            for patient, img_count in candidates:
                if img_count <= surplus + 200:  # Allow small undershoot
                    test_patients.add(patient)
                    train_patients.remove(patient)
                    test_counts += patient_vectors[patient]
                    train_counts -= patient_vectors[patient]
                    test_image_count += img_count
                    train_image_count -= img_count
                    surplus -= img_count
                    if surplus <= 0:
                        break

    # Phase 3: Fix zero-positive labels (swap patients if needed)
    zero_labels_in_train = []
    for label_idx in range(len(total_pos)):
        if train_counts[label_idx] < min_positives_per_label:
            zero_labels_in_train.append(label_idx)
    
    if zero_labels_in_train:
        print(f"\n⚠️  Fixing {len(zero_labels_in_train)} labels with <{min_positives_per_label} positives in train...")
        
        # Find patients in test with these positives
        for label_idx in zero_labels_in_train:
            candidates = []
            for patient in test_patients:
                if patient_vectors[patient][label_idx] > 0:
                    candidates.append((patient, patient_image_counts.get(patient, 1)))
            
            if candidates:
                # Swap: move candidate to train, move a train patient to test
                # Prefer swapping patients with similar image counts to maintain balance
                candidates.sort(key=lambda x: x[1])  # Sort by image count
                patient_to_move, move_img_count = candidates[0]
                
                # Find a train patient to swap (prefer one with low positives for this label and similar image count)
                swap_candidates = [
                    (p, patient_image_counts.get(p, 1))
                    for p in train_patients
                    if patient_vectors[p][label_idx] == 0
                ]
                if swap_candidates:
                    # Prefer swap candidates with similar image count
                    swap_candidates.sort(key=lambda x: abs(x[1] - move_img_count))
                    patient_to_swap, _ = swap_candidates[0]
                    
                    # Perform swap
                    train_patients.remove(patient_to_swap)
                    train_patients.add(patient_to_move)
                    test_patients.remove(patient_to_move)
                    test_patients.add(patient_to_swap)
                    
                    # Update counts
                    train_counts += patient_vectors[patient_to_move] - patient_vectors[patient_to_swap]
                    test_counts += patient_vectors[patient_to_swap] - patient_vectors[patient_to_move]
                    train_image_count += move_img_count - patient_image_counts.get(patient_to_swap, 1)
                    test_image_count += patient_image_counts.get(patient_to_swap, 1) - move_img_count
                    
                    print(f"   ✅ Swapped patient for label {label_idx}")

    return train_patients, test_patients


def build_splits(df: pd.DataFrame, train_patients: set[str], test_patients: set[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["patient_id"].isin(train_patients)].copy()
    test_df = df[df["patient_id"].isin(test_patients)].copy()
    return train_df, test_df


def report_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame, labels: List[str]) -> None:
    print("\n📈 Label distribution per split")
    print(f"{'Label':30s} {'Train Pos':>10} {'Test Pos':>10} {'Train %':>10} {'Test %':>10}")
    print("-" * 76)
    for label in labels:
        train_pos = int((train_df[label] == 1).sum())
        test_pos = int((test_df[label] == 1).sum())
        train_pct = train_pos / max(1, len(train_df)) * 100
        test_pct = test_pos / max(1, len(test_df)) * 100
        print(f"{label:30s} {train_pos:10d} {test_pos:10d} {train_pct:9.2f}% {test_pct:9.2f}%")


def save_split(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_images.txt").write_text("\n".join(train_df["image"]))
    (output_dir / "test_images.txt").write_text("\n".join(test_df["image"]))
    train_df.to_csv(output_dir / "ground_truth_train.csv", index=False)
    test_df.to_csv(output_dir / "ground_truth_test.csv", index=False)
    with open(output_dir / "train_patients.json", "w") as handle:
        json.dump(sorted(train_df["patient_id"].unique()), handle, indent=2)
    with open(output_dir / "test_patients.json", "w") as handle:
        json.dump(sorted(test_df["patient_id"].unique()), handle, indent=2)
    print(f"\n💾 Saved split artefacts to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-wise stratified split for CheXpert labels")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV manifest with absolute image paths and CheXpert labels")
    parser.add_argument("--predictions", type=Path, default=None,
                        help="Optional predictions CSV to filter available images (must contain 'filename')")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--output_dir", type=Path, default=Path("data/split_80_20"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", nargs="+", default=CHEXPERT13)
    args = parser.parse_args()

    df = load_manifest(args.manifest, args.labels)

    if args.predictions:
        pred_df = pd.read_csv(args.predictions)
        if "filename" not in pred_df.columns and "image" in pred_df.columns:
            pred_df["filename"] = pred_df["image"].apply(lambda p: Path(str(p)).name)
        if "filename" not in pred_df.columns:
            raise ValueError(f"{args.predictions} must contain 'filename' or 'image' column.")
        df = df[df["filename"].isin(pred_df["filename"])]
        print(f"Filtered manifest to {len(df)} images that have predictions.")

    print(f"Total images available: {len(df)}")

    patient_vectors, patient_image_counts = aggregate_patient_labels(df, args.labels)
    print(f"Unique patients: {len(patient_vectors)}")
    
    # Calculate total images and target image counts
    total_images = len(df)
    train_target_images = int(round(total_images * args.train_ratio))
    test_target_images = total_images - train_target_images
    print(f"Target image split: Train={train_target_images:,} ({args.train_ratio*100:.0f}%), Test={test_target_images:,} ({100-args.train_ratio*100:.0f}%)")

    train_patients, test_patients = stratified_patient_split(
        patient_vectors,
        patient_image_counts,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        min_positives_per_label=15,
    )

    train_df, test_df = build_splits(df, train_patients, test_patients)

    actual_train_ratio = len(train_df) / total_images
    actual_test_ratio = len(test_df) / total_images

    print(f"\n📊 Split summary:")
    print(f"  Train: {len(train_patients)} patients, {len(train_df):,} images ({actual_train_ratio*100:.1f}%)")
    print(f"  Test : {len(test_patients)} patients, {len(test_df):,} images ({actual_test_ratio*100:.1f}%)")
    
    # Verify image ratio
    ratio_diff = abs(actual_train_ratio - args.train_ratio)
    if ratio_diff <= 0.02:
        print(f"  ✅ Image ratio within 2% tolerance (diff: {ratio_diff*100:.2f}%)")
    else:
        print(f"  ⚠️  Image ratio differs by {ratio_diff*100:.2f}% from target")

    report_distribution(train_df, test_df, args.labels)
    save_split(train_df, test_df, args.output_dir)


if __name__ == "__main__":
    main()
