#!/usr/bin/env python3
"""
Patient-wise stratified splitting with positive-count balancing.

Goal: ensure each CheXpert label retains approximately the desired number of positives
in train vs validation while keeping patients disjoint.

The algorithm:
 1. Aggregate labels per patient (positive if any study for that patient has label==1).
 2. Sort patients by number of positive labels (descending).
 3. Greedily assign each patient to train/test to minimise the squared error between
    observed positive counts and the target counts (train_ratio * total positives).
 4. Fall back to patient-count heuristics if a split reaches its target size.

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


def aggregate_patient_labels(df: pd.DataFrame, labels: List[str]) -> Dict[str, np.ndarray]:
    patient_groups = df.groupby("patient_id")
    patient_vectors: Dict[str, np.ndarray] = {}
    for patient, group in patient_groups:
        label_matrix = group[labels].values
        positives = (label_matrix == 1).any(axis=0).astype(int)
        patient_vectors[patient] = positives
    return patient_vectors


def stratified_patient_split(
    patient_vectors: Dict[str, np.ndarray],
    train_ratio: float,
    random_seed: int,
) -> Tuple[set[str], set[str]]:
    rng = np.random.default_rng(random_seed)
    patients = list(patient_vectors.keys())
    total_patients = len(patients)
    train_target = int(round(total_patients * train_ratio))
    test_target = total_patients - train_target

    label_matrix = np.stack([patient_vectors[p] for p in patients])
    total_pos = label_matrix.sum(axis=0).astype(float)
    desired_train = total_pos * train_ratio
    desired_test = total_pos - desired_train

    # Order patients by number of positives (desc), tie-break randomly
    positives_counts = label_matrix.sum(axis=1)
    order = np.argsort(-positives_counts + rng.random(len(patients)) * 1e-6)

    train_patients: set[str] = set()
    test_patients: set[str] = set()
    train_counts = np.zeros(len(total_pos), dtype=float)
    test_counts = np.zeros(len(total_pos), dtype=float)

    for idx in order:
        patient = patients[idx]
        vector = patient_vectors[patient]

        # If one split already full, force assignment to the other
        if len(train_patients) >= train_target:
            test_patients.add(patient)
            test_counts += vector
            continue
        if len(test_patients) >= test_target:
            train_patients.add(patient)
            train_counts += vector
            continue

        # Compute squared error w.r.t. desired positive counts
        err_train = np.sum((train_counts + vector - desired_train) ** 2)
        err_test = np.sum((test_counts + vector - desired_test) ** 2)

        # Regularise by patient counts to avoid drift
        err_train += (len(train_patients) + 1 - train_target) ** 2 * 1e-3
        err_test += (len(test_patients) + 1 - test_target) ** 2 * 1e-3

        if err_train <= err_test:
            train_patients.add(patient)
            train_counts += vector
        else:
            test_patients.add(patient)
            test_counts += vector

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

    patient_vectors = aggregate_patient_labels(df, args.labels)
    print(f"Unique patients: {len(patient_vectors)}")

    train_patients, test_patients = stratified_patient_split(
        patient_vectors,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
    )

    train_df, test_df = build_splits(df, train_patients, test_patients)

    print(f"\n📊 Split summary:")
    print(f"  Train: {len(train_patients)} patients, {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test : {len(test_patients)} patients, {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")

    report_distribution(train_df, test_df, args.labels)
    save_split(train_df, test_df, args.output_dir)


if __name__ == "__main__":
    main()
