#!/usr/bin/env python3
"""
Convert the Phase-A manifest JSONL into a CSV with absolute image paths and CheXpert labels.

Example:
    python src/data_prep/build_absolute_manifest.py \
        --manifest ../radiology_report/src/data/processed/phaseA_manifest.jsonl \
        --image_root ../radiology_report \
        --out_csv data/evaluation_manifest_phaseA_full_abs.csv \
        --out_image_list data/image_list_phaseA_full_absolute.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

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

CHEXPERT14 = CHEXPERT13 + ["No Finding"]


def normalise_label_value(value) -> int:
    """
    Map CheXpert values to ints:
    - NaN/None/empty → -1 (uncertain/blank - to be masked later)
    - -1 → -1 (uncertain)
    - 0 → 0 (negative)
    - 1 → 1 (positive)
    
    CRITICAL: Blanks (NaN) are converted to -1 so they can be masked during evaluation.
    Converting blanks to 0 would treat missing information as negative, which is wrong.
    """
    import math
    
    if value is None:
        return -1  # Blank → -1 (uncertain, will be masked)
    if isinstance(value, str):
        value = value.strip()
        if value in {"", "null", "None", "nan", "NaN"}:
            return -1  # Blank → -1 (uncertain, will be masked)
        try:
            value = float(value)
        except ValueError:
            return -1  # Invalid → -1 (uncertain, will be masked)
    if isinstance(value, (float, int)):
        if math.isnan(value):
            return -1  # NaN → -1 (blank/uncertain, will be masked)
        if value > 0:
            return 1  # Positive
        if value == 0:
            return 0  # Negative
        if value < 0:
            return -1  # Uncertain (already -1)
    return -1  # Default: treat as uncertain/blank (will be masked)


def load_manifest(manifest_path: Path) -> List[Dict]:
    records: List[Dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build absolute-path manifest from Phase-A JSONL.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to phaseA_manifest.jsonl")
    parser.add_argument("--image_root", type=Path, default=Path("../radiology_report"),
                        help="Root directory that contains the 'files/' folder")
    parser.add_argument("--out_csv", type=Path, default=Path("data/evaluation_manifest_phaseA_full_abs.csv"))
    parser.add_argument("--out_image_list", type=Path, default=Path("data/image_list_phaseA_full_absolute.txt"))
    parser.add_argument("--mimic_csv", type=Path, default=None,
                        help="Path to mimic-cxr-2.0.0-chexpert.csv (for missing labels)")
    args = parser.parse_args()

    manifest_records = load_manifest(args.manifest)
    if not manifest_records:
        raise SystemExit(f"No records found in {args.manifest}")

    # Load mimic-cxr CSV if provided (for missing labels)
    mimic_df = None
    missing_labels_in_phaseA = []
    
    # Check which labels are missing in phaseA
    if manifest_records:
        sample_chexpert = manifest_records[0].get("chexpert", {})
        if isinstance(sample_chexpert, dict):
            missing_labels_in_phaseA = [label for label in CHEXPERT13 if label not in sample_chexpert]
    
    if missing_labels_in_phaseA:
        print(f"⚠️  Found {len(missing_labels_in_phaseA)} missing labels in phaseA: {missing_labels_in_phaseA}")
        
        # Try to find mimic-cxr CSV
        mimic_csv_paths = [
            args.mimic_csv if args.mimic_csv else None,
            Path("files/mimic-cxr-2.0.0-chexpert.csv"),
            Path("../radiology_report/files/mimic-cxr-2.0.0-chexpert.csv"),
            args.image_root.parent / "files" / "mimic-cxr-2.0.0-chexpert.csv",
        ]
        
        for path in mimic_csv_paths:
            if path and path.exists():
                print(f"📂 Loading mimic-cxr CSV: {path}")
                mimic_df = pd.read_csv(path)
                mimic_df["study_id"] = mimic_df["study_id"].astype(str)
                # Create lookup dict
                mimic_df = mimic_df.set_index("study_id")[missing_labels_in_phaseA].to_dict("index")
                print(f"   ✅ Loaded {len(mimic_df)} study records with missing labels")
                break
        
        if not mimic_df:
            print(f"   ⚠️  Could not find mimic-cxr CSV. Missing labels will be set to 0")

    rows: List[Dict] = []
    missing = 0
    for entry in manifest_records:
        rel_path = entry.get("image_path")
        if not rel_path:
            continue
        image_path = (args.image_root / rel_path).resolve()
        if not image_path.exists():
            missing += 1
            continue

        chex_labels = entry.get("chexpert", {})
        study_id = str(entry.get("study_id", ""))
        
        row: Dict[str, object] = {
            "image": str(image_path),
            "filename": Path(rel_path).name,
            "image_id": entry.get("image_id"),
            "study_id": study_id,
            "subject_id": entry.get("subject_id"),
            "view": entry.get("view"),
            "impression": entry.get("impression"),
        }
        
        # Extract labels from chexpert dict
        for label in CHEXPERT13:
            # First try phaseA manifest
            if isinstance(chex_labels, dict) and label in chex_labels:
            value = normalise_label_value(chex_labels.get(label))
            # Then try mimic-cxr CSV if available
            elif mimic_df and study_id in mimic_df:
                value = normalise_label_value(mimic_df[study_id].get(label))
            else:
                # Default to 0 (negative)
                value = 0
            row[label] = value
        
        rows.append(row)

    if not rows:
        raise SystemExit("No images were found on disk. Check --image_root.")

    df = pd.DataFrame(rows)
    df.sort_values("filename", inplace=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    args.out_image_list.parent.mkdir(parents=True, exist_ok=True)
    with args.out_image_list.open("w", encoding="utf-8") as handle:
        for path in df["image"].unique():
            handle.write(f"{path}\n")

    print(f"✅ Saved manifest to {args.out_csv} ({len(df)} rows)")
    print(f"✅ Saved image list to {args.out_image_list}")
    if missing:
        print(f"⚠️ Skipped {missing} entries because the image file was not found.")


if __name__ == "__main__":
    main()

