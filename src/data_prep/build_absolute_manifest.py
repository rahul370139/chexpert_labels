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


def normalise_label_value(value) -> int:
    """Map CheXpert {-1, 0, 1} to ints, treating missing as 0."""
    if value is None:
        return 0
    if isinstance(value, str):
        value = value.strip()
        if value in {"", "null", "None"}:
            return 0
        try:
            value = float(value)
        except ValueError:
            return 0
    if isinstance(value, (float, int)):
        if value > 0:
            return 1
        if value == 0:
            return 0
        return -1
    return 0


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
    args = parser.parse_args()

    manifest_records = load_manifest(args.manifest)
    if not manifest_records:
        raise SystemExit(f"No records found in {args.manifest}")

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
        row: Dict[str, object] = {
            "image": str(image_path),
            "filename": Path(rel_path).name,
            "image_id": entry.get("image_id"),
            "study_id": entry.get("study_id"),
            "subject_id": entry.get("subject_id"),
            "view": entry.get("view"),
            "impression": entry.get("impression"),
        }
        for label in CHEXPERT13:
            value = normalise_label_value(chex_labels.get(label))
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

