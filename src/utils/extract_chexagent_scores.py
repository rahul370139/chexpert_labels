#!/usr/bin/env python3
"""Extract probability scores from hybrid_ensemble CSV binary_outputs column."""

import argparse
import json
import pandas as pd
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="Extract CheXagent scores from binary_outputs")
    parser.add_argument("--input_csv", required=True, help="hybrid_ensemble CSV with binary_outputs")
    parser.add_argument("--output_csv", required=True, help="Output CSV with y_pred_{label} columns")
    parser.add_argument("--score_prefix", default="y_pred_", help="Prefix for score columns")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    
    if "binary_outputs" not in df.columns:
        raise ValueError("Input CSV must have 'binary_outputs' column")
    
    print(f"📊 Loading {len(df)} predictions from {args.input_csv}")
    
    # Parse binary_outputs JSON
    binary_dicts = df["binary_outputs"].fillna("{}").apply(json.loads)
    
    # Extract scores for each disease
    output_df = df[["image"]].copy() if "image" in df.columns else pd.DataFrame()
    
    # Add filename column for matching
    if "filename" not in output_df.columns:
        if "image" in output_df.columns:
            output_df["filename"] = output_df["image"].apply(lambda x: Path(x).name if pd.notna(x) else "")
        elif "Image" in df.columns:
            output_df["filename"] = df["Image"].apply(lambda x: Path(x).name if pd.notna(x) else "")
    
    for disease in CHEXPERT13:
        scores = binary_dicts.apply(
            lambda record: record.get(disease, {}).get("score", 0.0)
        )
        output_df[f"{args.score_prefix}{disease}"] = scores.astype(float)
        print(f"  ✅ Extracted {disease} scores (mean: {scores.mean():.3f})")
    
    output_df.to_csv(args.output_csv, index=False)
    print(f"✅ Saved {len(output_df)} rows with scores to {args.output_csv}")


if __name__ == "__main__":
    main()

