#!/usr/bin/env python3
"""
Generate impressions from CheXagent DI outputs for manager-ready output.

Extracts narrative text from `di_outputs` JSON and combines with predictions.
"""

import argparse
import json
import pandas as pd
import sys
from pathlib import Path


def extract_di_narrative(di_outputs_str):
    """Extract narrative text from DI outputs JSON string."""
    if pd.isna(di_outputs_str):
        return ""
    
    try:
        di_dict = json.loads(di_outputs_str)
        # DI output structure may vary, try common fields
        if isinstance(di_dict, dict):
            # Look for narrative/response fields
            narrative = di_dict.get("response", "") or di_dict.get("narrative", "") or di_dict.get("text", "")
            if narrative:
                return str(narrative)
    except (json.JSONDecodeError, TypeError):
        pass
    
    return ""


def generate_impression_from_labels(pred_row, chexpert_labels):
    """Generate impression text from predicted labels."""
    positive_labels = [label for label in chexpert_labels if pred_row.get(label, 0) == 1]
    
    if not positive_labels:
        return "No acute cardiopulmonary abnormalities identified."
    
    if len(positive_labels) == 1:
        return f"{positive_labels[0]} identified."
    
    findings = ", ".join(positive_labels[:-1]) + f" and {positive_labels[-1]}"
    return f"Findings include: {findings}."


def generate_impressions(
    chexagent_csv: Path,
    predictions_csv: Path,
    output_csv: Path,
) -> None:
    """
    Generate impressions CSV combining DI narratives and predictions.
    
    Args:
        chexagent_csv: CheXagent CSV with di_outputs
        predictions_csv: Predictions CSV with final predictions
        output_csv: Output CSV with impressions
    """
    print("=" * 80)
    print("GENERATING IMPRESSIONS FROM CHEXAGENT DI OUTPUTS")
    print("=" * 80)
    
    # Load CSVs
    print(f"\n📂 Loading CheXagent CSV: {chexagent_csv}")
    chex_df = pd.read_csv(chexagent_csv)
    print(f"   Rows: {len(chex_df)}")
    
    print(f"\n📂 Loading predictions CSV: {predictions_csv}")
    pred_df = pd.read_csv(predictions_csv)
    print(f"   Rows: {len(pred_df)}")
    
    # Ensure filename column
    if "filename" not in chex_df.columns:
        if "image" in chex_df.columns:
            chex_df["filename"] = chex_df["image"].apply(lambda x: Path(x).name if pd.notna(x) else "")
    
    if "filename" not in pred_df.columns:
        if "image" in pred_df.columns:
            pred_df["filename"] = pred_df["image"].apply(lambda x: Path(x).name if pd.notna(x) else "")
        else:
            print("❌ Predictions CSV missing 'filename' or 'image' column")
            sys.exit(1)
    
    # Merge on filename
    merged = pred_df.merge(
        chex_df[["filename", "di_outputs", "image"]],
        on="filename",
        how="left"
    )
    
    print(f"\n✅ Merged {len(merged)} rows")
    
    # CheXpert labels (13 + No Finding = 14)
    CHEXPERT13 = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
        "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
        "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
    ]
    CHEXPERT14 = CHEXPERT13 + ["No Finding"]
    
    # Generate impressions
    print(f"\n🔄 Generating impressions...")
    impressions = []
    
    for idx, row in merged.iterrows():
        # Try to extract DI narrative first
        di_narrative = extract_di_narrative(row.get("di_outputs", ""))
        
        if di_narrative and len(di_narrative) > 20:
            # Use DI narrative if available and substantial
            impression = di_narrative.strip()
        else:
            # Fallback: generate from predicted labels
            impression = generate_impression_from_labels(row, CHEXPERT13)
        
        impressions.append(impression)
    
    merged["impression"] = impressions
    
    # Select output columns
    output_cols = ["filename"]
    if "image" in merged.columns:
        output_cols.append("image")
    output_cols.append("impression")
    output_cols.extend(CHEXPERT14)  # Ground truth labels if present
    # Add predicted columns (might be named with _pred suffix or just label name)
    for label in CHEXPERT13:
        pred_col = label if label in merged.columns else f"{label}_pred"
        if pred_col in merged.columns and pred_col not in output_cols:
            output_cols.append(pred_col)
    if "No Finding" in merged.columns and "No Finding" not in output_cols:
        output_cols.append("No Finding")
    
    output_df = merged[output_cols].copy()
    
    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Impressions saved to: {output_csv}")
    print(f"   Total rows: {len(output_df)}")
    
    # Sample impressions
    print(f"\n📝 Sample impressions:")
    for i, row in output_df.head(3).iterrows():
        print(f"   {i+1}. {row['impression'][:100]}...")
    
    print("\n" + "=" * 80)
    print("IMPRESSION GENERATION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate impressions from CheXagent DI outputs")
    parser.add_argument(
        "--chexagent_csv",
        type=Path,
        required=True,
        help="CheXagent CSV with di_outputs"
    )
    parser.add_argument(
        "--predictions_csv",
        type=Path,
        required=True,
        help="Predictions CSV with final predictions"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs_5k/final/test_with_impressions.csv"),
        help="Output CSV with impressions"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    chexagent_csv = (project_root / args.chexagent_csv).resolve()
    predictions_csv = (project_root / args.predictions_csv).resolve()
    output_csv = (project_root / args.output).resolve()
    
    if not chexagent_csv.exists():
        print(f"❌ CheXagent CSV not found: {chexagent_csv}")
        sys.exit(1)
    
    if not predictions_csv.exists():
        print(f"❌ Predictions CSV not found: {predictions_csv}")
        sys.exit(1)
    
    generate_impressions(chexagent_csv, predictions_csv, output_csv)


if __name__ == "__main__":
    main()

