#!/usr/bin/env python3
"""
Create Phase-A manifest for 5k images matching CheXagent CSV.

Loads phaseA_manifest.jsonl, matches image filenames from CheXagent CSV,
converts CheXpert labels -1/0/1 → 0/1, and saves manifest.

Also supports -1/0/1 evaluation (three-class) mode for comparison.
"""

import argparse
import json
import pandas as pd
import sys
from pathlib import Path
from collections import Counter

# Import shared labels
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.common.labels import CHEXPERT14, CHEXPERT13


def load_phaseA_jsonl(jsonl_path: Path) -> pd.DataFrame:
    """Load phaseA_manifest.jsonl into DataFrame."""
    print(f"📂 Loading phaseA manifest: {jsonl_path}")
    
    records = []
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed JSON at line {line_num}: {e}")
                continue
    
    df = pd.DataFrame(records)
    print(f"   Loaded {len(df)} records")
    return df


def normalize_label(value, keep_uncertain: bool = False) -> int:
    """
    Normalize CheXpert label values:
    - NaN/None/empty → -1 (uncertain/blank - to be masked later)
    - -1 → -1 (uncertain)
    - 0 → 0 (negative)
    - 1 → 1 (positive)
    
    CRITICAL: Blanks (NaN) are ALWAYS converted to -1 so they can be masked during evaluation.
    Converting blanks to 0 would treat missing information as negative, which is wrong.
    """
    if pd.isna(value):
        return -1  # ALWAYS convert blanks to -1 (will be masked during evaluation)
    if isinstance(value, (int, float)):
        if keep_uncertain and value == -1:
            return -1
        return 1 if value > 0 else 0
    if isinstance(value, str):
        if value.lower() in {"-1", "uncertain"}:
            return -1 if keep_uncertain else 0
        if value.lower() in {"0", "false", "no"}:
            return 0
        if value.lower() in {"1", "true", "yes"}:
            return 1
    return 0 if not keep_uncertain else -1


def create_manifest_5k(
    phaseA_jsonl: Path,
    chexagent_csv: Path,
    output_csv: Path,
    keep_uncertain: bool = False,
) -> None:
    """
    Create 5k manifest matching CheXagent CSV images.
    
    Args:
        phaseA_jsonl: Path to phaseA_manifest.jsonl
        chexagent_csv: Path to CheXagent CSV (with image paths)
        output_csv: Output manifest CSV
        keep_uncertain: If True, keep -1 values (three-class mode)
    """
    # ALWAYS keep -1 for uncertain and blanks (will be masked during evaluation)
    # This ensures blanks and uncertain are both treated as -1 and masked
    keep_uncertain = True
    print("=" * 80)
    print(f"CREATING PHASE-A MANIFEST FOR 5K IMAGES [THREE-CLASS (-1/0/1)]")
    print("=" * 80)
    print("⚠️  Note: Blanks (NaN) → -1, Uncertain (-1) → -1 (both will be masked during evaluation)")
    
    # Load CheXagent CSV to get image list
    print(f"\n📥 Loading CheXagent CSV: {chexagent_csv}")
    chex_df = pd.read_csv(chexagent_csv)
    
    # Extract filenames
    if "filename" in chex_df.columns:
        chex_filenames = set(chex_df["filename"].dropna())
    elif "image" in chex_df.columns:
        chex_df = chex_df.copy()
        chex_df["filename"] = chex_df["image"].apply(lambda x: Path(x).name if pd.notna(x) else "")
        chex_filenames = set(chex_df["filename"].dropna())
    else:
        print("❌ CheXagent CSV missing 'filename' or 'image' column")
        sys.exit(1)
    
    print(f"   Found {len(chex_filenames)} unique filenames in CheXagent CSV")
    
    # Load phaseA data
    phaseA_df = load_phaseA_jsonl(phaseA_jsonl)
    
    # Extract filenames from phaseA (image_path column)
    if "image_path" in phaseA_df.columns:
        phaseA_df["filename"] = phaseA_df["image_path"].apply(
            lambda x: Path(x).name if pd.notna(x) else ""
        )
    elif "filename" in phaseA_df.columns:
        pass  # Already has filename
    else:
        print("❌ PhaseA manifest missing 'image_path' or 'filename' column")
        print(f"   Available columns: {list(phaseA_df.columns)}")
        sys.exit(1)

    # Flatten nested CheXpert labels if present (e.g., a 'chexpert' dict per row)
    if "chexpert" in phaseA_df.columns:
        try:
            chex_cols = pd.json_normalize(phaseA_df["chexpert"]).add_prefix("")
            # Only add missing label columns
            for col in CHEXPERT13:
                if col in chex_cols.columns and col not in phaseA_df.columns:
                    phaseA_df[col] = chex_cols[col]
        except Exception as e:
            print(f"⚠️  Could not normalize nested 'chexpert' column: {e}. Proceeding without flattening; labels may be missing.")
    
    # Fix: Fetch missing labels (Cardiomegaly, Atelectasis) from mimic-cxr CSV
    missing_labels = [label for label in CHEXPERT13 if label not in phaseA_df.columns]
    if missing_labels:
        print(f"\n🔧 Fetching {len(missing_labels)} missing labels from mimic-cxr CSV: {missing_labels}")
        
        # Try to find mimic-cxr CSV
        mimic_csv_paths = [
            project_root / "files" / "mimic-cxr-2.0.0-chexpert.csv",
            project_root.parent / "radiology_report" / "files" / "mimic-cxr-2.0.0-chexpert.csv",
            Path("../radiology_report/files/mimic-cxr-2.0.0-chexpert.csv"),
            Path("files/mimic-cxr-2.0.0-chexpert.csv"),
        ]
        
        mimic_csv = None
        for path in mimic_csv_paths:
            if path.exists():
                mimic_csv = path
                break
        
        if mimic_csv and mimic_csv.exists():
            print(f"   Loading: {mimic_csv}")
            mimic_df = pd.read_csv(mimic_csv)
            
            # Ensure study_id is string in both DataFrames
            if "study_id" in phaseA_df.columns:
                phaseA_df["study_id"] = phaseA_df["study_id"].astype(str)
            mimic_df["study_id"] = mimic_df["study_id"].astype(str)
            
            # Merge missing labels
            for label in missing_labels:
                if label in mimic_df.columns:
                    # Merge on study_id
                    merge_df = mimic_df[["study_id", label]].copy()
                    phaseA_df = phaseA_df.merge(
                        merge_df,
                        on="study_id",
                        how="left",
                        suffixes=("", "_mimic")
                    )
                    # Fill missing values with -1 (uncertain) or 0 (negative)
                    if label not in phaseA_df.columns:
                        phaseA_df[label] = phaseA_df[f"{label}_mimic"]
                    else:
                        # If already exists but has NaN, fill from mimic
                        phaseA_df[label] = phaseA_df[label].fillna(phaseA_df[f"{label}_mimic"])
                    phaseA_df = phaseA_df.drop(columns=[f"{label}_mimic"], errors="ignore")
                    print(f"   ✅ Fetched {label} from mimic-cxr CSV")
                else:
                    print(f"   ⚠️  {label} not found in mimic-cxr CSV either")
        else:
            print(f"   ⚠️  Could not find mimic-cxr CSV. Missing labels will be filled with 0")
    
    # Match filenames
    matched = phaseA_df[phaseA_df["filename"].isin(chex_filenames)].copy()
    print(f"\n✅ Matched {len(matched)} images from phaseA manifest")
    
    if len(matched) == 0:
        print("❌ No matches found. Check filename format in both files")
        sys.exit(1)
    
    # Extract and normalize CheXpert labels
    mode_label = "(-1/0/1)" if keep_uncertain else "(0/1)"
    print(f"\n🔄 Normalizing CheXpert labels {mode_label}...")
    
    label_counts = {label: Counter() for label in CHEXPERT14}
    
    for label in CHEXPERT13:
        if label in matched.columns:
            # Normalize: blanks (NaN) → -1, uncertain (-1) → -1, negative (0) → 0, positive (1) → 1
            # All blanks and uncertain will be -1 and masked during evaluation
            matched[label] = matched[label].apply(lambda x: normalize_label(x, keep_uncertain=True))
            # Count
            for val in [-1, 0, 1]:
                count = (matched[label] == val).sum()
                if count > 0:
                    label_counts[label][val] = count
            # Summary
            pos = (matched[label] == 1).sum()
            neg = (matched[label] == 0).sum()
            unc = (matched[label] == -1).sum() if keep_uncertain else 0
            print(f"   {label}: {pos} positive, {neg} negative" + (f", {unc} uncertain" if keep_uncertain else ""))
        else:
            # Missing label - fill with 0 or -1
            matched[label] = -1 if keep_uncertain else 0
            print(f"   ⚠️  {label}: missing in phaseA, filled with {'-1' if keep_uncertain else '0'}")
    
    # Compute "No Finding" (1 if all others are 0 or -1, otherwise 0)
    if keep_uncertain:
        # In three-class mode, No Finding = 1 only if all others are 0 or -1 (no positives)
        matched["No Finding"] = (
            (matched[CHEXPERT13] == 1).sum(axis=1) == 0
        ).astype(int)
    else:
        matched["No Finding"] = (
            matched[CHEXPERT13].sum(axis=1) == 0
        ).astype(int)
    
    no_finding_count = matched["No Finding"].sum()
    label_counts["No Finding"][1] = no_finding_count
    label_counts["No Finding"][0] = len(matched) - no_finding_count
    print(f"   No Finding: {no_finding_count} positive, {len(matched) - no_finding_count} negative")
    
    # Select columns for output
    output_cols = ["filename"]
    if "image_path" in matched.columns:
        output_cols.append("image_path")
    output_cols.extend(CHEXPERT14)
    
    # Add image column if it exists in CheXagent CSV
    if "image" in chex_df.columns:
        matched = matched.merge(
            chex_df[["filename", "image"]],
            on="filename",
            how="left"
        )
        if "image" not in output_cols:
            output_cols.insert(1, "image")
    
    output_df = matched[output_cols].copy()
    
    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Manifest saved to: {output_csv}")
    print(f"   Total images: {len(output_df)}")
    print(f"   Labels: {len(CHEXPERT14)}")
    print(f"   Mode: {mode_str}")
    
    # Summary
    print(f"\n📊 Label distribution:")
    for label in CHEXPERT14:
        pos = label_counts[label][1]
        neg = label_counts[label][0]
        unc = label_counts[label].get(-1, 0)
        pct_pos = pos / len(output_df) * 100 if len(output_df) > 0 else 0
        if keep_uncertain and unc > 0:
            print(f"   {label:30s} {pos:5d} pos, {neg:5d} neg, {unc:5d} unc ({pct_pos:5.1f}% pos)")
        else:
            print(f"   {label:30s} {pos:5d} positive, {neg:5d} negative ({pct_pos:5.1f}% positive)")
    
    print("\n" + "=" * 80)
    print("MANIFEST CREATION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Create Phase-A manifest for 5k images")
    parser.add_argument(
        "--phaseA_jsonl",
        type=Path,
        default=Path("../radiology_report/src/data/processed/phaseA_manifest.jsonl"),
        help="Path to phaseA_manifest.jsonl"
    )
    parser.add_argument(
        "--chexagent_csv",
        type=Path,
        default=Path("results/hybrid_ensemble_5826.csv"),
        help="CheXagent CSV with image paths"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation_manifest_phaseA_5k.csv"),
        help="Output manifest CSV"
    )
    parser.add_argument(
        "--keep_uncertain",
        action="store_true",
        help="Keep -1 (uncertain) values for three-class evaluation"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    phaseA_jsonl = (project_root / args.phaseA_jsonl).resolve()
    chexagent_csv = (project_root / args.chexagent_csv).resolve()
    output_csv = (project_root / args.output).resolve()
    
    if not phaseA_jsonl.exists():
        print(f"❌ PhaseA JSONL not found: {phaseA_jsonl}")
        sys.exit(1)
    
    if not chexagent_csv.exists():
        print(f"❌ CheXagent CSV not found: {chexagent_csv}")
        sys.exit(1)
    
    create_manifest_5k(phaseA_jsonl, chexagent_csv, output_csv, args.keep_uncertain)


if __name__ == "__main__":
    main()
