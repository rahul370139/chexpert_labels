#!/usr/bin/env python3
"""
Create ground truth manifest from raw phaseA_manifest.jsonl.

Converts CheXpert labels from -1/0/1 to 0/1 format:
- -1 (uncertain) → 0 (negative)
- 0 (negative) → 0 (negative) 
- 1 (positive) → 1 (positive)

Creates balanced sample for evaluation.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

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
    "No Finding"
]

def load_phaseA_data(jsonl_path: Path):
    """Load and parse phaseA_manifest.jsonl."""
    print(f"📂 Loading phaseA data from {jsonl_path}")
    
    data = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed JSON at line {line_num}: {e}")
                continue
    
    print(f"✅ Loaded {len(data)} records")
    return data

def convert_labels(chexpert_dict):
    """Convert CheXpert labels from -1/0/1 to 0/1 format."""
    converted = {}
    for disease in CHEXPERT14:
        if disease in chexpert_dict:
            # -1 (uncertain) → 0, 0 (negative) → 0, 1 (positive) → 1
            converted[disease] = 1 if chexpert_dict[disease] == 1 else 0
        else:
            converted[disease] = 0
    return converted

def create_balanced_sample(data, n_samples=1000):
    """Create balanced sample ensuring diversity across diseases."""
    print(f"🎯 Creating balanced sample of {n_samples} images")
    
    # Convert all data first
    converted_data = []
    for record in data:
        converted_labels = convert_labels(record.get('chexpert', {}))
        converted_data.append({
            'image': record['image_path'],
            'image_id': record['image_id'],
            'study_id': record['study_id'],
            'subject_id': record['subject_id'],
            'view': record.get('view', 'Unknown'),
            'impression': record.get('impression', ''),
            **converted_labels
        })
    
    df = pd.DataFrame(converted_data)
    print(f"✅ Converted {len(df)} records to 0/1 format")
    
    # Analyze disease distribution
    print("\n📊 Disease distribution in full dataset:")
    for disease in CHEXPERT14:
        if disease in df.columns:
            pos_count = df[disease].sum()
            pos_pct = (pos_count / len(df)) * 100
            print(f"  {disease}: {pos_count:,} ({pos_pct:.1f}%)")
    
    # Create balanced sample
    if n_samples >= len(df):
        print(f"⚠️  Requested {n_samples} samples but only {len(df)} available. Using all data.")
        return df
    
    # Sample strategy: ensure we get some positives for each disease
    sampled_indices = set()
    
    # First, sample positives for each disease (up to 50 per disease)
    for disease in CHEXPERT14:
        if disease in df.columns:
            positives = df[df[disease] == 1].index.tolist()
            if positives:
                # Sample up to 50 positives per disease
                n_pos = min(50, len(positives), n_samples // len(CHEXPERT14))
                if n_pos > 0:
                    selected = np.random.choice(positives, size=n_pos, replace=False)
                    sampled_indices.update(selected)
    
    # Fill remaining with random samples
    remaining_needed = n_samples - len(sampled_indices)
    if remaining_needed > 0:
        remaining_indices = list(set(df.index) - sampled_indices)
        if remaining_indices:
            additional = np.random.choice(
                remaining_indices, 
                size=min(remaining_needed, len(remaining_indices)), 
                replace=False
            )
            sampled_indices.update(additional)
    
    # Create final sample
    sampled_df = df.loc[list(sampled_indices)].copy()
    sampled_df = sampled_df.reset_index(drop=True)
    
    print(f"\n📊 Disease distribution in sampled dataset ({len(sampled_df)} images):")
    for disease in CHEXPERT14:
        if disease in sampled_df.columns:
            pos_count = sampled_df[disease].sum()
            pos_pct = (pos_count / len(sampled_df)) * 100
            print(f"  {disease}: {pos_count} ({pos_pct:.1f}%)")
    
    return sampled_df

def extract_for_specific_images(data, predictions_csv):
    """Extract ground truth for specific images we have predictions for."""
    print(f"📂 Loading predictions from {predictions_csv} to get exact image list")
    
    pred_df = pd.read_csv(predictions_csv)
    print(f"✅ Found predictions for {len(pred_df)} images")
    
    # Get filenames from predictions
    pred_filenames = set(pred_df['image'].apply(lambda x: Path(x).name).tolist())
    print(f"   Unique filenames: {len(pred_filenames)}")
    
    # Match against phaseA data
    matched_data = []
    for record in data:
        img_filename = Path(record['image_path']).name
        if img_filename in pred_filenames:
            matched_data.append(record)
    
    print(f"✅ Matched {len(matched_data)} images from phaseA manifest")
    
    # Convert to ground truth format
    converted_data = []
    for record in matched_data:
        converted_labels = convert_labels(record.get('chexpert', {}))
        converted_data.append({
            'image': record['image_path'],
            'image_id': record['image_id'],
            'study_id': record['study_id'],
            'subject_id': record['subject_id'],
            'view': record.get('view', 'Unknown'),
            'impression': record.get('impression', ''),
            **converted_labels
        })
    
    df = pd.DataFrame(converted_data)
    
    # Show disease distribution
    print(f"\n📊 Disease distribution in matched ground truth ({len(df)} images):")
    for disease in CHEXPERT14:
        if disease in df.columns:
            pos_count = df[disease].sum()
            pos_pct = (pos_count / len(df)) * 100
            print(f"  {disease}: {pos_count} ({pos_pct:.1f}%)")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Create ground truth from phaseA_manifest.jsonl")
    parser.add_argument("--input", type=str, required=True, help="Path to phaseA_manifest.jsonl")
    parser.add_argument("--output", type=str, default="data/evaluation_manifest_phaseA.csv", help="Output CSV path")
    parser.add_argument("--predictions", type=str, default=None, help="Path to predictions CSV (to extract specific images)")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to generate (if --predictions not provided)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    data = load_phaseA_data(input_path)
    
    # Extract ground truth for specific images or create balanced sample
    if args.predictions:
        sampled_df = extract_for_specific_images(data, args.predictions)
    else:
        sampled_df = create_balanced_sample(data, args.n_samples)
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_path, index=False)
    
    print(f"\n💾 Saved ground truth manifest to {output_path}")
    print(f"   Images: {len(sampled_df)}")
    print(f"   Columns: {list(sampled_df.columns)}")
    
    # Show sample of data
    print(f"\n📋 Sample of converted data:")
    print(sampled_df[['image', 'No Finding', 'Consolidation', 'Edema', 'Pneumonia']].head())

if __name__ == "__main__":
    main()
