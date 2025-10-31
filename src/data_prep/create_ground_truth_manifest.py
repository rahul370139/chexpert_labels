"""
Create Ground Truth Manifest from curriculum_train_final_clean.jsonl

This script extracts 1000 diverse samples from both Stage A and Stage B,
ensuring balanced representation of different disease patterns.
"""

import json
import pandas as pd
import random
from pathlib import Path
from collections import Counter
import argparse

# CheXpert 14 labels (matching our evaluation)
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

def load_curriculum_data(jsonl_path: Path) -> pd.DataFrame:
    """Load and parse the curriculum JSONL file."""
    print(f"📖 Loading data from {jsonl_path}")
    
    data = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed line {line_num}: {e}")
                continue
    
    df = pd.DataFrame(data)
    print(f"✅ Loaded {len(df)} records")
    return df

def analyze_data_distribution(df: pd.DataFrame) -> dict:
    """Analyze the distribution of diseases and stages."""
    print("\n📊 Data Distribution Analysis:")
    
    # Stage distribution
    stage_counts = df['stage'].value_counts()
    print(f"Stage A: {stage_counts.get('A', 0)} samples")
    print(f"Stage B: {stage_counts.get('B', 0)} samples")
    
    # Disease distribution
    disease_counts = {}
    for disease in CHEXPERT14:
        if disease in df.columns:
            disease_counts[disease] = df[disease].sum()
        else:
            # Extract from chexpert_labels column
            disease_counts[disease] = df['chexpert_labels'].apply(
                lambda x: x.get(disease, 0) if isinstance(x, dict) else 0
            ).sum()
    
    print("\nDisease Distribution:")
    for disease, count in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {disease}: {count}")
    
    return disease_counts

def extract_chexpert_labels(row) -> dict:
    """Extract CheXpert labels from the chexpert_labels column."""
    labels = row['chexpert_labels']
    if isinstance(labels, dict):
        # Ensure all 14 labels are present
        result = {}
        for disease in CHEXPERT14:
            result[disease] = labels.get(disease, 0)
        return result
    else:
        # Fallback: return zeros
        return {disease: 0 for disease in CHEXPERT14}

def create_balanced_sample(df: pd.DataFrame, target_size: int = 1000) -> pd.DataFrame:
    """
    Create a balanced sample ensuring:
    1. Mix of Stage A and Stage B
    2. Representation of different disease patterns
    3. Diverse clinical presentations
    """
    print(f"\n🎯 Creating balanced sample of {target_size} images...")
    
    # Extract CheXpert labels to separate columns
    chexpert_df = df['chexpert_labels'].apply(pd.Series)
    df_with_labels = pd.concat([df, chexpert_df], axis=1)
    
    # Add missing columns with default values
    for disease in CHEXPERT14:
        if disease not in df_with_labels.columns:
            df_with_labels[disease] = 0
            print(f"  ⚠️  Added missing column: {disease} (default=0)")
    
    # Strategy: Stratified sampling
    samples = []
    
    # 1. Ensure "No Finding" cases (normal chest X-rays)
    no_finding_cases = df_with_labels[df_with_labels['No Finding'] == 1]
    if len(no_finding_cases) > 0:
        no_finding_sample = no_finding_cases.sample(
            min(150, len(no_finding_cases)), random_state=42
        )
        samples.append(no_finding_sample)
        print(f"  📋 No Finding cases: {len(no_finding_sample)}")
    
    # 2. Ensure each disease has some representation
    disease_samples = []
    for disease in CHEXPERT14:
        if disease == "No Finding":
            continue
            
        disease_cases = df_with_labels[df_with_labels[disease] == 1]
        if len(disease_cases) > 0:
            # Sample 20-50 cases per disease depending on prevalence
            sample_size = min(50, max(20, len(disease_cases) // 10))
            disease_sample = disease_cases.sample(
                min(sample_size, len(disease_cases)), random_state=42
            )
            disease_samples.append(disease_sample)
            print(f"  🏥 {disease}: {len(disease_sample)} cases")
        else:
            print(f"  ⚠️  {disease}: No cases found")
    
    # 3. Add remaining samples randomly
    used_indices = set()
    for sample_df in samples + disease_samples:
        used_indices.update(sample_df.index)
    
    remaining_df = df_with_labels[~df_with_labels.index.isin(used_indices)]
    remaining_needed = target_size - sum(len(s) for s in samples + disease_samples)
    
    if remaining_needed > 0 and len(remaining_df) > 0:
        remaining_sample = remaining_df.sample(
            min(remaining_needed, len(remaining_df)), random_state=42
        )
        samples.append(remaining_sample)
        print(f"  🎲 Random additional: {len(remaining_sample)} cases")
    
    # Combine all samples
    if samples:
        final_df = pd.concat(samples, ignore_index=True)
    else:
        final_df = df_with_labels.sample(target_size, random_state=42)
    
    # Remove duplicates
    final_df = final_df.drop_duplicates(subset=['image_path'])
    
    # Ensure we have exactly target_size
    if len(final_df) > target_size:
        final_df = final_df.sample(target_size, random_state=42)
    elif len(final_df) < target_size:
        # Add more random samples if needed
        remaining_df = df_with_labels[~df_with_labels.index.isin(final_df.index)]
        additional_needed = target_size - len(final_df)
        if len(remaining_df) >= additional_needed:
            additional_sample = remaining_df.sample(additional_needed, random_state=42)
            final_df = pd.concat([final_df, additional_sample], ignore_index=True)
    
    print(f"✅ Final sample size: {len(final_df)}")
    return final_df

def create_evaluation_manifest(sample_df: pd.DataFrame, output_path: Path) -> None:
    """Create the evaluation manifest CSV file."""
    print(f"\n📝 Creating evaluation manifest: {output_path}")
    
    manifest_data = []
    
    for _, row in sample_df.iterrows():
        # Extract CheXpert labels
        chexpert_labels = extract_chexpert_labels(row)
        
        manifest_row = {
            'image': row['image_path'],
            'impression': row['impression'],
            'stage': row['stage'],
            **chexpert_labels
        }
        manifest_data.append(manifest_row)
    
    manifest_df = pd.DataFrame(manifest_data)
    manifest_df.to_csv(output_path, index=False)
    
    print(f"✅ Manifest saved with {len(manifest_df)} samples")
    
    # Show final distribution
    print("\n📊 Final Distribution:")
    stage_counts = manifest_df['stage'].value_counts()
    print(f"Stage A: {stage_counts.get('A', 0)}")
    print(f"Stage B: {stage_counts.get('B', 0)}")
    
    print("\nDisease Distribution:")
    for disease in CHEXPERT14:
        count = manifest_df[disease].sum()
        print(f"  {disease}: {count}")

def create_image_list(sample_df: pd.DataFrame, output_path: Path) -> None:
    """Create a text file with image paths for easy processing."""
    print(f"\n📋 Creating image list: {output_path}")
    
    image_paths = sample_df['image_path'].tolist()
    with open(output_path, 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")
    
    print(f"✅ Image list saved with {len(image_paths)} paths")

def main():
    parser = argparse.ArgumentParser(description="Create ground truth manifest from curriculum data")
    parser.add_argument("--input", type=str, 
                       default="radiology_report/src/data/processed/curriculum_train_final_clean.jsonl",
                       help="Path to curriculum JSONL file")
    parser.add_argument("--output_dir", type=str, default="chexagent_chexpert_eval/data",
                       help="Output directory for manifest files")
    parser.add_argument("--sample_size", type=int, default=1000,
                       help="Number of samples to extract")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    df = load_curriculum_data(input_path)
    
    # Analyze distribution
    analyze_data_distribution(df)
    
    # Create balanced sample
    sample_df = create_balanced_sample(df, args.sample_size)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manifest and image list
    manifest_path = output_dir / "evaluation_manifest_1000.csv"
    image_list_path = output_dir / "image_list_1000.txt"
    
    create_evaluation_manifest(sample_df, manifest_path)
    create_image_list(sample_df, image_list_path)
    
    print(f"\n🎉 Ground truth creation complete!")
    print(f"📁 Files created in: {output_dir}")
    print(f"  - {manifest_path.name}")
    print(f"  - {image_list_path.name}")

if __name__ == "__main__":
    main()
