"""
Use CheXagent's official disease_identification method for CheXpert label extraction.

This script uses the official CheXagent class from the Stanford repository
to perform structured disease identification instead of prompt engineering.
"""

import sys
import os
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add CheXagent repo to path
sys.path.append(str(Path(__file__).parent / "chexagent_repo"))

try:
    from model_chexagent.chexagent import CheXagent
    print("✅ Successfully imported CheXagent class")
except ImportError as e:
    print(f"❌ Failed to import CheXagent: {e}")
    sys.exit(1)

# 13 CheXpert findings (excluding "No Finding" which we compute separately)
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

# Full 14 labels including "No Finding"
CHEXPERT14 = CHEXPERT13 + ["No Finding"]

def extract_diseases_from_text(text: str) -> Dict[str, int]:
    """
    Extract disease names from CheXagent's text response.
    
    Args:
        text: Raw text response from CheXagent
        
    Returns:
        Dictionary mapping disease names to 1 (present) or 0 (absent)
    """
    text_lower = text.lower()
    diseases_found = {}
    
    # Initialize all diseases as absent (0)
    for disease in CHEXPERT13:
        diseases_found[disease] = 0
    
    # Check for each disease name in the text
    for disease in CHEXPERT13:
        disease_lower = disease.lower()
        # Check if disease name appears in the text
        if disease_lower in text_lower:
            diseases_found[disease] = 1
    
    return diseases_found

def normalize_prediction(value: Any) -> int:
    """
    Normalize prediction value to {-1, 0, 1}.
    
    Args:
        value: Raw prediction from CheXagent (could be int, float, str, etc.)
        
    Returns:
        Normalized integer in {-1, 0, 1}
    """
    if isinstance(value, (int, float)):
        if value > 0.5:
            return 1
        elif value < -0.5:
            return -1
        else:
            return 0
    elif isinstance(value, str):
        value_lower = value.lower().strip()
        if value_lower in ["yes", "positive", "present", "1", "true"]:
            return 1
        elif value_lower in ["no", "negative", "absent", "0", "false"]:
            return 0
        elif value_lower in ["uncertain", "unclear", "-1", "u"]:
            return -1
        else:
            return 0  # Default to 0 for unknown strings
    else:
        return 0  # Default to 0 for unknown types

def process_images_with_chexagent(
    image_paths: List[Path], 
    device: str = "cpu",
    output_csv: Path = Path("chexagent_disease_predictions.csv")
) -> List[Dict[str, Any]]:
    """
    Process images using CheXagent's disease_identification method.
    
    Args:
        image_paths: List of image file paths
        device: Device to run on ("cpu", "mps", "cuda")
        output_csv: Output CSV file path
        
    Returns:
        List of prediction dictionaries
    """
    print(f"🔧 Initializing CheXagent on device: {device}")
    chex = CheXagent(device=device)
    
    print(f"🔍 Processing {len(image_paths)} images...")
    
    # Convert Path objects to strings
    path_strings = [str(p) for p in image_paths]
    
    # Process each image individually
    rows = []
    raw_outputs = []  # Store raw outputs for debugging
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {img_path.name}")
        
        # Use disease_identification for single image
        result = chex.disease_identification([str(img_path)], CHEXPERT13)
        
        # result is a string, not a list
        response_text = result if isinstance(result, str) else str(result)
        print(f"  Raw response: {response_text[:200]}...")
        
        # Store full raw output
        raw_outputs.append({
            "image": str(img_path),
            "raw_output": response_text
        })
        
        # Extract diseases from text response
        diseases_found = extract_diseases_from_text(response_text)
        
        row = {"image": str(img_path)}
        normalized_labels = {}
        
        for label in CHEXPERT13:
            normalized_value = diseases_found[label]
            normalized_labels[label] = normalized_value
            row[label] = normalized_value
        
        # Compute "No Finding" - set to 1 if all other labels are 0
        other_values = [normalized_labels[k] for k in CHEXPERT13]
        if all(v == 0 for v in other_values):
            row["No Finding"] = 1
        else:
            row["No Finding"] = 0
        
        rows.append(row)
        
        # Print sample prediction
        print(f"  Sample predictions: {dict(list(normalized_labels.items())[:3])}...")
    
    # Write CSV
    fieldnames = ["image"] + CHEXPERT14
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"📄 Results saved to: {output_csv}")
    
    # Save raw outputs for debugging
    if raw_outputs:
        raw_output_file = output_csv.parent / f"raw_outputs_{output_csv.stem}.txt"
        with raw_output_file.open("w", encoding="utf-8") as f:
            for entry in raw_outputs:
                f.write(f"\n{'='*80}\n")
                f.write(f"Image: {entry['image']}\n")
                f.write(f"{'='*80}\n")
                f.write(f"{entry['raw_output']}\n")
        print(f"📝 Full raw outputs saved to: {raw_output_file}")
    
    return rows

def collect_image_paths(input_path: Path) -> List[Path]:
    """Collect image paths from various input formats."""
    if input_path.is_file():
        if input_path.suffix.lower() == ".txt":
            # Text file with image paths
            paths = []
            for line in input_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    path = Path(line)
                    if path.exists():
                        paths.append(path)
                    else:
                        print(f"⚠️  Warning: Image not found: {line}")
            return paths
        else:
            # Single image file
            return [input_path]
    elif input_path.is_dir():
        # Directory of images
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
        paths = []
        for ext in extensions:
            paths.extend(input_path.glob(f"*{ext}"))
            paths.extend(input_path.glob(f"*{ext.upper()}"))
        return sorted(paths)
    else:
        raise ValueError(f"Invalid input path: {input_path}")

def main():
    parser = argparse.ArgumentParser(description="Use CheXagent's disease_identification for CheXpert labels")
    parser.add_argument("--images", type=str, required=True, 
                       help="Path to image file, directory, or .txt file with image paths")
    parser.add_argument("--out_csv", type=str, default="chexagent_disease_predictions.csv",
                       help="Output CSV file path")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cpu", "mps", "cuda"],
                       help="Device to run on")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"🎯 Using device: {device}")
    
    # Collect image paths
    input_path = Path(args.images)
    image_paths = collect_image_paths(input_path)
    
    if not image_paths:
        print(f"❌ No images found in: {input_path}")
        return
    
    print(f"📸 Found {len(image_paths)} images")
    
    # Process images
    try:
        rows = process_images_with_chexagent(
            image_paths, 
            device=device,
            output_csv=Path(args.out_csv)
        )
        
        print(f"✅ Successfully processed {len(rows)} images")
        
        # Show sample results
        print("\n📊 Sample results:")
        for i, row in enumerate(rows[:3], 1):
            print(f"\nImage {i}: {Path(row['image']).name}")
            for label in CHEXPERT14:
                print(f"  {label}: {row[label]}")
                
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
