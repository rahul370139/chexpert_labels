#!/usr/bin/env python3
"""
Extract image list from CheXagent CSV and validate existence.

Converts image paths to absolute paths and validates all images exist.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path


def extract_image_list(
    csv_path: Path,
    output_txt: Path,
    missing_file: Path,
) -> None:
    """
    Extract image paths from CSV, convert to absolute, validate existence.
    
    Args:
        csv_path: Input CheXagent CSV
        output_txt: Output text file with image paths (one per line)
        missing_file: Output file for missing images
    """
    print("=" * 80)
    print("EXTRACTING IMAGE LIST FROM CSV")
    print("=" * 80)
    
    # Load CSV
    print(f"\n📂 Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df)}")
    
    # Get image column
    if "image" not in df.columns:
        if "filename" in df.columns:
            print("❌ CSV has 'filename' but not 'image' column")
            print("   Cannot determine full image paths")
            sys.exit(1)
        else:
            print("❌ CSV missing both 'image' and 'filename' columns")
            sys.exit(1)
    
    # Convert to absolute paths
    print(f"\n🔄 Converting to absolute paths...")
    image_paths = []
    missing = []
    
    for idx, path_str in enumerate(df["image"]):
        if pd.isna(path_str):
            missing.append((idx, "NaN"))
            continue
        
        path = Path(path_str)
        if not path.is_absolute():
            # Try to resolve relative to current working directory
            abs_path = Path(path).resolve()
        else:
            abs_path = path
        
        if abs_path.exists():
            image_paths.append(str(abs_path.absolute()))
        else:
            missing.append((idx, str(abs_path)))
    
    print(f"   Found: {len(image_paths)} images")
    print(f"   Missing: {len(missing)} images")
    
    if missing:
        print(f"\n⚠️  Missing images:")
        for idx, path in missing[:10]:  # Show first 10
            print(f"   Row {idx}: {path}")
        if len(missing) > 10:
            print(f"   ... and {len(missing) - 10} more")
        
        # Write missing file
        missing_file.parent.mkdir(parents=True, exist_ok=True)
        with open(missing_file, "w") as f:
            f.write("Row,Path\n")
            for idx, path in missing:
                f.write(f"{idx},{path}\n")
        
        print(f"\n❌ Found {len(missing)} missing images")
        print(f"   Missing list saved to: {missing_file}")
        print(f"   Cannot proceed without all images")
        sys.exit(1)
    
    # Write image list
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w") as f:
        for path in image_paths:
            f.write(f"{path}\n")
    
    print(f"\n✅ Image list saved to: {output_txt}")
    print(f"   Total images: {len(image_paths)}")
    
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Extract image list from CheXagent CSV")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Input CheXagent CSV path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/image_list_phaseA_5k_absolute.txt"),
        help="Output text file with image paths"
    )
    parser.add_argument(
        "--missing_file",
        type=Path,
        default=Path("results/missing_files.txt"),
        help="Output file for missing images (if any)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    csv_path = (project_root / args.csv).resolve()
    output_txt = (project_root / args.output).resolve()
    missing_file = (project_root / args.missing_file).resolve()
    
    if not csv_path.exists():
        print(f"❌ CSV not found: {csv_path}")
        sys.exit(1)
    
    extract_image_list(csv_path, output_txt, missing_file)


if __name__ == "__main__":
    main()

