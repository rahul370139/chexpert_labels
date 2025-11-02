#!/usr/bin/env python3
"""
Selective TXR inference for specific labels only.

This runs TorchXRayVision for 5 worst-performing labels and blends with existing predictions.
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.inference.txr_infer import infer_txr, load_image_paths, build_label_mapping, CHEXPERT13
from src.common.labels import CHEXPERT13 as CHEXPERT13_LIST

import torch
try:
    import torchxrayvision as xrv
except ImportError:
    raise SystemExit("torchxrayvision required. Install: pip install torchxrayvision")


def selective_txr_inference(
    images: list[Path],
    labels_to_use: list[str],
    device: torch.device,
    output_csv: Path,
    model_weights: str = "resnet50-res512-all",
    batch_size: int = 16,
    num_workers: int = 2,
) -> pd.DataFrame:
    """
    Run TXR inference for specific labels only, then merge with existing predictions.
    """
    print("=" * 80)
    print(f"SELECTIVE TXR INFERENCE FOR {len(labels_to_use)} LABELS")
    print("=" * 80)
    print(f"Labels: {', '.join(labels_to_use)}")
    
    # Run full TXR inference
    print(f"\n🔄 Running TXR inference...")
    txr_results = infer_txr(
        images=images,
        model_weights=model_weights,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Extract only the labels we care about
    print(f"\n🔄 Extracting probabilities for selected labels...")
    rows = []
    
    for img_path, record in txr_results.items():
        row = {"filename": Path(img_path).name, "image": str(img_path)}
        
        # Extract probabilities for selected labels
        # TXR uses "prob_<Label>" format (e.g., "prob_Enlarged Cardiomediastinum")
        for label in labels_to_use:
            # Try exact match first
            prob_key = f"prob_{label}"
            prob_value = None
            
            if prob_key in record:
                prob_value = record[prob_key]
            else:
                # Try normalized names (TXR may use different formatting)
                normalized = label.lower().replace(" ", "_")
                for key in record.keys():
                    if key.startswith("prob_") and key.lower().replace(" ", "_") == f"prob_{normalized}":
                        prob_value = record[key]
                        break
                # Also check raw_probs dict if it exists
                if prob_value is None and "raw_probs" in record:
                    import json
                    raw_dict = json.loads(record["raw_probs"]) if isinstance(record["raw_probs"], str) else record["raw_probs"]
                    # TXR pathologies might have different names
                    for path_name, path_prob in raw_dict.items():
                        path_norm = path_name.lower().replace("_", " ").replace("-", " ")
                        label_norm = label.lower()
                        if path_norm == label_norm or label_norm in path_norm or path_norm in label_norm:
                            prob_value = path_prob
                            break
            
            if prob_value is None or (isinstance(prob_value, float) and np.isnan(prob_value)):
                prob_value = np.nan

            row[f"prob_{label}"] = prob_value
            row[f"y_pred_{label}"] = prob_value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ TXR probabilities saved to: {output_csv}")
    print(f"   Rows: {len(df)}")
    
    # Stats per label
    for label in labels_to_use:
        col = f"y_pred_{label}"
        if col in df.columns:
            probs = df[col].dropna()
            if len(probs) > 0:
                print(f"   {label}: mean={probs.mean():.4f}, median={probs.median():.4f}, coverage={len(probs)/len(df)*100:.1f}%")
    
    return df


def blend_txr_with_existing(
    existing_csv: Path,
    txr_csv: Path,
    labels_to_blend: list[str],
    output_csv: Path,
    txr_weight: float = 0.7,
) -> pd.DataFrame:
    """
    Blend TXR predictions with existing predictions for selected labels.
    
    For labels in labels_to_blend: use weighted average of existing and TXR
    For other labels: keep existing predictions
    """
    print("=" * 80)
    print("BLENDING TXR WITH EXISTING PREDICTIONS")
    print("=" * 80)
    
    print(f"\n📂 Loading existing predictions: {existing_csv}")
    existing_df = pd.read_csv(existing_csv)
    print(f"   Rows: {len(existing_df)}")
    
    print(f"\n📂 Loading TXR predictions: {txr_csv}")
    txr_df = pd.read_csv(txr_csv)
    print(f"   Rows: {len(txr_df)}")
    
    # Ensure filename column
    if "filename" not in existing_df.columns:
        if "image" in existing_df.columns:
            existing_df["filename"] = existing_df["image"].apply(lambda x: Path(x).name if pd.notna(x) else "")
    
    if "filename" not in txr_df.columns:
        if "image" in txr_df.columns:
            txr_df["filename"] = txr_df["image"].apply(lambda x: Path(x).name if pd.notna(x) else "")
    
    # Merge
    merged = existing_df.merge(txr_df, on="filename", how="left", suffixes=("_existing", "_txr"))
    
    print(f"\n🔄 Blending probabilities (TXR weight={txr_weight})...")
    
    blended = merged.copy()
    
    for label in labels_to_blend:
        existing_col = f"y_cal_{label}" if f"y_cal_{label}" in merged.columns else label
        txr_col = f"y_pred_{label}_txr" if f"y_pred_{label}_txr" in merged.columns else f"y_pred_{label}"
        output_col = f"y_cal_{label}"
        
        if existing_col not in merged.columns:
            print(f"⚠️  Skipping {label}: no existing column")
            continue
        
        if txr_col not in merged.columns:
            print(f"⚠️  Skipping {label}: no TXR column, using existing only")
            blended[output_col] = merged[existing_col]
            continue
        
        # Blend
        existing_probs = merged[existing_col].fillna(0.0).astype(float)
        txr_probs = merged[txr_col].fillna(0.0).astype(float)
        
        # Weighted average
        blended_probs = (1 - txr_weight) * existing_probs + txr_weight * txr_probs
        
        # Handle NaNs (if TXR missing, fall back to existing)
        blended_probs = blended_probs.fillna(existing_probs)
        
        blended[output_col] = blended_probs
        
        print(f"   ✅ {label}: blended {len(blended_probs)} probabilities")
    
    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(output_csv, index=False)
    
    print(f"\n✅ Blended predictions saved to: {output_csv}")
    
    return blended


def main():
    parser = argparse.ArgumentParser(description="Selective TXR inference and blending")
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Image list file or directory"
    )
    parser.add_argument(
        "--existing_csv",
        type=Path,
        default=None,
        help="Optional existing predictions CSV to blend with"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["Enlarged Cardiomediastinum", "Lung Lesion", "Pneumothorax", "Pleural Other", "Fracture"],
        help="Labels to use TXR for (default: 5 worst performers)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs_full/final/predictions_txr_blended.csv"),
        help="Output blended predictions CSV"
    )
    parser.add_argument(
        "--txr_weight",
        type=float,
        default=0.7,
        help="Weight for TXR predictions (0-1, default 0.7)"
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default="resnet50-res512-all",
        help="TorchXRayVision weights identifier (6.8GB CheXpert ensemble by default)"
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["cpu", "cuda", "mps"],
        help="Device"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers"
    )
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    
    # Resolve paths
    images_path = (project_root / args.images).resolve()
    existing_csv = (project_root / args.existing_csv).resolve() if args.existing_csv else None
    output_csv = (project_root / args.output).resolve()
    
    if not images_path.exists():
        print(f"❌ Images path not found: {images_path}")
        sys.exit(1)
    
    # Load images
    images = load_image_paths(images_path)
    print(f"📂 Loaded {len(images)} images")
    
    # Device
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU")
        device = torch.device("cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    # Step 1: Run TXR inference
    txr_csv = output_csv.parent / f"{output_csv.stem}_txr_only.csv"
    txr_only_csv = output_csv if existing_csv is None else output_csv.parent / f"{output_csv.stem}_txr_only.csv"
    selective_txr_inference(
        images,
        args.labels,
        device,
        txr_only_csv,
        model_weights=args.model_weights,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    if existing_csv:
        if not existing_csv.exists():
            print(f"❌ Existing CSV not found: {existing_csv}")
            sys.exit(1)
        
        blend_txr_with_existing(
            existing_csv,
            txr_only_csv,
            args.labels,
            output_csv,
            args.txr_weight,
        )

        print("\n" + "=" * 80)
        print("SELECTIVE TXR INFERENCE + BLEND COMPLETE")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("SELECTIVE TXR INFERENCE COMPLETE (no blending requested)")
        print("=" * 80)


if __name__ == "__main__":
    main()
