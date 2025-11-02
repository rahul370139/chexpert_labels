#!/usr/bin/env python3
"""
Apply three-class thresholds: predict -1 (uncertain) if prob is between lower and upper threshold.

Decision logic:
- prob >= upper_threshold → 1 (positive)
- prob < lower_threshold → 0 (negative)
- lower_threshold <= prob < upper_threshold → -1 (uncertain)
"""

import argparse
import pandas as pd
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.common.labels import CHEXPERT13


def apply_three_class_thresholds(
    probs_csv: Path,
    thresholds_json: Path,
    output_csv: Path,
    uncertainty_margin: float = 0.15,
) -> None:
    """
    Apply three-class thresholds to probabilities.
    
    Args:
        probs_csv: Input probabilities CSV
        thresholds_json: Single threshold JSON (we'll derive upper/lower)
        output_csv: Output predictions CSV with -1/0/1
        uncertainty_margin: Margin around threshold for uncertain zone
                          (e.g., 0.15 means uncertain = [tau*0.85, tau*1.15])
    """
    print("=" * 80)
    print("APPLYING THREE-CLASS THRESHOLDS")
    print("=" * 80)
    
    # Load probabilities
    print(f"\n📂 Loading probabilities: {probs_csv}")
    probs_df = pd.read_csv(probs_csv)
    print(f"   Rows: {len(probs_df)}")
    
    # Load thresholds
    print(f"\n📂 Loading thresholds: {thresholds_json}")
    with open(thresholds_json) as f:
        single_thresholds = json.load(f)
    
    # Derive three-class thresholds
    print(f"\n🔄 Deriving three-class thresholds (uncertainty margin: {uncertainty_margin})")
    three_class_thresholds = {}
    
    for label in CHEXPERT13:
        tau = single_thresholds.get(label, 0.5)
        lower = max(0.0, tau * (1 - uncertainty_margin))
        upper = min(1.0, tau * (1 + uncertainty_margin))
        
        three_class_thresholds[label] = {
            "lower": lower,
            "upper": upper,
            "original": tau
        }
        
        print(f"   {label}:")
        print(f"     Original: {tau:.3f}")
        print(f"     Lower (negative): < {lower:.3f}")
        print(f"     Uncertain: [{lower:.3f}, {upper:.3f})")
        print(f"     Positive: >= {upper:.3f}")
    
    # Apply three-class logic
    print(f"\n🔄 Applying three-class predictions...")
    preds_df = probs_df[["filename"]].copy()
    
    uncertain_counts = {}
    positive_counts = {}
    negative_counts = {}
    
    for label in CHEXPERT13:
        prob_col = f"y_cal_{label}"
        if prob_col not in probs_df.columns:
            # Try alternate names
            prob_col = label if label in probs_df.columns else None
            if prob_col is None:
                print(f"⚠️  Skipping {label}: no probability column found")
                continue
        
        thresh_info = three_class_thresholds[label]
        lower = thresh_info["lower"]
        upper = thresh_info["upper"]
        
        probs = probs_df[prob_col].fillna(0.0).astype(float)
        
        # Three-class decision
        preds = pd.Series(0, index=probs.index)  # Default to negative
        preds[probs >= upper] = 1  # Positive
        preds[(probs >= lower) & (probs < upper)] = -1  # Uncertain
        
        preds_df[label] = preds
        
        # Counts
        uncertain_counts[label] = (preds == -1).sum()
        positive_counts[label] = (preds == 1).sum()
        negative_counts[label] = (preds == 0).sum()
        
        print(f"   {label}: {positive_counts[label]} pos, {uncertain_counts[label]} uncertain, {negative_counts[label]} neg")
    
    # Compute No Finding (1 only if all others are 0 or -1, and at least one is -1 means we're uncertain)
    # Actually, No Finding should be 1 if all others are 0 (certain negative)
    other_preds = preds_df[CHEXPERT13].fillna(0)
    preds_df["No Finding"] = ((other_preds == 0).all(axis=1)).astype(int)
    
    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Three-class predictions saved to: {output_csv}")
    
    # Summary
    print(f"\n📊 Summary:")
    total_preds = len(preds_df) * len(CHEXPERT13)
    total_uncertain = sum(uncertain_counts.values())
    total_positive = sum(positive_counts.values())
    total_negative = total_preds - total_uncertain - total_positive
    
    print(f"   Positive (1): {total_positive:,} ({total_positive/total_preds*100:.1f}%)")
    print(f"   Uncertain (-1): {total_uncertain:,} ({total_uncertain/total_preds*100:.1f}%)")
    print(f"   Negative (0): {total_negative:,} ({total_negative/total_preds*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("THREE-CLASS PREDICTION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Apply three-class thresholds")
    parser.add_argument(
        "--probs_csv",
        type=Path,
        required=True,
        help="Input probabilities CSV"
    )
    parser.add_argument(
        "--thresholds_json",
        type=Path,
        required=True,
        help="Single threshold JSON (we derive upper/lower)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs_full/final/test_preds_three_class.csv"),
        help="Output three-class predictions CSV"
    )
    parser.add_argument(
        "--uncertainty_margin",
        type=float,
        default=0.15,
        help="Margin around threshold for uncertain zone (default 0.15 = ±15%%)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    probs_csv = (project_root / args.probs_csv).resolve()
    thresholds_json = (project_root / args.thresholds_json).resolve()
    output_csv = (project_root / args.output).resolve()
    
    if not probs_csv.exists():
        print(f"❌ Probabilities CSV not found: {probs_csv}")
        sys.exit(1)
    
    if not thresholds_json.exists():
        print(f"❌ Thresholds JSON not found: {thresholds_json}")
        sys.exit(1)
    
    apply_three_class_thresholds(probs_csv, thresholds_json, output_csv, args.uncertainty_margin)


if __name__ == "__main__":
    main()

