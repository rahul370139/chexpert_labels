#!/usr/bin/env python3
"""
Unified threshold tuning script - consolidates all threshold optimization approaches.

Features:
1. Principled tuning (F-beta or min-precision) using threshold_tuner_impl
2. Per-label optimization with precision-recall curves
3. Automatic fallback for low-probability labels (raises threshold to reduce FPs)
4. Support for calibrated and raw probabilities

This is the CANONICAL threshold tuning script. Use this instead of:
- src/thresholds/threshold_tuner.py (old)
- src/evaluation/fix_thresholds_optimized.py (consolidated here)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.labels import get_label_list, CHEXPERT13
from src.thresholds.threshold_tuner_impl import tune_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimise per-label thresholds on calibrated probabilities.")
    parser.add_argument("--calibrated_train_csv", required=True, help="CSV containing y_true_<L> and score columns.")
    parser.add_argument("--labels", default="chexpert13")
    parser.add_argument("--exclude_labels", nargs="*", default=[], help="Labels to exclude from tuning")
    parser.add_argument("--score_prefix", default="y_cal_", help="Prefix for input probability columns.")
    parser.add_argument("--copy_prefix", default="y_pred_", help="Temporary prefix used for optimisation.")
    parser.add_argument("--mode", default="fbeta", choices=["fbeta", "min_precision"])
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--min_precision", type=float, default=0.6)
    parser.add_argument("--min_thresholds_json", default=None, help="Optional JSON with per-label floors.")
    parser.add_argument("--out_thresholds_json", default="outputs/thresholds/thresholds.json")
    parser.add_argument("--out_summary_csv", default="outputs/thresholds/summary.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.calibrated_train_csv)
    labels = get_label_list(args.labels)
    if args.exclude_labels:
        labels = [l for l in labels if l not in set(args.exclude_labels)]

    # Duplicate probability columns to the expected naming convention (y_pred_<Label>)
    # Also ensure y_true_<Label> columns exist (may be y_true_<Label>_true or raw label columns)
    for label in labels:
        src_col = f"{args.score_prefix}{label}"
        dst_col = f"{args.copy_prefix}{label}"
        if src_col not in df.columns:
            # Try y_pred_<Label> if score_prefix column doesn't exist
            if f"y_pred_{label}" in df.columns:
                src_col = f"y_pred_{label}"
            else:
                continue
        df[dst_col] = df[src_col]
        
        # Ensure y_true_<L> exists (may need to rename from y_true_<L>_true or raw label)
        true_col = f"y_true_{label}"
        if true_col not in df.columns:
            if f"y_true_{label}_true" in df.columns:
                df[true_col] = df[f"y_true_{label}_true"]
            elif label in df.columns:
                # Raw label column exists, use it as ground truth
                df[true_col] = df[label]

    tmp_path = Path(args.calibrated_train_csv).with_suffix(".tmp_threshold.csv")
    df.to_csv(tmp_path, index=False)

    floors = None
    if args.min_thresholds_json:
        floors = json.loads(Path(args.min_thresholds_json).read_text())

    tune_thresholds(
        csv_path=str(tmp_path),
        out_json=args.out_thresholds_json,
        out_metrics=args.out_summary_csv,
        mode=args.mode,
        beta=args.beta,
        min_macro_precision=args.min_precision,
        labels=labels,
        min_floors=floors,
    )
    tmp_path.unlink(missing_ok=True)
    
    # Post-process: Apply conservative thresholds for labels with very low median probabilities
    print("\n🔄 Post-processing thresholds for low-probability labels...")
    thresholds = json.loads(Path(args.out_thresholds_json).read_text())
    
    df_check = pd.read_csv(args.calibrated_train_csv)
    for label in labels:
        score_col = f"{args.score_prefix}{label}"
        if score_col not in df_check.columns:
            continue
        
        probs = df_check[score_col].dropna().astype(float)
        median_prob = np.median(probs)
        
        # If median is very low (< 0.01), raise threshold to >= 0.60 to reduce false positives
        if median_prob < 0.01 and thresholds.get(label, 0.5) < 0.60:
            old_thresh = thresholds.get(label, 0.5)
            thresholds[label] = max(0.60, old_thresh)
            print(f"   ⚠️  {label}: Raised threshold from {old_thresh:.3f} to {thresholds[label]:.3f} (median prob={median_prob:.4f} very low)")
    
    # Save updated thresholds
    Path(args.out_thresholds_json).write_text(json.dumps(thresholds, indent=2))
    
    print(f"\n✅ Thresholds stored in {args.out_thresholds_json}")
    print(f"✅ Metrics summary written to {args.out_summary_csv}")


if __name__ == "__main__":
    main()
