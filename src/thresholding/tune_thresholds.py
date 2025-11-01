#!/usr/bin/env python3
"""Convenience CLI around threshold tuning with configurable floors and prefixes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.labels import get_label_list
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
    print(f"✅ Thresholds stored in {args.out_thresholds_json}")
    print(f"✅ Metrics summary written to {args.out_summary_csv}")


if __name__ == "__main__":
    main()
