#!/usr/bin/env python3
"""
Convert a CheXpert ensemble predictions CSV (with y_pred_* columns) into a
source CSV compatible with our blending pipeline by renaming to y_cal_* and
optionally restricting to a subset of labels (e.g., the five competition labels).

Usage:
  python src/utils/convert_chexpert_ensemble_to_source.py \
    --in_csv server_synced/ensemble_pipeline/train_blend_probs.csv \
    --out_csv outputs_full/chex5/train_calibrated.csv \
    --labels Atelectasis Cardiomegaly Consolidation Edema Pleural\ Effusion

Note: This does not calibrate; it simply exposes the provided probabilities
under the column prefix the rest of the pipeline expects (y_cal_).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_FIVE = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--labels", nargs="*", default=DEFAULT_FIVE)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    out = pd.DataFrame({"filename": df["filename"]})
    for lab in args.labels:
        src_col = f"y_pred_{lab}"
        if src_col not in df.columns:
            # skip silently if absent
            continue
        out[f"y_cal_{lab}"] = df[src_col].astype(float)
    out.to_csv(args.out_csv, index=False)
    print(f"✅ Wrote {args.out_csv} with labels: {args.labels}")


if __name__ == "__main__":
    main()

