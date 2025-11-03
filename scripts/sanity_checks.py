#!/usr/bin/env python3
"""Lightweight sanity checks for final predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity checks for CheXpert predictions")
    parser.add_argument("--preds", required=True, help="Path to predictions CSV (post-gating)")
    parser.add_argument("--labels", required=True, help="Path to ground-truth CSV for the same split")
    parser.add_argument("--out", required=True, help="Markdown file to write summary")
    args = parser.parse_args()

    preds = pd.read_csv(args.preds)
    labels = pd.read_csv(args.labels)

    # Ensure filename alignment
    if "filename" not in preds.columns and "image" in preds.columns:
        preds["filename"] = preds["image"].apply(lambda x: Path(str(x)).name)
    if "filename" not in labels.columns and "image" in labels.columns:
        labels["filename"] = labels["image"].apply(lambda x: Path(str(x)).name)

    if "filename" in preds.columns and "filename" in labels.columns:
        labels = labels.set_index("filename").reindex(preds["filename"]).reset_index()

    label_cols = [c for c in preds.columns if c not in {"filename", "image", "study_id", "subject_id", "No Finding"}]

    lines = ["# Sanity summary", ""]
    for label in label_cols:
        gt_series = labels.get(label)
        if gt_series is None:
            gt_series = labels.get(f"y_true_{label}")
        if gt_series is None:
            continue
        gt_series = pd.to_numeric(gt_series, errors="coerce")
        certain = gt_series[gt_series.isin([0, 1])]
        coverage = len(certain)
        gt_pos = int((certain == 1).sum()) if coverage else 0
        preds_series = pd.to_numeric(preds[label], errors="coerce").fillna(0)
        all_zero = bool((preds_series == 0).all())
        status = "OK"
        if gt_pos > 0 and all_zero:
            status = "ALERT: all-zero predictions but positives exist"
        lines.append(
            f"{label}: coverage={coverage}, gt_pos={gt_pos}, all_zero_pred={all_zero} -> {status}"
        )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
