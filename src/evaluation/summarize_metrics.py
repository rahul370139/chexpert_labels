#!/usr/bin/env python3
"""
Summarize per-label metrics, computing macro over all labels and over labels
with nonzero positives in the given ground truth CSV.

Inputs:
- --metrics_csv: CSV with columns [label, precision, recall, f1]
- --labels_csv: ground truth CSV with label columns or y_true_<Label> columns
- --out_json: optional JSON to write macro summaries

Prints macro_all and macro_nonzero summaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def compute_pos_for_labels(labels_csv: Path, labels: list[str]) -> dict[str, int]:
    df = pd.read_csv(labels_csv)
    pos: dict[str, int] = {}
    for lab in labels:
        col = f"y_true_{lab}" if f"y_true_{lab}" in df.columns else lab
        if col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            pos[lab] = int(series.sum())
        else:
            pos[lab] = 0
    return pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    metrics = pd.read_csv(args.metrics_csv)
    label_list = metrics['label'].tolist()
    pos = compute_pos_for_labels(Path(args.labels_csv), label_list)
    # Filter to labels present in metrics
    metrics = metrics[metrics['label'].isin(pos.keys())].copy()
    macro_all = {
        "macro_precision": float(metrics['precision'].mean()),
        "macro_recall": float(metrics['recall'].mean()),
        "macro_f1": float(metrics['f1'].mean()),
    }
    nonzero_labels = [lab for lab, n in pos.items() if n > 0]
    metrics_nz = metrics[metrics['label'].isin(nonzero_labels)]
    macro_nz = {
        "macro_precision": float(metrics_nz['precision'].mean()),
        "macro_recall": float(metrics_nz['recall'].mean()),
        "macro_f1": float(metrics_nz['f1'].mean()),
        "labels": nonzero_labels,
    }
    print("macro_all:", macro_all)
    print("macro_nonzero:", macro_nz)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({"macro_all": macro_all, "macro_nonzero": macro_nz}, indent=2))
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
