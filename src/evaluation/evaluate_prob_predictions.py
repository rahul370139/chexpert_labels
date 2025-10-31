"""
Evaluate probabilistic predictions against CheXpert ground truth using per-label thresholds.

Designed for calibrated probability tables (e.g., output from apply_label_calibrators.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

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

CHEXPERT14 = CHEXPERT13 + ["No Finding"]


def load_thresholds(thresholds_path: Path, labels: List[str]) -> Dict[str, float]:
    data = json.loads(thresholds_path.read_text())
    thresholds = {}
    for label in labels:
        if label not in data:
            raise KeyError(f"Threshold for '{label}' not found in {thresholds_path}.")
        thresholds[label] = float(data[label])
    return thresholds


def evaluate_predictions(
    predictions_csv: Path,
    ground_truth_csv: Path,
    thresholds_path: Path,
    score_prefix: str,
) -> pd.DataFrame:
    preds = pd.read_csv(predictions_csv)
    gt = pd.read_csv(ground_truth_csv)

    preds["filename"] = preds["image"].apply(lambda x: Path(x).name) if "image" in preds.columns else preds["filename"]
    gt["filename"] = gt["image"].apply(lambda x: Path(x).name)
    merged = pd.merge(preds, gt, on="filename", suffixes=("_pred", "_gt"))
    if merged.empty:
        raise ValueError("No overlapping images between predictions and ground truth.")

    thresholds = load_thresholds(thresholds_path, CHEXPERT13)

    rows = []
    all_y_true, all_y_pred = [], []

    for label in CHEXPERT13:
        score_col = f"{score_prefix}{label}"
        gt_col = f"{label}_gt"
        if score_col not in merged.columns:
            raise KeyError(f"Score column '{score_col}' missing in {predictions_csv}.")
        if gt_col not in merged.columns:
            alt = label
            if alt in merged.columns:
                gt_col = alt
            else:
                raise KeyError(f"Ground truth column '{gt_col}' missing in {ground_truth_csv}.")

        scores = merged[score_col].astype(float).values
        y_true = merged[gt_col].replace(-1, 0).astype(int).values
        y_pred = (scores >= thresholds[label]).astype(int)

        if len(np.unique(y_true)) < 2:
            # Skip labels with no positives/negatives in evaluation set
            continue

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        rows.append(
            {
                "label": label,
                "threshold": thresholds[label],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": acc,
                "positives": int(y_true.sum()),
            }
        )

        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())

    results = pd.DataFrame(rows)
    if not results.empty:
        macro_precision = results["precision"].mean()
        macro_recall = results["recall"].mean()
        macro_f1 = results["f1"].mean()
    else:
        macro_precision = macro_recall = macro_f1 = 0.0

    micro_precision = precision_score(all_y_true, all_y_pred, zero_division=0)
    micro_recall = recall_score(all_y_true, all_y_pred, zero_division=0)
    micro_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)

    # No Finding evaluation
    nf_score_col = f"{score_prefix}No Finding"
    if nf_score_col in merged.columns:
        no_finding_threshold = load_thresholds(thresholds_path, ["No Finding"]).get("No Finding", 0.5)
        nf_scores = merged[nf_score_col].astype(float).values
        nf_gt_col = "No Finding_gt" if "No Finding_gt" in merged.columns else "No Finding"
        nf_true = merged[nf_gt_col].replace(-1, 0).astype(int).values
        nf_pred = (nf_scores >= no_finding_threshold).astype(int)
        nf_precision = precision_score(nf_true, nf_pred, zero_division=0)
        nf_recall = recall_score(nf_true, nf_pred, zero_division=0)
        nf_f1 = f1_score(nf_true, nf_pred, zero_division=0)
    else:
        nf_precision = nf_recall = nf_f1 = float("nan")

    summary = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "no_finding_precision": nf_precision,
        "no_finding_recall": nf_recall,
        "no_finding_f1": nf_f1,
        "n_images": len(merged),
    }

    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate probabilistic CheXpert predictions")
    parser.add_argument("--predictions", required=True, help="CSV with calibrated probabilities")
    parser.add_argument("--ground_truth", required=True, help="Ground truth CSV")
    parser.add_argument("--thresholds", required=True, help="JSON file with per-label thresholds")
    parser.add_argument("--score_prefix", default="y_cal_", help="Prefix for probability columns (default: y_cal_)")
    parser.add_argument("--out_metrics", default=None, help="Optional CSV to write per-label metrics")
    args = parser.parse_args()

    per_label, summary = evaluate_predictions(
        predictions_csv=Path(args.predictions),
        ground_truth_csv=Path(args.ground_truth),
        thresholds_path=Path(args.thresholds),
        score_prefix=args.score_prefix,
    )

    if not per_label.empty:
        print("\n=== Per-label metrics ===")
        print(per_label.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    else:
        print("No labels had positive samples in the evaluation set.")

    print("\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")

    if args.out_metrics:
        per_label.to_csv(args.out_metrics, index=False)
        print(f"\nSaved per-label metrics to {args.out_metrics}")


if __name__ == "__main__":
    main()
