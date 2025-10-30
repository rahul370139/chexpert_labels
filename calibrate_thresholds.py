"""
Utility for sweeping per-label binary thresholds using stored CheXagent scores.

Usage:
    python calibrate_thresholds.py --pred smart_ensemble_1000.csv \
        --manifest data/evaluation_manifest_1000.csv \
        --out config/label_thresholds.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

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


def parse_binary_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    if "binary_outputs" not in predictions.columns:
        raise ValueError("Predictions file does not contain 'binary_outputs' column produced by smart_ensemble.py")

    binary_dicts = predictions["binary_outputs"].fillna("{}").apply(json.loads)
    for disease in CHEXPERT13:
        predictions[f"{disease}_score"] = binary_dicts.apply(
            lambda record: record.get(disease, {}).get("score", np.nan)
        )
    return predictions


def sweep_thresholds(merged: pd.DataFrame, min_support: int = 5) -> Dict[str, float]:
    tau_grid = np.linspace(0.1, 0.9, 17)
    thresholds: Dict[str, float] = {}

    for disease in CHEXPERT13:
        score_col = f"{disease}_score"
        if score_col not in merged.columns:
            continue

        scores = merged[score_col].values
        mask = ~np.isnan(scores)
        if mask.sum() == 0:
            continue

        y_true = merged.loc[mask, f"{disease}_gt"].values
        if y_true.sum() < min_support:
            thresholds[disease] = float(np.nan)
            continue

        best_tau = 0.5
        best_f1 = 0.0

        for tau in tau_grid:
            y_pred = (scores[mask] >= tau).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1 + 1e-6:
                best_f1 = f1
                best_tau = tau

        thresholds[disease] = round(best_tau, 3)

    return thresholds


def apply_thresholds(merged: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, float]:
    adjusted = merged.copy()
    for disease, tau in thresholds.items():
        if np.isnan(tau):
            continue
        score_col = f"{disease}_score"
        pred_col = f"{disease}_pred"
        if score_col in adjusted.columns and pred_col in adjusted.columns:
            adjusted[pred_col] = (adjusted[score_col] >= tau).astype(int)

    adjusted["No Finding_pred"] = (
        adjusted[[f"{label}_pred" for label in CHEXPERT13]].sum(axis=1) == 0
    ).astype(int)

    labels = CHEXPERT13 + ["No Finding"]
    y_true = adjusted[[f"{label}_gt" for label in labels]].values.ravel()
    y_pred = adjusted[[f"{label}_pred" if label != "No Finding" else "No Finding_pred" for label in labels]].values.ravel()

    micro_precision = precision_score(y_true, y_pred, zero_division=0)
    micro_recall = recall_score(y_true, y_pred, zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = np.mean(
        [
            f1_score(
                adjusted[f"{label}_gt"],
                adjusted[f"{label}_pred"] if label != "No Finding" else adjusted["No Finding_pred"],
                zero_division=0,
            )
            for label in labels
        ]
    )

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate per-label thresholds for CheXagent hybrid ensemble.")
    parser.add_argument("--pred", type=Path, required=True, help="CSV output from smart_ensemble.py containing binary_outputs.")
    parser.add_argument("--manifest", type=Path, required=True, help="Ground truth manifest CSV.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the threshold JSON file.")
    parser.add_argument("--min_support", type=int, default=5, help="Minimum positive examples required to tune a label.")
    args = parser.parse_args()

    predictions = parse_binary_scores(pd.read_csv(args.pred))
    ground_truth = pd.read_csv(args.manifest)

    predictions["filename"] = predictions["image"].apply(lambda x: Path(x).name)
    ground_truth["filename"] = ground_truth["image"].apply(lambda x: Path(x).name)

    merged = predictions.merge(ground_truth, on="filename", suffixes=("_pred", "_gt"))

    thresholds = sweep_thresholds(merged, min_support=args.min_support)
    metrics = apply_thresholds(merged, thresholds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(thresholds, indent=2))

    print("Saved thresholds to", args.out)
    print(
        "Projected micro F1={micro_f1:.3f} (P={micro_precision:.3f}, R={micro_recall:.3f}), macro F1={macro_f1:.3f}".format(
            **metrics
        )
    )


if __name__ == "__main__":
    main()
