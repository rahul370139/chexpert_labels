"""
Blend multiple probabilistic prediction tables into an averaged ensemble.

Example:
    python blend_probabilities.py \
        --predictions txr_calibrated.csv 0.6 \
        --predictions chex_calibrated.csv 0.4 \
        --out_csv ensemble_calibrated.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

CHEXPERT14 = [
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
    "No Finding",
]


def parse_prediction_arg(arg: str) -> Tuple[Path, float]:
    parts = arg.split(",")
    if len(parts) == 1:
        return Path(parts[0]), 1.0
    if len(parts) == 2:
        path, weight = parts
        return Path(path), float(weight)
    raise ValueError(f"Invalid prediction argument: {arg}")


def main():
    parser = argparse.ArgumentParser(description="Blend probabilistic predictions via weighted averaging")
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Prediction CSV and optional weight: path[,weight]. "
             "Provide multiple --predictions flags for each source.",
    )
    parser.add_argument("--score_prefix", default="y_cal_", help="Prefix for probability columns")
    parser.add_argument("--out_csv", default="ensemble_predictions.csv")
    args = parser.parse_args()

    sources: List[Tuple[Path, float]] = [parse_prediction_arg(p) for p in args.predictions]
    total_weight = sum(weight for _, weight in sources)
    if total_weight <= 0:
        raise ValueError("Total ensemble weight must be > 0.")

    base_df = None
    ensemble_scores = None

    for path, weight in sources:
        df = pd.read_csv(path)
        if base_df is None:
            base_df = df.copy()
            ensemble_scores = {label: df[f"{args.score_prefix}{label}"].astype(float).values * weight for label in CHEXPERT14 if f"{args.score_prefix}{label}" in df.columns}
        else:
            if "filename" in base_df.columns and "filename" in df.columns:
                if not (base_df["filename"] == df["filename"]).all():
                    raise ValueError("Prediction files do not align on filename column.")
            elif "image" in base_df.columns and "image" in df.columns:
                if not (base_df["image"] == df["image"]).all():
                    raise ValueError("Prediction files do not align on image column.")
            else:
                raise ValueError("Prediction files must share either 'filename' or 'image' column.")

            for label in CHEXPERT14:
                col = f"{args.score_prefix}{label}"
                if col not in df.columns or label not in ensemble_scores:
                    continue
                ensemble_scores[label] += df[col].astype(float).values * weight

    if base_df is None or ensemble_scores is None:
        raise ValueError("No predictions were loaded.")

    for label, scores in ensemble_scores.items():
        base_df[f"{args.score_prefix}{label}"] = scores / total_weight

    base_df.to_csv(args.out_csv, index=False)
    print(f"✅ Saved ensemble predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
