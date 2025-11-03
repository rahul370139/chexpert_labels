#!/usr/bin/env python3
"""
Per-label weight search to blend multiple calibrated probability streams.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.labels import get_label_list
from src.utils.align_by_filename import align_to_reference, ensure_filename_column


@dataclass
class SourceSpec:
    name: str
    path: Path


def parse_source(arg: str) -> SourceSpec:
    parts = arg.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid --probs_csv argument '{arg}'. Expected format path,name.")
    path = Path(parts[0].strip())
    name = parts[1].strip()
    return SourceSpec(name=name, path=path)


def generate_weight_grid(n_sources: int, step: float) -> List[Tuple[float, ...]]:
    grid_values = [round(i * step, 10) for i in range(int(1 / step) + 1)]
    combos = []
    for combo in itertools.product(grid_values, repeat=n_sources):
        if sum(combo) == 0:
            continue
        norm = sum(combo)
        combos.append(tuple(value / norm for value in combo))
    return combos


def best_fbeta(y_true: np.ndarray, y_scores: np.ndarray, beta: float) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    best_f, best_threshold = -1.0, 0.5
    beta_sq = beta * beta
    for idx in range(1, len(precision)):
        p = precision[idx]
        r = recall[idx]
        if p == 0 and r == 0:
            continue
        f = (1 + beta_sq) * p * r / (beta_sq * p + r + 1e-12)
        if f > best_f:
            best_f = f
            best_threshold = thresholds[idx - 1]
    return best_f, best_threshold


def best_recall_with_precision(y_true: np.ndarray, y_scores: np.ndarray, min_precision: float) -> Tuple[float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    best_recall, best_precision, best_threshold = -1.0, 0.0, 0.5
    for idx in range(1, len(precision)):
        p = precision[idx]
        r = recall[idx]
        if p >= min_precision and r > best_recall:
            best_recall, best_precision = r, p
            best_threshold = thresholds[idx - 1]
    if best_recall < 0:
        return best_fbeta(y_true, y_scores, beta=0.3)[0], min_precision, best_threshold
    return best_recall, best_precision, best_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search blend weights per label.")
    parser.add_argument(
        "--probs_csv",
        action="append",
        required=True,
        help="Prediction CSV and nickname: path,name (repeat per source).",
    )
    parser.add_argument("--labels_csv", required=True, help="CSV with ground-truth labels.")
    parser.add_argument("--labels", default="chexpert13")
    parser.add_argument("--score_prefix", default="y_pred_", help="Probability column prefix in input CSVs.")
    parser.add_argument("--metric", default="fbeta", choices=["fbeta", "min_precision"])
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--min_precision", type=float, default=0.6)
    parser.add_argument("--weight_step", type=float, default=0.1, help="Grid resolution for weights.")
    parser.add_argument("--out_weights_json", default="outputs/blend/blend_weights.json")
    parser.add_argument("--out_blended_csv", default="outputs/blend/train_blended.csv")
    parser.add_argument("--align", choices=["reference", "intersection"], default="reference",
                        help="How to align sources to labels: reference enforces full coverage; intersection uses common filenames across all sources.")
    args = parser.parse_args()

    sources = [parse_source(spec) for spec in args.probs_csv]
    if len(sources) < 1:
        raise ValueError("At least one probability source is required.")

    label_list = get_label_list(args.labels)
    labels_df = ensure_filename_column(pd.read_csv(args.labels_csv))
    reference_filenames = labels_df["filename"].tolist()

    source_frames: Dict[str, pd.DataFrame] = {}
    for spec in sources:
        df = pd.read_csv(spec.path)
        if args.align == "reference":
            aligned = align_to_reference(reference_filenames, df)
        else:
            # Intersection mode: compute common filenames across all sources
            df = ensure_filename_column(df)
            source_frames[spec.name] = df  # temporarily store raw
            continue
        source_frames[spec.name] = aligned

    if args.align == "intersection":
        # Compute common filenames across labels_df and all sources
        common = set(reference_filenames)
        for spec in sources:
            df = source_frames[spec.name]
            fns = set(df["filename"].tolist())
            common &= fns
        # Reduce to intersection
        common_list = sorted(common)
        dropped = len(reference_filenames) - len(common_list)
        if dropped > 0:
            print(f"⚠️  Intersection alignment: dropping {dropped} train rows without predictions across all sources.")
        reference_filenames = common_list
        # Filter labels_df to intersection
        labels_df = labels_df[labels_df["filename"].isin(reference_filenames)].reset_index(drop=True)
        # Now align each frame strictly to the intersection
        for spec in sources:
            df = source_frames[spec.name]
            aligned = df.set_index("filename").reindex(reference_filenames).reset_index()
            source_frames[spec.name] = aligned

    # Prepare base output with filenames and ground truth for convenience
    blended = labels_df[["filename"]].copy()
    for label in label_list:
        if label in labels_df.columns:
            blended[f"y_true_{label}"] = labels_df[label].astype(int)
        elif f"y_true_{label}" in labels_df.columns:
            blended[f"y_true_{label}"] = labels_df[f"y_true_{label}"].astype(int)

    weight_grid = generate_weight_grid(len(sources), args.weight_step)
    chosen_weights: Dict[str, Dict[str, float]] = {}

    for label in label_list:
        available_idx = []
        for idx, spec in enumerate(sources):
            col = f"{args.score_prefix}{label}"
            df = source_frames[spec.name]
            if col in df.columns and not df[col].isna().all():
                available_idx.append(idx)
        if not available_idx:
            print(f"⚠️  No sources provide label '{label}'. Skipping.")
            continue

        y_true_col = f"y_true_{label}"
        if y_true_col not in blended.columns:
            print(f"⚠️  Ground truth missing for '{label}'. Skipping weight search.")
            continue
        # Convert to binary (handle -1 uncertain as 0)
        y_true = blended[y_true_col].replace(-1, 0).astype(int)
        # Ensure binary (only 0 and 1)
        y_true = np.clip(y_true, 0, 1).astype(int)

        best_score = -np.inf
        best_vector = None

        for weights in weight_grid:
            # Ignore weights for sources without this label
            skip = False
            for idx, weight in enumerate(weights):
                col_name = f"{args.score_prefix}{label}"
                if weight > 0 and col_name not in source_frames[sources[idx].name].columns:
                    skip = True
                    break
            if skip:
                continue

            combined = np.zeros_like(y_true, dtype=np.float32)
            for idx, spec in enumerate(sources):
                weight = weights[idx]
                if weight == 0:
                    continue
                col = f"{args.score_prefix}{label}"
                if col not in source_frames[spec.name].columns:
                    continue
                combined += weight * source_frames[spec.name][col].to_numpy(dtype=np.float32)

            if args.metric == "fbeta":
                score, _ = best_fbeta(y_true, combined, args.beta)
            else:
                score, _, _ = best_recall_with_precision(y_true, combined, args.min_precision)

            if score > best_score:
                best_score = score
                best_vector = weights

        if best_vector is None:
            print(f"⚠️  Could not find valid weights for label '{label}', defaulting to equal weights.")
            best_vector = tuple(
                1.0 / len(available_idx) if idx in available_idx else 0.0 for idx in range(len(sources))
            )
        chosen_weights[label] = {spec.name: float(best_vector[idx]) for idx, spec in enumerate(sources)}

        # Apply chosen weights to create blended probabilities
        blended[f"y_pred_{label}"] = 0.0
        for idx, spec in enumerate(sources):
            weight = chosen_weights[label].get(spec.name, 0.0)
            if weight == 0.0:
                continue
            col = f"{args.score_prefix}{label}"
            if col not in source_frames[spec.name].columns:
                continue
            blended[f"y_pred_{label}"] += weight * source_frames[spec.name][col].to_numpy(dtype=np.float32)

    out_csv = Path(args.out_blended_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(out_csv, index=False)
    Path(args.out_weights_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_weights_json).write_text(json.dumps(chosen_weights, indent=2))

    print(f"✅ Saved blended probabilities to {out_csv}")
    print(f"✅ Saved blend weights to {args.out_weights_json}")


if __name__ == "__main__":
    main()
