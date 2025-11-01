#!/usr/bin/env python3
"""
Blend calibrated probability streams, meta-calibrate, threshold, gate, and evaluate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.labels import get_label_list
from src.calibration.platt_utils import apply_platt_to_scores
from src.utils.align_by_filename import align_to_reference, ensure_filename_column


def parse_source(arg: str) -> Tuple[str, Path]:
    parts = arg.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid --probs_csv argument '{arg}'. Expected format path,name.")
    path = Path(parts[0].strip())
    name = parts[1].strip()
    return name, path


def load_sources(source_args: List[str], filenames: List[str]) -> Dict[str, pd.DataFrame]:
    sources: Dict[str, pd.DataFrame] = {}
    for arg in source_args:
        name, path = parse_source(arg)
        df = pd.read_csv(path)
        aligned = align_to_reference(filenames, df)
        sources[name] = aligned
    return sources


def blend_probabilities(
    labels: Iterable[str],
    sources: Dict[str, pd.DataFrame],
    weights: Dict[str, Dict[str, float]],
    score_prefix: str,
) -> pd.DataFrame:
    blended = pd.DataFrame({"filename": next(iter(sources.values()))["filename"]})
    for label in labels:
        label_weights = weights.get(label, {})
        scores = np.zeros(len(blended), dtype=np.float32)
        for source_name, weight in label_weights.items():
            if weight == 0:
                continue
            df = sources.get(source_name)
            if df is None:
                continue
            col = f"{score_prefix}{label}"
            if col not in df.columns:
                continue
            scores += weight * df[col].astype(float).fillna(0.0).to_numpy()
        blended[f"y_pred_{label}"] = scores
    return blended


def load_meta_params(path: Path) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    return json.loads(path.read_text())


def apply_meta_calibration(
    blended: pd.DataFrame,
    labels: Iterable[str],
    params: Dict[str, Dict[str, float]],
) -> None:
    if not params:
        for label in labels:
            blended[f"y_cal_{label}"] = blended[f"y_pred_{label}"]
        return
    for label in labels:
        coeffs = params.get(label)
        pred_col = f"y_pred_{label}"
        if pred_col not in blended.columns:
            continue
        scores = blended[pred_col].astype(float).to_numpy()
        if not coeffs:
            blended[f"y_cal_{label}"] = scores
            continue
        calibrated = apply_platt_to_scores(scores, coeffs["a"], coeffs["b"])
        blended[f"y_cal_{label}"] = calibrated


def load_thresholds(path: Path) -> Dict[str, float]:
    return json.loads(path.read_text())


def parse_metadata(metadata_csv: Path | None, filenames: List[str]) -> Dict[str, Dict[str, Dict]]:
    if metadata_csv is None:
        return {}
    df = pd.read_csv(metadata_csv)
    df = ensure_filename_column(df)
    # Drop duplicate filenames, keep first occurrence
    df = df.drop_duplicates(subset=["filename"], keep="first")
    aligned = df.set_index("filename").reindex(filenames)
    meta: Dict[str, Dict[str, Dict]] = {}
    for filename, row in aligned.iterrows():
        entry: Dict[str, Dict] = {}
        if pd.notna(row.get("binary_outputs")):
            try:
                entry["binary"] = json.loads(row["binary_outputs"])
            except Exception:
                entry["binary"] = {}
        else:
            entry["binary"] = {}
        if pd.notna(row.get("di_outputs")):
            try:
                entry["di"] = json.loads(row["di_outputs"])
            except Exception:
                entry["di"] = {}
        else:
            entry["di"] = {}
        meta[filename] = entry
    return meta


def di_positive(di_entry: Dict, min_strength: float) -> bool:
    if not di_entry:
        return False
    if di_entry.get("negated"):
        return False
    if di_entry.get("uncertain"):
        return False
    strength_val = di_entry.get("strength")
    if strength_val is None:
        strength = 1.0 if di_entry.get("mentioned") else 0.0
    else:
        strength = float(strength_val)
    mentioned = bool(di_entry.get("mentioned"))
    return mentioned and strength >= min_strength


def di_negated(di_entry: Dict) -> bool:
    if not di_entry:
        return False
    return bool(di_entry.get("negated"))


def apply_gating(
    probs: pd.DataFrame,
    labels_df: pd.DataFrame,
    thresholds: Dict[str, float],
    gating_config: Path | None,
    metadata: Dict[str, Dict[str, Dict]],
    score_prefix: str,
) -> pd.DataFrame:
    labels = [label for label in thresholds.keys() if f"{score_prefix}{label}" in probs.columns]
    preds = probs[["filename"]].copy()
    preds["No Finding"] = 0

    if not gating_config:
        for label in labels:
            thresh = thresholds.get(label, 0.5)
            preds[label] = (probs[f"{score_prefix}{label}"] >= thresh).astype(int)
        preds["No Finding"] = (preds[labels].sum(axis=1) == 0).astype(int)
        return preds

    config = json.loads(Path(gating_config).read_text())
    hard = set(config.get("hard_labels", []))
    easy = set(config.get("easy_labels", []))
    rules = config.get("rules", {})
    hard_di_min = float(rules.get("di_min_hard", rules.get("hard_di_min", 0.75)))
    easy_di_min = float(rules.get("di_min_easy", rules.get("easy_di_min", 0.6)))
    rescue_margin = float(rules.get("rescue_margin", 0.05))
    rescue_easy = bool(rules.get("below_threshold_rescue_easy", True))
    rescue_hard = bool(rules.get("below_threshold_rescue_hard", False))
    hard_require_di_if_mentioned = bool(rules.get("hard_require_di_if_mentioned", True))
    high_prob_override = float(rules.get("high_prob_override", 0.80))
    high_prob_bypass_di_easy = bool(rules.get("high_prob_bypass_di_easy", False))
    consistency = rules.get("consistency", {})

    for idx, row in probs.iterrows():
        filename = row["filename"]
        meta_entry = metadata.get(filename, {})
        di_map = meta_entry.get("di", {})
        row_preds: Dict[str, int] = {}

        for label in labels:
            prob = float(row[f"{score_prefix}{label}"])
            if math.isnan(prob):
                prob = 0.0
            threshold = float(thresholds.get(label, 0.5))
            if math.isnan(threshold):
                threshold = 0.5
            decision = int(prob >= threshold)
            di_entry = di_map.get(label, {})

            if di_negated(di_entry):
                decision = 0
            elif label in hard:
                mention_present = bool(di_entry.get("mentioned"))
                di_pos = di_positive(di_entry, hard_di_min)
                if mention_present and not di_pos and hard_require_di_if_mentioned and decision == 1:
                    decision = 0
                if di_pos and ((decision == 0 and rescue_hard) or prob >= threshold - rescue_margin):
                    decision = 1
            elif label in easy:
                # High-prob override: bypass DI for easy labels if prob >= high_prob_override
                if high_prob_bypass_di_easy and prob >= high_prob_override:
                    # Allow high-prob predictions to bypass DI check
                    if prob >= threshold:
                        decision = 1
                    else:
                        decision = 0
                else:
                    # Normal DI gating
                    if decision == 0 and rescue_easy and di_positive(di_entry, easy_di_min) and prob >= threshold - rescue_margin:
                        decision = 1
                    elif decision == 1 and not di_positive(di_entry, easy_di_min) and di_entry and di_entry.get("uncertain"):
                        decision = 0
            
            # Consistency check (e.g., Pneumonia requires Lung Opacity or Consolidation)
            if consistency and label == "Pneumonia" and decision == 1:
                apply_below = float(consistency.get("apply_if_prob_below", 1.0))
                if prob < apply_below:
                    requires = consistency.get("Pneumonia_requires_one_of", [])
                    has_required = False
                    for req_label in requires:
                        req_prob = float(row.get(f"{score_prefix}{req_label}", 0.0))
                        req_thresh = thresholds.get(req_label, 0.5)
                        if req_prob >= req_thresh:
                            has_required = True
                            break
                    if not has_required:
                        decision = 0

            row_preds[label] = decision

        preds.loc[idx, labels] = pd.Series(row_preds)
        preds.loc[idx, "No Finding"] = 1 if sum(row_preds.values()) == 0 else 0

    return preds


def compute_metrics(
    labels: Iterable[str],
    preds_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    y_true_cols = []
    for label in labels:
        if f"y_true_{label}" in labels_df.columns:
            y_true_cols.append(f"y_true_{label}")
        elif label in labels_df.columns:
            y_true_cols.append(label)
        else:
            raise ValueError(f"Ground truth missing for label '{label}'.")

    y_true = labels_df[y_true_cols].to_numpy(dtype=int)
    y_pred = preds_df[list(labels)].to_numpy(dtype=int)

    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)
    micro_f = f1_score(y_true, y_pred, average="micro", zero_division=0)

    agg = {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f,
    }

    per_label = []
    for idx, label in enumerate(labels):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        per_label.append(
            {
                "label": label,
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
            }
        )
    return agg, per_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate blended CheXpert probabilities on held-out test split.")
    parser.add_argument(
        "--probs_csv",
        action="append",
        required=True,
        help="Prediction CSV with alias: path,name (repeat per source).",
    )
    parser.add_argument("--blend_weights_json", required=True)
    parser.add_argument("--meta_platt_json", required=False)
    parser.add_argument("--thresholds_json", required=True)
    parser.add_argument("--test_labels_csv", required=True)
    parser.add_argument("--labels", default="chexpert13")
    parser.add_argument("--exclude_labels", nargs="*", default=[], help="Labels to exclude from gating and metrics")
    parser.add_argument("--score_prefix", default="y_cal_")
    parser.add_argument("--meta_prefix", default="y_cal_")
    parser.add_argument("--gating_config", default=None)
    parser.add_argument("--metadata_csv", default=None, help="CheXagent metadata (binary_outputs/di_outputs).")
    parser.add_argument("--out_probs_csv", default="outputs/final/test_probs.csv")
    parser.add_argument("--out_preds_csv", default="outputs/final/test_preds.csv")
    parser.add_argument("--out_metrics_csv", default="outputs/final/test_metrics.csv")
    args = parser.parse_args()

    labels_df = ensure_filename_column(pd.read_csv(args.test_labels_csv))
    filenames = labels_df["filename"].tolist()
    sources = load_sources(args.probs_csv, filenames)

    weights = json.loads(Path(args.blend_weights_json).read_text())
    labels = get_label_list(args.labels)
    excluded = set(args.exclude_labels)
    if excluded:
        labels = [l for l in labels if l not in excluded]
    blended = blend_probabilities(labels, sources, weights, score_prefix=args.score_prefix)

    meta_params = load_meta_params(Path(args.meta_platt_json)) if args.meta_platt_json else {}
    apply_meta_calibration(blended, labels, meta_params)

    thresholds = load_thresholds(Path(args.thresholds_json))
    # Drop thresholds for excluded labels to avoid key errors
    if excluded:
        thresholds = {k: v for k, v in thresholds.items() if k not in excluded}
    metadata = parse_metadata(Path(args.metadata_csv) if args.metadata_csv else None, filenames)
    preds = apply_gating(
        blended,
        labels_df,
        thresholds,
        Path(args.gating_config) if args.gating_config else None,
        metadata,
        score_prefix=args.meta_prefix,
    )

    probs_out = Path(args.out_probs_csv)
    preds_out = Path(args.out_preds_csv)
    metrics_out = Path(args.out_metrics_csv)
    probs_out.parent.mkdir(parents=True, exist_ok=True)
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    blended.to_csv(probs_out, index=False)
    preds.to_csv(preds_out, index=False)

    agg_metrics, per_label = compute_metrics(labels, preds, labels_df)
    metrics_df = pd.DataFrame(per_label)
    metrics_df.to_csv(metrics_out, index=False)
    agg_path = metrics_out.with_suffix(".macro_micro.json")
    agg_path.write_text(json.dumps(agg_metrics, indent=2))

    print(f"✅ Probabilities saved to {probs_out}")
    print(f"✅ Predictions saved to {preds_out}")
    print(f"✅ Metrics saved to {metrics_out}")
    print(f"Macro/Micro: {agg_metrics}")


if __name__ == "__main__":
    main()
