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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.labels import get_label_list
from src.calibration.platt_utils import apply_platt_to_scores
from src.utils.align_by_filename import align_to_reference, ensure_filename_column

INVERTED_PROB_LABELS: set[str] = set()


def parse_source(arg: str) -> Tuple[str, Path]:
    parts = arg.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid --probs_csv argument '{arg}'. Expected format path,name.")
    path = Path(parts[0].strip())
    name = parts[1].strip()
    return name, path


def load_sources(source_args: List[str], filenames: List[str], align: str = "reference") -> Dict[str, pd.DataFrame]:
    sources: Dict[str, pd.DataFrame] = {}
    raw: Dict[str, pd.DataFrame] = {}
    for arg in source_args:
        name, path = parse_source(arg)
        df = pd.read_csv(path)
        if align == "reference":
            aligned = align_to_reference(filenames, df)
            sources[name] = aligned
        else:
            raw[name] = ensure_filename_column(df)
    if align == "intersection":
        common = set(filenames)
        for name, df in raw.items():
            common &= set(df["filename"].tolist())
        common_list = sorted(common)
        for name, df in raw.items():
            aligned = df.set_index("filename").reindex(common_list).reset_index()
            sources[name] = aligned
        # Replace filenames with intersection for downstream ops
        return {"__filenames__": pd.DataFrame({"filename": common_list}), **sources}
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
        contribution = 0.0
        if not label_weights:
            label_weights = {}
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
            contribution += abs(weight)
        if contribution == 0.0:
            for source_name, df in sources.items():
                if source_name == "__filenames__":
                    continue
                col = f"{score_prefix}{label}"
                if col in df.columns:
                    scores = df[col].astype(float).fillna(0.0).to_numpy()
                    break
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
    import numpy as np
    if not params:
        for label in labels:
            pred_col = f"y_pred_{label}"
            if pred_col in blended.columns:
                blended[f"y_cal_{label}"] = blended[pred_col]
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
        method = coeffs.get("method", "platt")
        if method == "isotonic" and "x" in coeffs and "y" in coeffs:
            x = np.array(coeffs["x"], dtype=float)
            y = np.array(coeffs["y"], dtype=float)
            blended[f"y_cal_{label}"] = np.interp(scores, x, y).astype("float32")
        else:
            a = float(coeffs.get("a", 1.0))
            b = float(coeffs.get("b", 0.0))
            blended[f"y_cal_{label}"] = apply_platt_to_scores(scores, a, b)


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
    force_zero_labels: Iterable[str] | None = None,
) -> pd.DataFrame:
    labels = [label for label in thresholds.keys() if f"{score_prefix}{label}" in probs.columns]
    if not labels:
        # Fallback: infer labels from available score columns
        inferred = []
        prefix = f"{score_prefix}"
        for col in probs.columns:
            if col.startswith(prefix):
                inferred.append(col[len(prefix):])
        labels = inferred
    preds = probs[["filename"]].copy()
    preds["No Finding"] = 0
    force_zero = set(force_zero_labels or [])

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
    per_label_rescue_margin = rules.get("per_label_rescue_margin", {})
    per_label_di_min = rules.get("per_label_di_min", {})
    per_label_prob_floor = rules.get("per_label_prob_floor", {})
    rescue_easy = bool(rules.get("below_threshold_rescue_easy", True))
    rescue_hard = bool(rules.get("below_threshold_rescue_hard", False))
    hard_require_di_if_mentioned = bool(rules.get("hard_require_di_if_mentioned", True))
    high_prob_override = float(rules.get("high_prob_override", 0.80))
    high_prob_bypass_di_easy = bool(rules.get("high_prob_bypass_di_easy", False))
    high_prob_bypass_labels = set(rules.get("high_prob_bypass_labels", []))
    consistency = rules.get("consistency", {})
    no_rescue_labels = set(rules.get("no_rescue_labels", []))
    no_uncertain_veto_labels = set(rules.get("no_uncertain_veto_labels", []))

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
            
            # Start conservative: only predict positive if prob clearly exceeds threshold
            decision = int(prob >= threshold)
            di_entry = di_map.get(label, {})

            # hard override: never predict these labels
            if label in force_zero:
                decision = 0
            elif di_negated(di_entry):
                decision = 0
            elif label in hard:
                mention_present = bool(di_entry.get("mentioned")) if di_entry else False
                di_pos = di_positive(di_entry, hard_di_min)
                # For hard labels: require DI confirmation ONLY if mentioned in text
                # If not mentioned, allow prediction if prob >= threshold (model confidence)
                if decision == 1:
                    # If DI mentions the label but strength is weak, require stronger evidence
                    if mention_present and not di_pos and hard_require_di_if_mentioned:
                        decision = 0
                    # Don't block predictions when DI is not mentioned - trust the model if prob >= threshold
                # Use per-label rescue margin if available, otherwise use default
                label_rescue_margin = float(per_label_rescue_margin.get(label, rescue_margin))
                if label not in no_rescue_labels and di_pos and ((decision == 0 and rescue_hard) or prob >= threshold - label_rescue_margin):
                    decision = 1
            elif label in easy:
                # High-prob override: bypass DI for easy labels if prob >= high_prob_override
                # Check if this label is in the bypass list (if list is non-empty), or apply to all if list is empty
                use_bypass = high_prob_bypass_di_easy and (
                    len(high_prob_bypass_labels) == 0 or label in high_prob_bypass_labels
                )
                if use_bypass and prob >= high_prob_override:
                    # Allow high-prob predictions to bypass DI check
                    if prob >= threshold:
                        decision = 1
                    else:
                        decision = 0
                else:
                    # Normal DI gating
                    # Use per-label rescue margin if available, otherwise use default
                    label_rescue_margin = float(per_label_rescue_margin.get(label, rescue_margin))
                    # Use per-label DI min threshold if available, otherwise use default
                    rescue_di_min = float(per_label_di_min.get(label, easy_di_min))
                    
                    # Pleural Other: rescue only when DI strength >= 0.4 OR prob >= prob_floor (0.01)
                    if label == "Pleural Other":
                        prob_floor_val = float(per_label_prob_floor.get(label, 0.0))
                        di_pos_for_rescue = di_positive(di_entry, rescue_di_min)
                        # Rescue if: (DI strong >= 0.4 AND prob >= floor) OR (prob >= max(floor, threshold))
                        if decision == 0 and rescue_easy and label not in no_rescue_labels:
                            effective_threshold = max(prob_floor_val, threshold)
                            if (di_pos_for_rescue and prob >= prob_floor_val) or (prob >= effective_threshold):
                                decision = 1
                        # Also allow if prob >= threshold (normal case)
                        elif prob >= threshold:
                            decision = 1
                    else:
                        # Support Devices: More aggressive rescue since devices are clearly visible
                        if label == "Support Devices":
                            # For Support Devices: Very aggressive rescue - devices are obvious in images
                            mentioned = bool(di_entry.get("mentioned", False)) if di_entry else False
                            di_strength = float(di_entry.get("strength", 0.0) or 0.0) if di_entry else 0.0
                            # Rescue if: (1) DI mentions it with any strength AND prob > 0, OR (2) prob >= 0.01 regardless of DI
                            if decision == 0 and rescue_easy and label not in no_rescue_labels:
                                if (mentioned and prob >= 0.005) or (prob >= 0.01) or (di_strength >= 0.2 and prob >= threshold - 0.2):
                                    decision = 1
                        else:
                            # Normal rescue logic for other labels
                            if decision == 0 and rescue_easy and label not in no_rescue_labels:
                                di_pos_for_rescue = di_positive(di_entry, rescue_di_min)
                                if di_pos_for_rescue and prob >= threshold - label_rescue_margin:
                                    decision = 1
                    
                    # Uncertain veto: flip positive predictions to 0 if DI is uncertain and not strong enough
                    # Skip this veto for labels in no_uncertain_veto_labels
                    if decision == 1 and label not in no_uncertain_veto_labels and not di_positive(di_entry, easy_di_min) and di_entry and di_entry.get("uncertain"):
                        decision = 0
            
            # Label-specific overrides for hard labels where DI mention should rescue low probabilities
            # Only apply if di_entry exists and is not negated
            mentioned = False
            strength = 0.0
            if di_entry:
                mentioned = bool(di_entry.get("mentioned", False))
                strength = float(di_entry.get("strength", 0.0) or 0.0)
                
                # Pleural Other: handled in easy label section with prob floor
                # No additional override needed here
                
                if label == "Lung Lesion" and mentioned and not di_entry.get("uncertain", False):
                    # Lung Lesion override: rescue if DI is strong (>=0.5) AND prob is reasonable
                    # This prevents always predicting 1 but allows DI to rescue
                    if strength >= 0.5 and prob >= max(0.35, threshold * 0.8):
                        decision = 1
                
                # Fracture: Hard label but model probabilities are good (mean 0.506)
                # Allow if prob >= threshold and either DI mentions it or prob is high enough
                if label == "Fracture":
                    if prob >= threshold:
                        # If prob is clearly above threshold, trust it
                        if prob >= 0.45:
                            decision = 1
                        # If prob is near threshold, require DI confirmation
                        elif mentioned and strength >= 0.4:
                            decision = 1
            
            # Note: Support Devices special override removed - let normal gating logic handle it
            # This allows No Finding to be predicted correctly when all labels are 0

            # Consistency check (e.g., Pneumonia requires Lung Opacity or Consolidation)
            # Only apply this check when probability is uncertain (between threshold and high confidence)
            # Don't block high-confidence predictions even if supporting evidence is weak
            if consistency and label == "Pneumonia" and decision == 1:
                apply_below = float(consistency.get("apply_if_prob_below", 1.0))
                # Only apply consistency check for medium-confidence predictions
                # High confidence (>= 0.70) or strong DI can bypass this check
                if prob < apply_below and prob < 0.65:
                    requires = consistency.get("Pneumonia_requires_one_of", [])
                    has_required = False
                    for req_label in requires:
                        req_prob = float(row.get(f"{score_prefix}{req_label}", 0.0))
                        req_thresh = thresholds.get(req_label, 0.5)
                        if req_prob >= req_thresh:
                            has_required = True
                            break
                    # Also check if DI mentions pneumonia strongly
                    di_strong = di_positive(di_entry, 0.5) if di_entry else False
                    if not has_required and not di_strong:
                        decision = 0

            # Hyper-positive labels guard: apply FINAL check to prevent over-prediction
            # These labels (Lung Opacity, Atelectasis) tend to fire on everything
            # Fracture removed: has good model probabilities and needs better recall
            # Apply this AFTER all DI/rescue logic to ensure it's the final gate
            hyper_positive_labels = {"Lung Opacity", "Atelectasis"}
            if label in hyper_positive_labels and decision == 1:
                # For hyper-positive labels, require prob >= 0.40 for low-threshold labels
                # This prevents over-prediction when model probabilities are systematically high
                if threshold < 0.4:
                    min_prob_required = max(0.40, threshold + 0.15)
                    if prob < min_prob_required:
                        decision = 0
                else:
                    # For higher thresholds, use moderate offset
                    if prob < threshold + 0.08:
                        decision = 0

            row_preds[label] = decision

        # Apply force-zero after per-label loop to be safe
        for lz in force_zero:
            row_preds[lz] = 0
        preds.loc[idx, labels] = pd.Series(row_preds)
        preds.loc[idx, "No Finding"] = 1 if sum(row_preds.values()) == 0 else 0

    return preds


def compute_metrics(
    labels: Iterable[str],
    preds_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    thresholds: Dict[str, float],
    eval_mode: str,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    labels = list(labels)
    present_pred_labels = [l for l in labels if l in preds_df.columns]
    if not present_pred_labels:
        raise ValueError("No predicted label columns present in preds_df.")

    results: List[Dict[str, float]] = []
    macro_precisions: List[float] = []
    macro_recalls: List[float] = []
    macro_f1s: List[float] = []
    macro_accuracies: List[float] = []
    gt_column_map: Dict[str, str] = {}
    base_label_set = set(present_pred_labels)
    micro_tp = micro_fp = micro_fn = 0

    for label in present_pred_labels:
        if f"y_true_{label}" in labels_df.columns:
            gt_series = labels_df[f"y_true_{label}"]
            gt_column_map[label] = f"y_true_{label}"
        elif label in labels_df.columns:
            gt_series = labels_df[label]
            gt_column_map[label] = label
        else:
            continue

        pred_series = preds_df[label]

        if eval_mode == "binary":
            gt_processed = gt_series.replace(-1, 0).fillna(0).astype(int)
            mask = gt_processed.isin([0, 1])
        else:  # certain_only
            mask = gt_series.isin([0, 1])
            gt_processed = gt_series[mask].astype(int)

        preds_filtered = pred_series[mask].fillna(0).astype(int)
        coverage = int(mask.sum())
        total = int(len(gt_series))
        gt_pos = int((gt_processed == 1).sum()) if coverage else 0
        gt_neg = int((gt_processed == 0).sum()) if coverage else 0
        pred_pos = int((preds_filtered == 1).sum()) if coverage else 0

        if coverage == 0:
            precision = recall = f1 = accuracy = float("nan")
            tp = fp = fn = tn = 0
        else:
            tp = int(((gt_processed == 1) & (preds_filtered == 1)).sum())
            fp = int(((gt_processed == 0) & (preds_filtered == 1)).sum())
            fn = int(((gt_processed == 1) & (preds_filtered == 0)).sum())
            tn = int(((gt_processed == 0) & (preds_filtered == 0)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if gt_pos == 0:
                f1 = float("nan")
            else:
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Calculate accuracy for this label (only on certain samples)
            accuracy = (tp + tn) / coverage if coverage > 0 else float("nan")

        if coverage > 0:
            micro_tp += tp
            micro_fp += fp
            micro_fn += fn
            # Note: micro_tn is accumulated in the aggregation step

        if gt_pos > 0 and not math.isnan(precision) and not math.isnan(recall) and not math.isnan(f1):
            macro_precisions.append(precision)
            macro_recalls.append(recall)
            macro_f1s.append(f1)
        if not math.isnan(accuracy):
            macro_accuracies.append(accuracy)

        results.append({
            "label": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "gt_positives": gt_pos,
            "gt_negatives": gt_neg,
            "pred_positives": pred_pos,
            "coverage": coverage,
            "total": total,
            "threshold": float(thresholds.get(label, float("nan"))),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "coverage_rate": coverage / total if total else float("nan"),
            "prevalence": (gt_pos / coverage) if coverage else float("nan"),
        })

    if "No Finding" not in labels_df.columns and gt_column_map:
        gt_subset = pd.DataFrame({lab: labels_df[col] for lab, col in gt_column_map.items()})
        gt_values = gt_subset.astype(float).to_numpy()
        all_blank = np.isnan(gt_values).all(axis=1)
        has_positive = (gt_values == 1).any(axis=1)
        nf_values = np.where(all_blank, np.nan, np.where(has_positive, 0.0, 1.0))
        labels_df["No Finding"] = nf_values

    if "No Finding" in preds_df.columns and "No Finding" in labels_df.columns:
        nf_pred = preds_df["No Finding"]
        nf_gt = labels_df["No Finding"]

        if eval_mode == "binary":
            nf_processed = nf_gt.fillna(0).replace(-1, 0).astype(int)
            mask_nf = nf_processed.isin([0, 1])
            nf_targets = nf_processed[mask_nf]
        else:
            mask_nf = nf_gt.isin([0, 1])
            nf_targets = nf_gt[mask_nf].astype(int) if mask_nf.sum() > 0 else pd.Series([], dtype=int)

        nf_preds = nf_pred[mask_nf].fillna(0).astype(int) if mask_nf.sum() > 0 else pd.Series([], dtype=int)
        coverage_nf = int(mask_nf.sum())
        total_nf = int(len(nf_gt))
        if coverage_nf > 0:
            tp_nf = int(((nf_targets == 1) & (nf_preds == 1)).sum())
            fp_nf = int(((nf_targets == 0) & (nf_preds == 1)).sum())
            fn_nf = int(((nf_targets == 1) & (nf_preds == 0)).sum())
            tn_nf = int(((nf_targets == 0) & (nf_preds == 0)).sum())
            precision_nf = tp_nf / (tp_nf + fp_nf) if (tp_nf + fp_nf) > 0 else np.nan
            recall_nf = tp_nf / (tp_nf + fn_nf) if (tp_nf + fn_nf) > 0 else 0.0
            f1_nf = (2 * precision_nf * recall_nf) / (precision_nf + recall_nf) if (precision_nf + recall_nf) > 0 else 0.0
            accuracy_nf = (tp_nf + tn_nf) / coverage_nf if coverage_nf > 0 else float("nan")
        else:
            tp_nf = fp_nf = fn_nf = tn_nf = 0
            precision_nf = recall_nf = f1_nf = accuracy_nf = float("nan")

        results.append({
            "label": "No Finding",
            "precision": precision_nf,
            "recall": recall_nf,
            "f1": f1_nf,
            "accuracy": accuracy_nf,
            "gt_positives": int((nf_targets == 1).sum()) if coverage_nf else 0,
            "gt_negatives": int((nf_targets == 0).sum()) if coverage_nf else 0,
            "pred_positives": int((nf_preds == 1).sum()) if coverage_nf else 0,
            "coverage": coverage_nf,
            "total": total_nf,
            "threshold": float("nan"),
            "tp": tp_nf,
            "fp": fp_nf,
            "fn": fn_nf,
            "tn": tn_nf,
            "coverage_rate": coverage_nf / total_nf if total_nf else float("nan"),
            "prevalence": (int((nf_targets == 1).sum()) / coverage_nf) if coverage_nf else float("nan"),
        })

    macro_precision = float(np.mean(macro_precisions)) if macro_precisions else float("nan")
    macro_recall = float(np.mean(macro_recalls)) if macro_recalls else float("nan")
    macro_f1 = float(np.mean(macro_f1s)) if macro_f1s else float("nan")
    
    # Calculate macro accuracy
    macro_accuracy = float(np.mean(macro_accuracies)) if macro_accuracies else float("nan")

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    # Calculate micro accuracy (TP + TN) / (TP + TN + FP + FN)
    micro_tn = sum(r.get("tn", 0) for r in results if r.get("label") in base_label_set)
    micro_total = micro_tp + micro_tn + micro_fp + micro_fn
    micro_accuracy = (micro_tp + micro_tn) / micro_total if micro_total > 0 else 0.0

    agg = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_accuracy": macro_accuracy,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "micro_accuracy": micro_accuracy,
        "micro_tp": micro_tp,
        "micro_tn": micro_tn,
        "micro_fp": micro_fp,
        "micro_fn": micro_fn,
    }
    return agg, results


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
    parser.add_argument("--align", choices=["reference", "intersection"], default="reference",
                        help="Alignment mode across sources and labels.")
    parser.add_argument("--out_probs_csv", default="outputs/final/test_probs.csv")
    parser.add_argument("--out_preds_csv", default="outputs/final/test_preds.csv")
    parser.add_argument("--out_metrics_csv", default="outputs/final/test_metrics.csv")
    parser.add_argument("--force_zero_labels", nargs="*", default=[], help="Always predict 0 for these labels")
    parser.add_argument("--eval_mode", choices=["certain_only", "binary"], default="certain_only",
                        help="Evaluation regime: certain_only masks -1/NaN; binary maps -1→0.")
    parser.add_argument("--skip_meta_calibration", action="store_true",
                        help="Skip meta calibration step (useful when inputs already include calibrated scores).")
    args = parser.parse_args()

    labels_df = ensure_filename_column(pd.read_csv(args.test_labels_csv))
    filenames = labels_df["filename"].tolist()
    sources = load_sources(args.probs_csv, filenames, align=args.align)
    if args.align == "intersection":
        inter_df = sources.pop("__filenames__")
        # Reduce labels_df to intersection
        labels_df = labels_df.merge(inter_df, on="filename", how="inner")
        filenames = labels_df["filename"].tolist()

    weights = json.loads(Path(args.blend_weights_json).read_text())
    labels = get_label_list(args.labels)
    excluded = set(args.exclude_labels)
    if excluded:
        labels = [l for l in labels if l not in excluded]
    blended = blend_probabilities(labels, sources, weights, score_prefix=args.score_prefix)

    if args.skip_meta_calibration:
        meta_params: Dict[str, Dict[str, float]] = {}
        for label in labels:
            cal_col = f"{args.meta_prefix}{label}"
            if cal_col in blended.columns:
                continue
            for cand_prefix in (args.score_prefix, "y_pred_"):
                cand_col = f"{cand_prefix}{label}"
                if cand_col in blended.columns:
                    blended[cal_col] = blended[cand_col]
                    break
    else:
        meta_params = load_meta_params(Path(args.meta_platt_json)) if args.meta_platt_json else {}
        apply_meta_calibration(blended, labels, meta_params)

    if not args.skip_meta_calibration:
        for label in labels:
            if label in INVERTED_PROB_LABELS:
                for prefix in {args.meta_prefix, args.score_prefix}:
                    col = f"{prefix}{label}"
                    if col in blended.columns:
                        blended[col] = 1.0 - blended[col].astype(float)

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
        force_zero_labels=args.force_zero_labels,
    )

    probs_out = Path(args.out_probs_csv)
    preds_out = Path(args.out_preds_csv)
    metrics_out = Path(args.out_metrics_csv)
    probs_out.parent.mkdir(parents=True, exist_ok=True)
    preds_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    blended.to_csv(probs_out, index=False)
    preds.to_csv(preds_out, index=False)

    agg_metrics, per_label = compute_metrics(labels, preds, labels_df, thresholds, args.eval_mode)
    metrics_df = pd.DataFrame(per_label)
    metrics_df.to_csv(metrics_out, index=False)
    agg_path = metrics_out.with_suffix(".macro_micro.json")
    agg_metrics_serializable = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v)) for k, v in agg_metrics.items()}
    agg_metrics_serializable["eval_mode"] = args.eval_mode
    agg_path.write_text(json.dumps(agg_metrics_serializable, indent=2))

    print(f"✅ Probabilities saved to {probs_out}")
    print(f"✅ Predictions saved to {preds_out}")
    print(f"✅ Metrics saved to {metrics_out}")
    print(f"Macro/Micro ({args.eval_mode}): {agg_metrics}")


if __name__ == "__main__":
    main()
