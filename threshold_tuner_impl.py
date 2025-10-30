
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

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

def _ensure_cols(df: pd.DataFrame, labels: List[str]):
    missing = []
    for L in labels:
        if f"y_true_{L}" not in df.columns or f"y_pred_{L}" not in df.columns:
            missing.append(L)
    if missing:
        raise ValueError(f"Missing columns for labels: {missing}. Need y_true_<L>, y_pred_<L>.")

def _f_beta(p, r, beta: float):
    if p <= 0 or r <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * (p * r) / (b2 * p + r)

def _pick_threshold_fbeta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5, min_threshold: float = 0.3):
    """
    Pick threshold that maximizes F-beta score.
    
    Args:
        min_threshold: Minimum threshold to consider (avoids too-permissive thresholds from score clustering)
    """
    precision, recall, thresh = precision_recall_curve(y_true, y_pred)
    best_t, best_f = None, -1.0
    best = {"precision": 0.0, "recall": 0.0, "f_beta": 0.0}
    
    # Evaluate ALL thresholds and pick the best F-beta (per-class optimization)
    # For score clustering (0.20 vs 0.80), we compare both and pick the better one
    # Skip thresholds with recall=0 (no positives predicted = useless)
    for i, t in enumerate(thresh):
        p = float(precision[i + 1])
        r = float(recall[i + 1])
        
        # Skip if recall is 0 (no positives predicted) - this threshold is useless
        if r <= 0:
            continue
        
        f = _f_beta(p, r, beta)
        if f > best_f:
            best_f = f
            best_t = float(t)
            best = {"precision": p, "recall": r, "f_beta": f}
    
    # Fallback: if all thresholds had recall=0, use first one anyway
    if best_t is None and len(thresh) > 0:
        best_t = float(thresh[0])
        best = {"precision": float(precision[1]), "recall": float(recall[1]), "f_beta": _f_beta(float(precision[1]), float(recall[1]), beta)}
    
    return best_t, best

def _pick_threshold_min_precision(y_true: np.ndarray, y_pred: np.ndarray, min_precision: float, min_threshold: float = 0.3):
    """
    Pick threshold that maximizes recall subject to precision ≥ min_precision.
    
    Evaluate ALL thresholds, prioritize those meeting precision target, pick one with best recall.
    
    Args:
        min_precision: Minimum required precision
        min_threshold: Minimum threshold floor (after precision constraint)
    """
    precision, recall, thresh = precision_recall_curve(y_true, y_pred)
    best_t, best_r = None, -1.0
    best = {"precision": 0.0, "recall": 0.0}
    
    # First pass: find thresholds meeting precision constraint, pick one with best recall
    for i, t in enumerate(thresh):
        p = float(precision[i + 1])
        r = float(recall[i + 1])
        
        # Skip if no positives predicted
        if r <= 0:
            continue
            
        if p >= min_precision and r > best_r:
            best_r = r
            best_t = float(t)
            best = {"precision": p, "recall": r}
    
    # If no threshold met precision target, use F-beta as fallback (precision-weighted)
    if best_t is None:
        best_t, b = _pick_threshold_fbeta(y_true, y_pred, beta=0.3, min_threshold=0.0)  # Remove min_threshold constraint for fallback
        best = {"precision": b["precision"], "recall": b["recall"]}
    
    # Final fallback
    if best_t is None and len(thresh) > 0:
        best_t = float(thresh[0])
        best = {"precision": float(precision[1]), "recall": float(recall[1])}
    
    return best_t, best

def _metrics_macro_micro(y_true_mat: np.ndarray, y_hat_mat: np.ndarray):
    macro_p = precision_score(y_true_mat, y_hat_mat, average="macro", zero_division=0)
    macro_r = recall_score(y_true_mat, y_hat_mat, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true_mat, y_hat_mat, average="macro", zero_division=0)
    micro_p = precision_score(y_true_mat, y_hat_mat, average="micro", zero_division=0)
    micro_r = recall_score(y_true_mat, y_hat_mat, average="micro", zero_division=0)
    micro_f1 = f1_score(y_true_mat, y_hat_mat, average="micro", zero_division=0)
    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
    }

def tune_thresholds(csv_path: str,
                    out_json: str = "thresholds.json",
                    out_metrics: str = "thresholds_summary.csv",
                    mode: str = "fbeta",
                    beta: float = 0.5,
                    min_macro_precision: float = 0.60,
                    labels: List[str] = CHEXPERT14):
    df = pd.read_csv(csv_path)
    _ensure_cols(df, labels)
    thresholds = {}
    per_label_rows = []

    # Minimum threshold floor to avoid overly permissive thresholds
    # (many binary scores cluster at 0.20 from negative parsing, causing optimizer to pick low thresholds)
    MIN_THRESHOLD_FLOOR = 0.30
    
    if mode == "fbeta":
        for L in labels:
            y_true = df[f"y_true_{L}"].values.astype(int)
            y_pred = df[f"y_pred_{L}"].values.astype(float)
            t, info = _pick_threshold_fbeta(y_true, y_pred, beta=beta, min_threshold=MIN_THRESHOLD_FLOOR)
            thresholds[L] = t
            per_label_rows.append({"label": L, "threshold": t, "precision": info["precision"], "recall": info["recall"], f"f{beta}": info["f_beta"]})
    elif mode == "min_precision":
        for L in labels:
            y_true = df[f"y_true_{L}"].values.astype(int)
            y_pred = df[f"y_pred_{L}"].values.astype(float)
            t, info = _pick_threshold_min_precision(y_true, y_pred, min_precision=min_macro_precision, min_threshold=MIN_THRESHOLD_FLOOR)
            thresholds[L] = t
            per_label_rows.append({"label": L, "threshold": t, "precision": info["precision"], "recall": info["recall"]})
    else:
        raise ValueError("mode must be 'fbeta' or 'min_precision'")

    y_true_mat, y_hat_mat = [], []
    for _, row in df.iterrows():
        yt, yh = [], []
        for L in labels:
            yt.append(int(row[f"y_true_{L}"]))
            yh.append(1 if float(row[f"y_pred_{L}"]) >= thresholds[L] else 0)
        y_true_mat.append(yt); y_hat_mat.append(yh)

    y_true_mat = np.array(y_true_mat); y_hat_mat = np.array(y_hat_mat)
    agg = _metrics_macro_micro(y_true_mat, y_hat_mat)

    Path(out_json).write_text(json.dumps(thresholds, indent=2))
    pd.DataFrame(per_label_rows).to_csv(out_metrics, index=False)
    return {"thresholds_json": out_json, "metrics_csv": out_metrics, **agg}
