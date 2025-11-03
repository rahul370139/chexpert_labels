#!/usr/bin/env python3
"""
Per-label logistic stacker that combines multiple probability sources.

Inputs (train):
  --probs_csv path,name    (repeat) Calibrated probs with columns y_cal_{Label}
  --labels_csv path        Ground-truth with y_true_{Label} or {Label} columns
  --labels chexpert13|chexpert14
  --keep_labels ...        Optional explicit list to restrict labels
  --score_prefix y_cal_    Prefix of probability columns in sources

Outputs:
  - Stacked train probabilities CSV with y_pred_{Label}
  - JSON of per-label weights and bias

Apply mode (optional):
  Provide --apply and a matching set of --probs_csv for test to emit stacked test probs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Add project root
import sys

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.labels import get_label_list
from src.utils.align_by_filename import align_to_reference, ensure_filename_column


def parse_source(arg: str) -> Tuple[str, Path]:
    parts = arg.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid --probs_csv arg: {arg}. Expected path,name")
    return parts[1].strip(), Path(parts[0].strip())


def load_sources(sources: List[str], reference_filenames: List[str], align: str = "reference") -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    raw: Dict[str, pd.DataFrame] = {}
    if align == "reference":
        for spec in sources:
            name, path = parse_source(spec)
            df = pd.read_csv(path)
            aligned = align_to_reference(reference_filenames, df)
            frames[name] = aligned
        return frames
    # intersection
    for spec in sources:
        name, path = parse_source(spec)
        df = ensure_filename_column(pd.read_csv(path))
        raw[name] = df
    common = set(reference_filenames)
    for df in raw.values():
        common &= set(df["filename"].tolist())
    common_list = sorted(common)
    for name, df in raw.items():
        frames[name] = df.set_index("filename").reindex(common_list).reset_index()
    return frames


def assemble_feature_matrix(
    labels: Iterable[str],
    sources: Dict[str, pd.DataFrame],
    score_prefix: str,
) -> Dict[str, np.ndarray]:
    X_by_label: Dict[str, np.ndarray] = {}
    for label in labels:
        feats: List[np.ndarray] = []
        for name, df in sources.items():
            col = f"{score_prefix}{label}"
            if col not in df.columns:
                # fall back to raw label name
                col = label if label in df.columns else None
            if col is None or col not in df.columns:
                # If missing, contribute zeros
                feats.append(np.zeros((len(df), 1), dtype=np.float32))
            else:
                feats.append(df[col].astype(float).to_numpy().reshape(-1, 1))
        X = np.concatenate(feats, axis=1) if feats else np.zeros((len(next(iter(sources.values()))), 0))
        X_by_label[label] = X
    return X_by_label


def get_y_true(labels_df: pd.DataFrame, label: str) -> np.ndarray:
    if f"y_true_{label}" in labels_df.columns:
        return labels_df[f"y_true_{label}"].astype(int).to_numpy()
    if label in labels_df.columns:
        return labels_df[label].astype(int).to_numpy()
    raise ValueError(f"Ground truth column missing for {label}")


def train_stack(
    labels: List[str],
    train_sources: Dict[str, pd.DataFrame],
    labels_df: pd.DataFrame,
    score_prefix: str,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    filenames = labels_df["filename"].tolist()
    X_map = assemble_feature_matrix(labels, train_sources, score_prefix)
    stacked = pd.DataFrame({"filename": filenames})
    params: Dict[str, Dict[str, float]] = {}
    for label in labels:
        y = get_y_true(labels_df, label)
        X = X_map[label]
        if X.shape[1] == 0:
            stacked[f"y_pred_{label}"] = 0.0
            continue
        if len(np.unique(y)) < 2:
            # Not trainable; fallback to first source
            stacked[f"y_pred_{label}"] = X[:, 0]
            continue
        lr = LogisticRegression(max_iter=1000, class_weight="balanced")
        lr.fit(X, y)
        prob = lr.predict_proba(X)[:, 1]
        stacked[f"y_pred_{label}"] = prob.astype(np.float32)
        coeff = lr.coef_.ravel().tolist()
        params[label] = {"bias": float(lr.intercept_[0]), "weights": coeff, "sources": list(train_sources.keys())}
    return params, stacked


def apply_stack(labels: List[str], params: Dict[str, Dict[str, float]], test_sources: Dict[str, pd.DataFrame], score_prefix: str) -> pd.DataFrame:
    filenames = next(iter(test_sources.values()))["filename"].tolist()
    X_map = assemble_feature_matrix(labels, test_sources, score_prefix)
    out = pd.DataFrame({"filename": filenames})
    for label in labels:
        X = X_map[label]
        par = params.get(label)
        if par is None or X.shape[1] == 0:
            out[f"y_pred_{label}"] = X[:, 0] if X.shape[1] > 0 else 0.0
            continue
        w = np.array(par["weights"], dtype=float).reshape(-1, 1)
        b = float(par["bias"])
        logits = X @ w + b
        prob = 1.0 / (1.0 + np.exp(-logits))
        out[f"y_pred_{label}"] = prob.ravel().astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-label logistic stacking + export probs")
    ap.add_argument("--probs_csv", action="append", required=True, help="Train source CSV with alias: path,name (repeat)")
    ap.add_argument("--labels_csv", required=True, help="Ground truth train CSV")
    ap.add_argument("--labels", default="chexpert13")
    ap.add_argument("--keep_labels", nargs="*", default=None)
    ap.add_argument("--score_prefix", default="y_cal_")
    ap.add_argument("--out_params_json", default="outputs/blend/stack_params.json")
    ap.add_argument("--out_train_csv", default="outputs/blend/train_stacked.csv")
    ap.add_argument("--align", choices=["reference", "intersection"], default="reference")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--apply_probs_csv", action="append", help="Test source CSV with alias: path,name (repeat)")
    ap.add_argument("--apply_out_csv", default="outputs/blend/test_stacked.csv")
    args = ap.parse_args()

    labels = get_label_list(args.labels)
    if args.keep_labels:
        keep = set(args.keep_labels)
        labels = [l for l in labels if l in keep]

    labels_df = ensure_filename_column(pd.read_csv(args.labels_csv))
    reference_filenames = labels_df["filename"].tolist()
    train_sources = load_sources(args.probs_csv, reference_filenames, align=args.align)
    if args.align == "intersection":
        # reduce labels_df to intersection
        inter_fns = next(iter(train_sources.values()))["filename"].tolist()
        labels_df = labels_df[labels_df["filename"].isin(inter_fns)].reset_index(drop=True)

    params, stacked_train = train_stack(labels, train_sources, labels_df, args.score_prefix)
    Path(args.out_train_csv).parent.mkdir(parents=True, exist_ok=True)
    stacked_train.to_csv(args.out_train_csv, index=False)
    Path(args.out_params_json).write_text(json.dumps(params, indent=2))
    print(f"✅ Wrote {args.out_train_csv} and {args.out_params_json}")

    if args.apply:
        if not args.apply_probs_csv:
            raise SystemExit("--apply requires --apply_probs_csv")
        test_sources = load_sources(args.apply_probs_csv, reference_filenames)
        stacked_test = apply_stack(labels, params, test_sources, args.score_prefix)
        stacked_test.to_csv(args.apply_out_csv, index=False)
        print(f"✅ Wrote {args.apply_out_csv}")


if __name__ == "__main__":
    main()
