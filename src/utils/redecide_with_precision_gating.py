#!/usr/bin/env python3
"""
Re-decide labels from an existing CheXagent hybrid CSV using stricter precision-first gating.

Reads JSON fields (binary_outputs, di_outputs), applies margin + DI-strength rules,
writes an improved CSV, and reports per-label and overall metrics.

Usage:
  python redecide_with_precision_gating.py \
      --pred hybrid_ensemble_1000.csv \
      --ground_truth data/evaluation_manifest_phaseA_matched.csv \
      --out hybrid_ensemble_1000_improved.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

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


def compute_metrics(merged: pd.DataFrame, labels: List[str], pred_cols: Dict[str, str]):
    rows = []
    for label in labels + ["No Finding"]:
        # Ground truth column
        if f"{label}_gt" in merged.columns:
            y_true = merged[f"{label}_gt"].astype(int).values
        elif label in merged.columns:
            y_true = merged[label].astype(int).values
        else:
            continue
        # Prediction column
        col = pred_cols.get(label)
        if col is None:
            if label in merged.columns:
                col = label
            elif f"{label}_pred" in merged.columns:
                col = f"{label}_pred"
            else:
                continue
        y_pred = merged[col].astype(int).values
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        rows.append((label, p, r, f))

    macro_p = np.mean([r[1] for r in rows])
    macro_r = np.mean([r[2] for r in rows])
    macro_f = np.mean([r[3] for r in rows])
    y_true_all = merged[[f"{l}_gt" for l in labels]].values.ravel()
    y_pred_all = merged[[pred_cols.get(l, l) for l in labels]].values.ravel()
    micro_p = precision_score(y_true_all, y_pred_all, zero_division=0)
    micro_r = recall_score(y_true_all, y_pred_all, zero_division=0)
    micro_f = f1_score(y_true_all, y_pred_all, zero_division=0)
    return rows, (macro_p, macro_r, macro_f), (micro_p, micro_r, micro_f)


def redecide(pred_csv: Path, gt_csv: Path, out_csv: Path):
    pred = pd.read_csv(pred_csv)
    gt = pd.read_csv(gt_csv)
    pred["filename"] = pred["image"].apply(lambda x: Path(x).name)
    gt["filename"] = gt["image"].apply(lambda x: Path(x).name)

    # Gating knobs
    CONF_MARGIN = 0.05
    DI_STRICT_STRENGTH = 0.70
    DI_BORDERLINE_WINDOW = 0.15
    DI_BOOST_LABELS = {"Edema","Pleural Effusion","Pleural Other"}
    DI_DISABLED = {"Pleural Other"}  # cut FPs drastically

    if "binary_outputs" not in pred.columns or "di_outputs" not in pred.columns:
        raise SystemExit("CSV must contain 'binary_outputs' and 'di_outputs' JSON columns.")

    bod = pred["binary_outputs"].fillna("{}").apply(json.loads)
    dio = pred["di_outputs"].fillna("{}").apply(json.loads)

    # Redecide
    improved_pred = pred.copy()
    for label in CHEXPERT13:
        scores, taus, ment, neg, unc, strength = [], [], [], [], [], []
        for b, d in zip(bod, dio):
            br = b.get(label, {})
            dr = d.get(label, {})
            scores.append(float(br.get("score", 0.5)))
            taus.append(float(br.get("threshold", 0.5)))
            ment.append(int(dr.get("mentioned",0)))
            neg.append(int(dr.get("negated",0)))
            unc.append(int(dr.get("uncertain",0)))
            strength.append(float(dr.get("strength",0.0)))
        scores = np.asarray(scores); taus = np.asarray(taus)
        ment = np.asarray(ment); neg = np.asarray(neg); unc = np.asarray(unc); strength = np.asarray(strength)

        decision = np.zeros_like(scores, dtype=int)
        decision[neg == 1] = 0
        strong = scores >= (taus + CONF_MARGIN)
        decision[strong] = 1
        # Borderline above tau
        border = (scores >= taus) & (scores < (taus + CONF_MARGIN))
        if (label in DI_BOOST_LABELS) and (label not in DI_DISABLED):
            idx = np.where(border & (unc==0) & (ment==1) & (strength >= DI_STRICT_STRENGTH))[0]
            decision[idx] = 1
        # Slightly below tau
        close = (scores >= (taus - DI_BORDERLINE_WINDOW)) & (scores < taus)
        if (label in DI_BOOST_LABELS) and (label not in DI_DISABLED):
            idx = np.where(close & (unc==0) & (ment==1) & (strength >= DI_STRICT_STRENGTH))[0]
            decision[idx] = 1

        improved_pred[label] = decision

    improved_pred["No Finding"] = (improved_pred[CHEXPERT13].sum(axis=1) == 0).astype(int)

    # Metrics: merge improved with GT
    merged_new = improved_pred.merge(gt, on="filename", suffixes=("_pred","_gt"))
    merged_base = pred.merge(gt, on="filename", suffixes=("_pred","_gt"))

    base_pred_cols = {l: f"{l}_pred" for l in CHEXPERT13}
    base_rows, base_macro, base_micro = compute_metrics(merged_base, CHEXPERT13, base_pred_cols)

    new_pred_cols = {l: f"{l}_pred" for l in CHEXPERT13}
    # align column names for improved: copy improved decisions into merged_new expected columns via filename
    dec_map = improved_pred.drop_duplicates("filename").set_index("filename")[CHEXPERT13]
    for l in CHEXPERT13:
        series = merged_new["filename"].map(dec_map[l])
        merged_new[f"{l}_pred"] = series.fillna(0).astype(int)
    new_rows, new_macro, new_micro = compute_metrics(merged_new, CHEXPERT13, new_pred_cols)

    # Save improved CSV (same schema as input, but with updated binary columns)
    improved_pred.to_csv(out_csv, index=False)

    print("\n=== Baseline (from *_pred columns) ===")
    print(f"Micro P/R/F: {base_micro}")
    print(f"Macro P/R/F: {base_macro}")
    print("\n=== Improved (new gating decisions) ===")
    print(f"Micro P/R/F: {new_micro}")
    print(f"Macro P/R/F: {new_macro}")
    print("\nPer-label improved (P, R, F1):")
    for label, p, r, f in new_rows:
        print(f"{label:24s} P={p:.3f} R={r:.3f} F1={f:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ground_truth", required=True)
    ap.add_argument("--out", default="hybrid_ensemble_1000_improved.csv")
    args = ap.parse_args()
    redecide(Path(args.pred), Path(args.ground_truth), Path(args.out))


if __name__ == "__main__":
    main()
