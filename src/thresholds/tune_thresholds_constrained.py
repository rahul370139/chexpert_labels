#!/usr/bin/env python3
"""
Per-label constrained threshold tuning with optional prevalence guard.

Inputs:
  --calibrated_train_csv: CSV with y_true_<L>, y_pred_<L>
  --labels chexpert13|chexpert14
  --exclude_labels <list>
  --per_label_constraints_json: JSON mapping label->{precision_min, recall_min}
  --minfloors_json: JSON with per-label floors and _default_
  --prevalence_guard: float (default 0.10) allowed absolute diff vs train prevalence
  --out_thresholds_json, --out_summary_csv

Selection:
  - If constraint has recall_min: pick candidate with max precision among those with recall>=min
  - If constraint has precision_min: pick candidate with max recall among those with precision>=min
  - Else: pick candidate with max F1
  - Enforce min floor if provided
  - Enforce prevalence guard: abs(pred_prevalence - train_prevalence) <= guard
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score


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


def get_label_list(name: str) -> List[str]:
    key = name.strip().lower()
    if key in {"chexpert13", "chexpert_13", "13"}:
        return CHEXPERT13
    raise ValueError(f"Unsupported label group: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-label constrained threshold tuning")
    ap.add_argument("--calibrated_train_csv", required=True)
    ap.add_argument("--labels", default="chexpert13")
    ap.add_argument("--exclude_labels", nargs="*", default=[])
    ap.add_argument("--per_label_constraints_json", required=True)
    ap.add_argument("--minfloors_json", default=None)
    ap.add_argument("--prevalence_guard", type=float, default=0.10)
    ap.add_argument("--out_thresholds_json", required=True)
    ap.add_argument("--out_summary_csv", required=True)
    args = ap.parse_args()

    labels = [l for l in get_label_list(args.labels) if l not in set(args.exclude_labels)]
    df = pd.read_csv(args.calibrated_train_csv)

    constraints = json.loads(Path(args.per_label_constraints_json).read_text())
    minfloors = json.loads(Path(args.minfloors_json).read_text()) if args.minfloors_json else {}
    floor_default = float(minfloors.get("_default_", 0.0))

    thresholds: Dict[str, float] = {}
    rows = []

    for lab in labels:
        yt_col = f"y_true_{lab}"
        yp_col = f"y_pred_{lab}"
        if yt_col not in df.columns or yp_col not in df.columns:
            continue
        y_true = df[yt_col].astype(int).to_numpy()
        y_pred = df[yp_col].astype(float).to_numpy()
        prev_train = float(np.mean(y_true))

        p, r, t = precision_recall_curve(y_true, y_pred)
        candidates = []
        for i, th in enumerate(t):
            prec = float(p[i+1])
            rec = float(r[i+1])
            # prevalence at this threshold
            prev_hat = float(np.mean(y_pred >= th))
            # prevalence guard
            if abs(prev_hat - prev_train) > args.prevalence_guard:
                continue
            candidates.append((float(th), prec, rec, prev_hat))

        # enforce min floor
        min_floor = float(minfloors.get(lab, floor_default))
        candidates = [c for c in candidates if c[0] >= min_floor]

        chosen = None
        c = constraints.get(lab, {})
        if c.get("recall_min") is not None:
            rmin = float(c.get("recall_min"))
            feas = [cand for cand in candidates if cand[2] >= rmin]
            if feas:
                # pick max precision
                chosen = max(feas, key=lambda x: (x[1], x[2]))
        elif c.get("precision_min") is not None:
            pmin = float(c.get("precision_min"))
            feas = [cand for cand in candidates if cand[1] >= pmin]
            if feas:
                # pick max recall
                chosen = max(feas, key=lambda x: (x[2], x[1]))
        # fallback to best F1 if no feasible
        if chosen is None and candidates:
            best_f1 = -1.0
            for th, prec, rec, prev_hat in candidates:
                f1 = (2*prec*rec)/(prec+rec+1e-12) if (prec+rec) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    chosen = (th, prec, rec, prev_hat)

        if chosen is None:
            # last resort: median threshold
            th = float(np.median(t)) if len(t) > 0 else 0.5
            prec = rec = 0.0
            prev_hat = float(np.mean(y_pred >= th))
            chosen = (max(th, min_floor), prec, rec, prev_hat)

        th, prec, rec, prev_hat = chosen
        thresholds[lab] = th
        rows.append({
            "label": lab,
            "threshold": th,
            "precision": prec,
            "recall": rec,
            "prev_train": prev_train,
            "prev_hat": prev_hat,
            "constraint": constraints.get(lab, {})
        })

    Path(args.out_thresholds_json).write_text(json.dumps(thresholds, indent=2))
    pd.DataFrame(rows).to_csv(args.out_summary_csv, index=False)
    print(f"✅ Wrote {args.out_thresholds_json} and {args.out_summary_csv}")


if __name__ == "__main__":
    main()

