#!/usr/bin/env python3
import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

PREDICTIONS = Path('hybrid_ensemble_1000.csv')
GROUND_TRUTH = Path('data/evaluation_manifest_phaseA_matched.csv')
THRESHOLDS = Path('config/label_thresholds.json')

thresholds = json.loads(THRESHOLDS.read_text())
CHEXPERT13 = list(thresholds.keys())

pred = pd.read_csv(PREDICTIONS)
gt = pd.read_csv(GROUND_TRUTH)

bd = pred['binary_outputs'].fillna('{}').apply(json.loads)
for d in CHEXPERT13:
    pred[f'{d}_score'] = bd.apply(lambda r: r.get(d, {}).get('score', np.nan))

# Better path matching - extract the unique part: p10/subject/study/filename.jpg
def extract_key(path_str):
    p = Path(path_str)
    parts = p.parts
    # Find 'p10' and get everything after it
    if 'p10' in parts:
        idx = parts.index('p10')
        return '/'.join(parts[idx:])
    return p.name

pred['match_key'] = pred['image'].apply(extract_key)
gt['match_key'] = gt['image'].apply(extract_key)

merged = pd.merge(pred, gt, on='match_key', suffixes=('_pred','_gt'))
print(f'Matched: {len(merged)} images\n')

rows = []
all_y_true, all_y_pred = [], []
for d in CHEXPERT13:
    sc, gt_col = f'{d}_score', f'{d}_gt'
    if sc not in merged.columns or gt_col not in merged.columns: continue
    scores, y_true = merged[sc].values, merged[gt_col].values.astype(int)
    mask = ~np.isnan(scores)
    if mask.sum() == 0: continue
    y_pred, y_true_m = (scores[mask] >= thresholds[d]).astype(int), y_true[mask]
    if len(np.unique(y_true_m)) >= 2:
        p, r, f1, acc = precision_score(y_true_m, y_pred, zero_division=0), recall_score(y_true_m, y_pred, zero_division=0), f1_score(y_true_m, y_pred, zero_division=0), accuracy_score(y_true_m, y_pred)
        rows.append((d, thresholds[d], p, r, f1, acc))
    all_y_true.extend(y_true_m.tolist()); all_y_pred.extend(y_pred.tolist())

if rows:
    res = pd.DataFrame(rows, columns=['disease','tau','P','R','F1','Acc']).sort_values('F1', ascending=False)
    print('=== PER-DISEASE ===')
    for _, r in res.iterrows():
        print(f"{r['disease']:<30} tau={r['tau']:.2f}  P={r['P']:.3f}  R={r['R']:.3f}  F1={r['F1']:.3f}  Acc={r['Acc']:.3f}")
    print(f'\n=== MACRO-13 ===\nP={res.P.mean():.3f}  R={res.R.mean():.3f}  F1={res.F1.mean():.3f}  Acc={res.Acc.mean():.3f}')

if all_y_true:
    mp, mr, mf1 = precision_score(all_y_true, all_y_pred, zero_division=0), recall_score(all_y_true, all_y_pred, zero_division=0), f1_score(all_y_true, all_y_pred, zero_division=0)
    print(f'\n=== MICRO-13 ===\nP={mp:.3f}  R={mr:.3f}  F1={mf1:.3f}')

for d in CHEXPERT13:
    if f'{d}_score' in merged.columns:
        merged[f'{d}_pred_bin'] = (merged[f'{d}_score'] >= thresholds[d]).astype(int)
nf_pred = (merged[[f'{d}_pred_bin' for d in CHEXPERT13 if f'{d}_pred_bin' in merged.columns]].sum(axis=1)==0).astype(int)
nf_gt = (merged[[f'{d}_gt' for d in CHEXPERT13 if f'{d}_gt' in merged.columns]].sum(axis=1)==0).astype(int)
p, r, f1, acc = precision_score(nf_gt, nf_pred, zero_division=0), recall_score(nf_gt, nf_pred, zero_division=0), f1_score(nf_gt, nf_pred, zero_division=0), accuracy_score(nf_gt, nf_pred)
print(f'\n=== NO FINDING ===\nP={p:.3f}  R={r:.3f}  F1={f1:.3f}  Acc={acc:.3f}')

if rows:
    print(f'\n=== MACRO-14 ===\nP={(res.P.mean()+p)/2:.3f}  R={(res.R.mean()+r)/2:.3f}  F1={(res.F1.mean()+f1)/2:.3f}')
