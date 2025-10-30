#!/usr/bin/env python3
"""Calculate detailed per-disease and overall performance metrics."""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load new thresholds
thresholds = json.loads(Path('config/label_thresholds.json').read_text())

# Load existing predictions and ground truth
predictions_df = pd.read_csv('hybrid_ensemble_1000.csv')
gt_df = pd.read_csv('data/evaluation_manifest_1000.csv')

# Parse binary outputs
bd = predictions_df['binary_outputs'].fillna('{}').apply(json.loads)
CHEXPERT13 = list(thresholds.keys())

for disease in CHEXPERT13:
    predictions_df[f'{disease}_score'] = bd.apply(lambda r: r.get(disease, {}).get('score', np.nan))

# Match on filename
predictions_df['filename'] = predictions_df['image'].apply(lambda x: Path(x).name)
gt_df['filename'] = gt_df['image'].apply(lambda x: Path(x).name)
merged = pd.merge(predictions_df, gt_df, on='filename', suffixes=('_pred', '_gt'))

print('='*100)
print('📊 COMPREHENSIVE PERFORMANCE REPORT - Per-Class Thresholds')
print('='*100)
print(f'\nDataset: {len(merged)} matched images\n')

# Calculate for all diseases
all_results = []
all_y_true_micro = []
all_y_pred_micro = []

for disease in CHEXPERT13:
    score_col = f'{disease}_score'
    gt_col = f'{disease}_gt'
    tau = thresholds[disease]
    
    scores = merged[score_col].values
    y_true = merged[gt_col].values.astype(int)
    mask = ~np.isnan(scores)
    
    if mask.sum() == 0:
        continue
    
    y_true_masked = y_true[mask]
    scores_masked = scores[mask]
    y_pred_masked = (scores_masked >= tau).astype(int)
    
    # Metrics
    precision = precision_score(y_true_masked, y_pred_masked, zero_division=0)
    recall = recall_score(y_true_masked, y_pred_masked, zero_division=0)
    f1 = f1_score(y_true_masked, y_pred_masked, zero_division=0)
    accuracy = accuracy_score(y_true_masked, y_pred_masked)
    
    # Counts
    tp = ((y_true_masked == 1) & (y_pred_masked == 1)).sum()
    fp = ((y_true_masked == 0) & (y_pred_masked == 1)).sum()
    fn = ((y_true_masked == 1) & (y_pred_masked == 0)).sum()
    tn = ((y_true_masked == 0) & (y_pred_masked == 0)).sum()
    total_positives_gt = y_true_masked.sum()
    total_positives_pred = y_pred_masked.sum()
    
    all_results.append({
        'disease': disease,
        'threshold': tau,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'gt_positives': total_positives_gt,
        'pred_positives': total_positives_pred,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    })
    
    all_y_true_micro.extend(y_true_masked)
    all_y_pred_micro.extend(y_pred_masked)

# Handle No Finding
for d in CHEXPERT13:
    score_col = f'{d}_score'
    tau = thresholds[d]
    scores = merged[score_col].values
    mask = ~np.isnan(scores)
    merged.loc[mask, f'{d}_pred_calc'] = (merged.loc[mask, score_col].values >= tau).astype(int)

no_finding_pred_actual = (merged[[f'{d}_pred_calc' for d in CHEXPERT13]].sum(axis=1) == 0).astype(int)
no_finding_gt_actual = (merged[[f'{d}_gt' for d in CHEXPERT13]].sum(axis=1) == 0).astype(int)

nf_precision = precision_score(no_finding_gt_actual, no_finding_pred_actual, zero_division=0)
nf_recall = recall_score(no_finding_gt_actual, no_finding_pred_actual, zero_division=0)
nf_f1 = f1_score(no_finding_gt_actual, no_finding_pred_actual, zero_division=0)
nf_accuracy = accuracy_score(no_finding_gt_actual, no_finding_pred_actual)

nf_tp = ((no_finding_gt_actual == 1) & (no_finding_pred_actual == 1)).sum()
nf_fp = ((no_finding_gt_actual == 0) & (no_finding_pred_actual == 1)).sum()
nf_fn = ((no_finding_gt_actual == 1) & (no_finding_pred_actual == 0)).sum()
nf_tn = ((no_finding_gt_actual == 0) & (no_finding_pred_actual == 0)).sum()

# Print per-disease results
print('PER-DISEASE RESULTS:')
print('-'*100)
print(f"{'Disease':<30} {'τ':<6} {'TP':<5} {'FP':<5} {'FN':<5} {'TN':<6} {'P':<7} {'R':<7} {'F1':<7} {'Acc':<7}")
print('-'*100)

for r in sorted(all_results, key=lambda x: x['f1'], reverse=True):
    d = r['disease']
    t = r['threshold']
    print(f'{d:<30} {t:<6.2f} {r["tp"]:<5} {r["fp"]:<5} {r["fn"]:<5} {r["tn"]:<6} '
          f'{r["precision"]:<7.3f} {r["recall"]:<7.3f} {r["f1"]:<7.3f} {r["accuracy"]:<7.3f}')

# No Finding
print(f'{"No Finding":<30} {"N/A":<6} {nf_tp:<5} {nf_fp:<5} {nf_fn:<5} {nf_tn:<6} '
      f'{nf_precision:<7.3f} {nf_recall:<7.3f} {nf_f1:<7.3f} {nf_accuracy:<7.3f}')

print('-'*100)

# Overall metrics (CHEXPERT13)
results_df = pd.DataFrame(all_results)
macro_p = results_df['precision'].mean()
macro_r = results_df['recall'].mean()
macro_f1 = results_df['f1'].mean()
macro_acc = results_df['accuracy'].mean()

# Micro-averaged (CHEXPERT13)
micro_p = precision_score(all_y_true_micro, all_y_pred_micro, zero_division=0)
micro_r = recall_score(all_y_true_micro, all_y_pred_micro, zero_division=0)
micro_f1 = f1_score(all_y_true_micro, all_y_pred_micro, zero_division=0)

# Include No Finding in overall metrics
all_diseases_results = all_results + [{
    'disease': 'No Finding',
    'precision': nf_precision,
    'recall': nf_recall,
    'f1': nf_f1,
    'accuracy': nf_accuracy
}]
macro_p_all = np.mean([r['precision'] for r in all_diseases_results])
macro_r_all = np.mean([r['recall'] for r in all_diseases_results])
macro_f1_all = np.mean([r['f1'] for r in all_diseases_results])
macro_acc_all = np.mean([r['accuracy'] for r in all_diseases_results])

all_y_true_micro_all = list(all_y_true_micro) + list(no_finding_gt_actual)
all_y_pred_micro_all = list(all_y_pred_micro) + list(no_finding_pred_actual)
micro_p_all = precision_score(all_y_true_micro_all, all_y_pred_micro_all, zero_division=0)
micro_r_all = recall_score(all_y_true_micro_all, all_y_pred_micro_all, zero_division=0)
micro_f1_all = f1_score(all_y_true_micro_all, all_y_pred_micro_all, zero_division=0)

print('\n' + '='*100)
print('OVERALL PERFORMANCE METRICS')
print('='*100)
print(f'\n📊 CHEXPERT13 (13 diseases, excluding No Finding):')
print(f'   Macro-averaged:')
print(f'      Precision: {macro_p:.4f}')
print(f'      Recall:    {macro_r:.4f}')
print(f'      F1-Score:  {macro_f1:.4f}')
print(f'      Accuracy:  {macro_acc:.4f}')
print(f'\n   Micro-averaged:')
print(f'      Precision: {micro_p:.4f}')
print(f'      Recall:    {micro_r:.4f}')
print(f'      F1-Score:  {micro_f1:.4f}')

print(f'\n📊 CHEXPERT14 (including No Finding):')
print(f'   Macro-averaged:')
print(f'      Precision: {macro_p_all:.4f}')
print(f'      Recall:    {macro_r_all:.4f}')
print(f'      F1-Score:  {macro_f1_all:.4f}')
print(f'      Accuracy:  {macro_acc_all:.4f}')
print(f'\n   Micro-averaged:')
print(f'      Precision: {micro_p_all:.4f}')
print(f'      Recall:    {micro_r_all:.4f}')
print(f'      F1-Score:  {micro_f1_all:.4f}')

print('\n' + '='*100)

