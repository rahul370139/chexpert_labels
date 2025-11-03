# CheXpert Hybrid Evaluation Report

## Executive Summary (Certain-Only)

### Macro Metrics (excluding No Finding)

- Precision: 0.856
- Recall: 0.755
- F1: 0.763
- Accuracy: 0.723

### Micro Metrics (excluding No Finding)

- Precision: 0.874
- Recall: 0.729
- F1: 0.795
- Accuracy: 0.700

## Binary Summary (−1 → 0)

### Macro Metrics

- Precision: 0.199
- Recall: 0.755
- F1: 0.267
- Accuracy: 0.418

### Micro Metrics

- Precision: 0.193
- Recall: 0.729
- F1: 0.306
- Accuracy: 0.418

## No Finding Metrics

- Precision: 0.000
- Recall: 0.000
- F1: 0.000
- Accuracy: 0.874
- Coverage: 1167/1167
- GT Positives: 147
- Pred Positives: 0

## Per-Label Performance (Certain-Only)

| Label | P | R | F1 | Accuracy | Coverage | Prevalence | τ |
|---|---|---|---|---|---|---|---|
| Enlarged Cardiomediastinum | 0.824 | 0.824 | 0.824 | 0.733 | 45/1167 | 0.756 | - |
| Cardiomegaly | 0.866 | 0.866 | 0.866 | 0.778 | 378/1167 | 0.828 | - |
| Lung Opacity | 0.966 | 0.963 | 0.964 | 0.931 | 363/1167 | 0.967 | - |
| Lung Lesion | 0.917 | 1.000 | 0.957 | 0.917 | 36/1167 | 0.917 | - |
| Edema | 0.796 | 0.842 | 0.818 | 0.754 | 289/1167 | 0.657 | - |
| Consolidation | 0.902 | 0.979 | 0.939 | 0.893 | 112/1167 | 0.839 | - |
| Pneumonia | 0.648 | 0.680 | 0.663 | 0.606 | 175/1167 | 0.571 | - |
| Atelectasis | 0.980 | 0.959 | 0.969 | 0.941 | 320/1167 | 0.981 | - |
| Pneumothorax | 0.412 | 0.420 | 0.416 | 0.663 | 392/1167 | 0.286 | - |
| Pleural Effusion | 0.896 | 0.905 | 0.901 | 0.829 | 592/1167 | 0.856 | - |
| Pleural Other | 1.000 | 0.160 | 0.276 | 0.160 | 25/1167 | 1.000 | - |
| Fracture | 0.950 | 1.000 | 0.974 | 0.950 | 20/1167 | 0.950 | - |
| Support Devices | 0.976 | 0.216 | 0.353 | 0.238 | 596/1167 | 0.965 | - |
