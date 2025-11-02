# CheXpert Hybrid Ensemble

CheXagent’s textual reasoning is powerful but its binary scores cluster at 0.2/0.8. This project turns CheXagent into a production-friendly hybrid by adding TorchXRayVision vision models, linear probes, constrained thresholds, and DI-based gating. It also exposes a Streamlit demo so you can browse results or upload a new CXR.

## What’s New

| Component | Purpose |
|-----------|---------|
| TorchXRayVision `densenet121-res224-chex` | Continuous probabilities for all CheXpert labels |
| TorchXRayVision **`resnet50-res512-all` (6.8 GB)** | High-capacity ensemble for the 5 chronic troublemakers: Enlarged Cardiomediastinum, Lung Lesion, Pneumothorax, Pleural Other, Fracture |
| CheXagent linear probes | Logistic head on frozen CLIP embeddings (per-label Platt calibration) |
| Blend search + meta calibration | Per-label weights across txr / probe / txr_heavy, followed by auto Platt/Isotonic |
| Constrained threshold tuner | Precision-first (default) & recall-first profiles with prevalence guard |
| DI gating + impression booster | AND/OR logic, keyword fixes, “No Finding” enforcement |
| Three-class thresholds | Optional -1/0/1 output with an uncertainty band around τ |
| `app_demo_chexagent.py` | Streamlit app to browse evaluation samples or upload a new study |

## Repository Map

```
src/
├── inference/
│   ├── txr_infer.py                # TorchXRayVision inference (any weights)
│   ├── txr_selective_infer.py      # Heavy TXR for selected labels (no/optional blending)
│   └── smart_ensemble.py           # CheXagent DI + binary reasoning
├── embeddings/
│   └── extract_chexagent_embeddings.py
├── models/
│   └── train_linear_probe.py       # Logistic regression per label
├── calibration/
│   ├── fit_label_calibrators.py    # Platt calibration per source
│   ├── apply_label_calibrators.py
│   └── meta_calibrate.py           # Auto (Platt/Isotonic) meta calibration
├── blending/
│   └── search_blend_weights.py     # Per-label weight grid search (any # of sources)
├── thresholding/
│   └── tune_thresholds_constrained.py
├── evaluation/
│   ├── run_test_eval.py            # Blend → meta-cal → DI gating
│   ├── apply_three_class_thresholds.py
│   └── evaluate_three_class.py
└── pipelines/
    └── run_5k_blend_eval.py        # End-to-end orchestration
```

## Requirements

1. **Phase-A images (≈5.8 k)**  
   `data/image_list_phaseA_5k_absolute.txt` and `data/evaluation_manifest_phaseA_5k.csv` reference files under `../radiology_report/files`. Mount/copy the dataset accordingly.

2. **TorchXRayVision weights**  
   Download once on a machine with network access:
   ```bash
   python - <<'PY'
   import torchxrayvision as xrv
   xrv.models.DenseNet(weights="densenet121-res224-chex")
   xrv.models.DenseNet(weights="resnet50-res512-all")
   PY
   ```
   Copy the resulting `.pth` files from `~/.cache/torchxrayvision/` to the offline machine if necessary.

3. `venv` with the dependencies from `requirements.txt` (torch, torchvision, torchxrayvision, transformers, scikit-learn, streamlit).

## End-to-End Pipeline

```bash
cd chexagent_chexpert_eval
./venv/bin/python src/pipelines/run_5k_blend_eval.py \
    --images data/image_list_phaseA_5k_absolute.txt \
    --manifest data/evaluation_manifest_phaseA_5k.csv \
    --chexagent_metadata results/hybrid_ensemble_5826.csv \
    --device mps \
    --out_root outputs_full \
    --train_ratio 0.8 \
    --resume
```

The pipeline is idempotent and performs:
1. TXR inference (`densenet121-res224-chex`) + Platt calibration
2. Heavy TXR inference (`resnet50-res512-all`) on the five worst labels + calibration
3. CheXagent CLIP embeddings (train/test) + linear probes + Platt calibration
4. Per-label blend search (txr / probe / txr_heavy) with β = 0.3
5. Auto meta calibration (Platt or Isotonic based on positives count)
6. Precision-first constrained thresholds (`config/per_label_constraints_precision_first.json`)
7. DI-gated evaluation, impression generation, and manager report

### Refreshing only the thresholds

```bash
./venv/bin/python src/thresholds/tune_thresholds_constrained.py \
    --calibrated_train_csv outputs_full/blend/train_blended_calibrated.csv \
    --labels chexpert13 \
    --exclude_labels Cardiomegaly Atelectasis \
    --per_label_constraints_json config/per_label_constraints_precision_first.json \
    --minfloors_json config/minfloors.json \
    --prevalence_guard 0.10 \
    --out_thresholds_json outputs_analysis/thresholds_precision.json \
    --out_summary_csv outputs_analysis/thresholds_precision.csv

./venv/bin/python src/evaluation/run_test_eval.py \
    --probs_csv outputs_full/txr/test_txr_calibrated.csv,txr \
    --probs_csv outputs_full/linear_probe/test_calibrated.csv,probe \
    --probs_csv outputs_full/txr/test_txr_heavy_calibrated.csv,txr_heavy \
    --blend_weights_json outputs_full/blend/blend_weights_refit_full.json \
    --meta_platt_json outputs_full/calibration/meta_auto_refit_full.json \
    --thresholds_json outputs_analysis/thresholds_precision.json \
    --test_labels_csv outputs_full/splits/test.csv \
    --labels chexpert13 \
    --exclude_labels Cardiomegaly Atelectasis \
    --gating_config config/gating.json \
    --metadata_csv results/hybrid_ensemble_5826.csv \
    --out_probs_csv outputs_analysis/test_probs_precision.csv \
    --out_preds_csv outputs_analysis/test_preds_precision.csv \
    --out_metrics_csv outputs_analysis/test_metrics_precision.csv
```

### Three-class evaluation

```bash
./venv/bin/python src/data_prep/create_phaseA_manifest_5k.py \
    --phaseA_jsonl ../radiology_report/src/data/processed/phaseA_manifest.jsonl \
    --chexagent_csv results/hybrid_ensemble_5826.csv \
    --output data/evaluation_manifest_phaseA_5k_three.csv \
    --keep_uncertain

./venv/bin/python src/evaluation/apply_three_class_thresholds.py \
    --probs_csv outputs_full/final/test_probs_refit_full.csv \
    --thresholds_json outputs_full/thresholds/thresholds_constrained_refit_full.json \
    --output outputs_full/final/test_preds_three_class.csv \
    --uncertainty_margin 0.15

./venv/bin/python src/evaluation/evaluate_three_class.py \
    --predictions outputs_full/final/test_preds_three_class.csv \
    --ground_truth data/evaluation_manifest_phaseA_5k_three.csv \
    --output outputs_full/final/three_class_evaluation.csv
```

## Current Metrics (server-synced split, 2-stream blend)

Summary JSON files are stored in `outputs_analysis/`.

| Profile | Macro P | Macro R | Macro F1 | Micro P | Micro R | Micro F1 |
|---------|---------|---------|----------|---------|---------|----------|
| Baseline (`server_synced` thresholds) | 0.178 | 0.143 | 0.155 | 0.345 | 0.334 | 0.339 |
| **Precision-first constrained (refit)** | **0.190** | **0.252** | **0.213** | **0.327** | **0.437** | **0.374** |

Worst-F1 labels remain Lung Lesion, Fracture, Pleural Other, Enlarged Cardiomediastinum and Pneumothorax – the heavy TXR stream plus DI keyword booster is designed to pull these up when you rerun the full pipeline against the complete 5.8 k images.

## Streamlit Demo

```bash
streamlit run app_demo_chexagent.py
```

The app:
- Loads artefacts from `outputs_full/` (falls back to `server_synced/outputs_full`)
- Lets you **browse** evaluation samples (image, probabilities, binary predictions, tri-state view, ground truth)
- Supports **image upload** – runs TXR (base + heavy), CheXagent linear probes, blend + meta-calibration, DI gating, and emits a concise impression highlighting positive & uncertain findings

## Comparing to Raw CheXpert Labels

1. **Binary (0/1)** – use `run_test_eval.py` followed by `summarize_metrics.py`  
   Example:
   ```bash
   ./venv/bin/python src/evaluation/summarize_metrics.py \
       --metrics_csv outputs_analysis/test_metrics_precision.csv \
       --labels_csv server_synced/data_full/ground_truth_test_30.csv \
       --out_json outputs_analysis/metrics_precision_summary.json
   ```

2. **Three-class (-1/0/1)** – generate the uncertain manifest (`create_phaseA_manifest_5k.py --keep_uncertain`), then run `apply_three_class_thresholds.py` and `evaluate_three_class.py`.

Both scripts list macro/micro metrics **and** the lowest-F1 labels so you can tweak constraints (e.g. increase precision floors for Fracture/Pleural Other or reduce the DI rescue window).

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Missing image …` | The Phase-A files are not mounted. Copy them to `../radiology_report/files` or update `data/image_list_phaseA_5k_absolute.txt`. |
| Unable to download 6.8 GB TXR weights | Download once on an online machine and copy the cached `.pth` file to the offline box. |
| Streamlit app warns about missing artefacts | Run `run_5k_blend_eval.py` (see above) or copy the `server_synced/outputs_full` artefacts. |
| Need recall-first thresholds | Use `config/per_label_constraints_recall_first.json` with `tune_thresholds_constrained.py`. |
| Tri-state manifest missing | Run `src/data_prep/create_phaseA_manifest_5k.py --keep_uncertain` (requires `phaseA_manifest.jsonl` from the radiology pipeline). |

---

For production, rerun the pipeline on your patient-wise split, keep the artefacts under `outputs_full/` (probabilities, thresholds, blend weights, meta calibration, DI logs) and deploy the Streamlit app to clinicians as a decision-support front end.

