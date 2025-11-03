#!/usr/bin/env bash
set -euo pipefail

export COMPUTE_TARGET=STUDIO
export PYTHONUNBUFFERED=1

OUT=outputs_full_final
LOG="$OUT/logs"
mkdir -p "$LOG" "$OUT/txr" "$OUT/final" "$OUT/calibration" "$OUT/thresholds"

PYTHON="${PYTHON:-./venv/bin/python}"
META_JSON="$OUT/calibration/meta_platt.json"

echo "[1/7] TXR-HEAVY selective inference (deferred)" | tee "$LOG/01_txr_heavy.log"
echo "Heavy TXR selective heads are disabled for this run; continuing with base TXR + probe + CheXagent ensemble." | tee -a "$LOG/01_txr_heavy.log"

echo "[2/7] Blend + meta-cal + thresholds (Certain-Only)" | tee "$LOG/02_blend_thresholds.log"
$PYTHON -u src/pipelines/run_5k_blend_eval.py \
  --images data/image_list_phaseA_full_absolute.txt \
  --manifest data/evaluation_manifest_phaseA_full_abs.csv \
  --device mps \
  --out_root "$OUT" \
  --chexagent_metadata results/hybrid_ensemble_5826.csv \
  --resume | tee -a "$LOG/02_blend_thresholds.log"

echo "[3/7] Evaluate Certain-Only & Binary" | tee "$LOG/03_eval.log"

# Find test probabilities from blend step
TXR_TEST="${OUT}/txr/test_txr_calibrated.csv"
PROBE_TEST="${OUT}/linear_probe/test_calibrated.csv"
BLEND_WEIGHTS="${OUT}/blend/blend_weights.json"
THRESHOLDS="${OUT}/thresholds/thresholds_90pct_target.json"

# Use updated gating script logic to find files
if [ ! -f "$TXR_TEST" ]; then
    TXR_TEST=$(find "$OUT" -name "*txr*test*csv" -type f | head -1)
fi
if [ ! -f "$PROBE_TEST" ]; then
    PROBE_TEST=$(find "$OUT" -name "*probe*test*csv" -type f | head -1)
fi
if [ ! -f "$THRESHOLDS" ]; then
    THRESHOLDS=$(find "$OUT/thresholds" -name "*.json" -type f | head -1)
fi

$PYTHON -u src/evaluation/run_test_eval.py \
  --probs_csv "${TXR_TEST},txr" \
  --probs_csv "${PROBE_TEST},probe" \
  --blend_weights_json "$BLEND_WEIGHTS" \
  --meta_platt_json "$META_JSON" \
  --thresholds_json "$THRESHOLDS" \
  --test_labels_csv "$OUT/splits/ground_truth_test.csv" \
  --labels chexpert13 \
  --score_prefix y_cal_ \
  --meta_prefix y_cal_ \
  --eval_mode certain_only \
  --gating_config config/gating.json \
  --metadata_csv results/hybrid_ensemble_5826.csv \
  --skip_meta_calibration \
  --out_probs_csv "$OUT/final/test_probs_certain.csv" \
  --out_preds_csv "$OUT/final/test_preds_certain.csv" \
  --out_metrics_csv "$OUT/final/test_metrics_certain.csv" | tee -a "$LOG/03_eval.log"

$PYTHON -u src/evaluation/run_test_eval.py \
  --probs_csv "${TXR_TEST},txr" \
  --probs_csv "${PROBE_TEST},probe" \
  --blend_weights_json "$BLEND_WEIGHTS" \
  --meta_platt_json "$META_JSON" \
  --thresholds_json "$THRESHOLDS" \
  --test_labels_csv "$OUT/splits/ground_truth_test.csv" \
  --labels chexpert13 \
  --score_prefix y_cal_ \
  --meta_prefix y_cal_ \
  --eval_mode binary \
  --gating_config config/gating.json \
  --metadata_csv results/hybrid_ensemble_5826.csv \
  --skip_meta_calibration \
  --out_probs_csv "$OUT/final/test_probs_binary.csv" \
  --out_preds_csv "$OUT/final/test_preds_binary.csv" \
  --out_metrics_csv "$OUT/final/test_metrics_binary.csv" | tee -a "$LOG/03_eval.log"

cp "$OUT/final/test_metrics_certain.macro_micro.json" "$OUT/final/test_metrics.macro_micro.json" 2>/dev/null || true

echo "[4/7] Three-class evaluation" | tee "$LOG/04_threeclass.log"
$PYTHON -u src/evaluation/evaluate_three_class.py \
  --predictions "$OUT/final/test_preds_certain.csv" \
  --ground_truth "$OUT/splits/ground_truth_test.csv" \
  --output "$OUT/final/test_metrics_three_class.csv" | tee -a "$LOG/04_threeclass.log" || true

echo "[5/7] Manager report with correct Micro/Macro" | tee "$LOG/05_report.log"
$PYTHON -u src/reporting/create_manager_report.py \
  --metrics_certain "$OUT/final/test_metrics_certain.csv" \
  --metrics_binary "$OUT/final/test_metrics_binary.csv" \
  --macro_micro_json "$OUT/final/test_metrics_certain.macro_micro.json" \
  --macro_micro_json_binary "$OUT/final/test_metrics_binary.macro_micro.json" \
  --out_md "$OUT/final/MANAGER_REPORT.md" \
  --out_csv "$OUT/final/MANAGER_REPORT.csv" | tee -a "$LOG/05_report.log"

echo "[6/7] Sanity checks" | tee "$LOG/06_sanity.log"
$PYTHON -u scripts/sanity_checks.py \
  --preds "$OUT/final/test_preds_certain.csv" \
  --labels "$OUT/splits/ground_truth_test.csv" \
  --out "$OUT/final/SANITY_SUMMARY.md" | tee -a "$LOG/06_sanity.log" || true

echo "[7/7] Streamlit ready (you can preview on laptop too)" | tee "$LOG/07_streamlit.log"
echo "Run:  streamlit run streamlit/app_demo_chexagent.py" | tee -a "$LOG/07_streamlit.log"

echo "DONE."
