# 5k Pipeline Setup and Execution Plan (Final)

## Goals

- Copy server CheXagent results (5,826 images) to local
- Convert server paths to local paths
- Create 5k image list and Phase-A manifest
- Run full **TXR + CheXagent (linear probe) blend** with **high precision/recall**
- Fix **idempotency** (no unnecessary re-runs)
- Optimize for **high precision AND recall**
- **Generate impressions** from CheXagent for manager-ready output
- Produce **manager-ready metrics + artifacts**

## Lessons from Past Results (100/1000 images)

**Key Issues Identified:**
- ❌ **Low precision** (0.075-0.295 macro) - too many false positives
- ❌ **Only 5-6 labels working** - DI gating too strict, blocking many labels
- ❌ **Many labels with 0.0 metrics** - need better threshold tuning and gating

**Improvements in This Plan:**
- ✅ **Linear probe** provides continuous probabilities (fixes 0.2/0.8 clustering)
- ✅ **Per-label blend weights** (TXR weight = 0.0 for Support Devices/Pleural Other)
- ✅ **Relaxed DI gating** (hard_di_min: 0.55, easy_di_min: 0.50)
- ✅ **Precision-first threshold tuning** with floors
- ✅ **High-prob override** (≥0.80 bypasses DI for easy labels)
- ✅ **No Finding rule** enforced correctly

## 0) One-time Schema & Safety Checklist

- **Join key:** standardize to a single column `filename` across *all* CSVs (TXR, CheXagent, probe, blends)
- **Patient-wise leakage guard:** assert `|patients(train) ∩ patients(test)| = 0` (fail fast if violated)
- **Central labels:** `src/common/labels.py` with `CHEXPERT13`, `CHEXPERT14`, and `SCORE_PREFIX = {"raw":"y_pred_", "cal":"y_cal_"}`
- **No Finding policy:** enforce **NoFinding = 1 iff all other 13 = 0** right before metrics (do not threshold NoFinding independently)
- **NaN guards:** if a label has `< 15` positives in train: skip Platt for that label (identity mapping) and pin threshold to a floor (e.g., 0.5 or min-floor). Log clearly.
- **Image root resolution:** handle `radiology_report/files/p10` which is outside project parent (`../radiology_report/files/p10`)

## Step 1: Copy and Convert Server CSV

**Script**: `src/utils/copy_server_results.py`

- SSH copy `chex_full/hybrid_full.csv` from server
- Convert paths: `/Users/bilbouser/radiology_report/...` → `/Users/rahul/Downloads/Code scripts/radiology_report/...`
- Save to `results/hybrid_ensemble_5826.csv`
- **Assert columns present:** `filename`, `binary_outputs`, `di_outputs` (fail fast with message)

## Step 2: Extract Image List from CheXagent CSV

**Script**: `src/data_prep/extract_image_list_from_csv.py`

- Read `results/hybrid_ensemble_5826.csv`
- Extract image column → convert to local absolute paths
- **Handle image root:** resolve `../radiology_report/files/p10` correctly
- Write `data/image_list_phaseA_5k_absolute.txt`
- **Validate existence** of every image; if missing, write `results/missing_files.txt` and stop

## Step 3: Create Phase-A Manifest for 5k Images

**Script**: `src/data_prep/create_phaseA_manifest_5k.py`

- Load `radiology_report/src/data/processed/phaseA_manifest.jsonl`
- Match **5k** image filenames from CheXagent CSV
- Convert CheXpert labels `-1/0/1 → 0/1` (log counts per label)
- Save `data/evaluation_manifest_phaseA_5k.csv` (all 14 labels present)
- **Recompute No Finding** after label normalization

## Step 3b: Patient-wise Split (train/test 80/20)

**Script**: `src/data_prep/patientwise_split.py`

- Input: `data/evaluation_manifest_phaseA_5k.csv`
- Output:
    - `outputs_5k/splits/train.csv`
    - `outputs_5k/splits/test.csv`
- Deterministic seed, **no leakage** (assert and print patient overlap = 0)
- Print label counts per split (positive/negative)

## Step 3c: CheXagent Embeddings & Linear Probe (fills TXR head gaps)

**Script A (extract):** `src/embeddings/extract_chexagent_embeddings.py`

- Args: `--images_csv`, `--out_npz`, `--device mps|cuda|cpu`, `--image_root ../radiology_report/files` (relative to project root)
- **Image root resolution:** automatically detect `../radiology_report/files` if exists, else use provided `--image_root`
- Output: `outputs_5k/embeddings/{train|test}_chexagent_cxr.npz` (keys: `filenames`, `embeddings`)

**Script B (train):** `src/models/train_linear_probe.py`

- Train **13** one-vs-rest `LogisticRegression(class_weight='balanced', max_iter=1000)` on **train** embeddings → save `linear_probe.pkl`
- Write **raw probabilities**:
    - `outputs_5k/linear_probe/train_raw_probs.csv`
    - `outputs_5k/linear_probe/test_raw_probs.csv` (aligned by `filename`)

**Script C (Platt on probe):**

- `src/calibration/platt_calibrate.py` → fit per-label Platt on **train** raw probs → outputs:
    - `outputs_5k/calibration/linear_probe_platt.json`
    - `outputs_5k/linear_probe/train_calibrated.csv`
- `src/calibration/apply_platt.py` → apply same params to **test**:
    - `outputs_5k/linear_probe/test_calibrated.csv`

This gives **continuous probabilities for all 13 labels** from CheXagent, fixing the 0.2/0.8 discretization and removing dependency on TXR for coverage gaps.

## Step 4: Create 5k Pipeline Orchestrator (idempotent)

**Script**: `src/pipelines/run_5k_blend_eval.py` (new; based on `run_1k_blend_eval.py`)

- Orchestrates: **TXR (skip if exists) → Linear Probe (skip if exists) → Blend weight search → Meta-calibration → Threshold tuning → Test eval with gating → Impression generation**
- **Idempotency checks** at each stage (if output exists and non-empty, print "⏭️ Skipped … (exists)")
- **Args:**
    - `--images data/image_list_phaseA_5k_absolute.txt`
    - `--manifest data/evaluation_manifest_phaseA_5k.csv`
    - `--device mps`
    - `--out_root outputs_5k`
    - `--chexagent_metadata results/hybrid_ensemble_5826.csv`
    - `--train_ratio 0.8`
    - `--image_root ../radiology_report/files` (auto-detected if not provided)
    - `--resume` (resume partial runs without re-starting earlier stages)

**Idempotency Example:**
```python
if txr_train.exists() and txr_train.stat().st_size > 0:
    print("⏭️  Skipping TXR train (already exists)")
else:
    run_txr_train(...)
```

## Step 5: Optimize Gating Config for High P+R

**File**: `config/gating.json`

```json
{
  "hard_labels": ["Fracture","Lung Lesion","Pleural Other","Consolidation","Pneumonia","Enlarged Cardiomediastinum"],
  "easy_labels": ["Pleural Effusion","Edema","Lung Opacity","Pneumothorax","Support Devices","Atelectasis","Cardiomegaly"],
  "rules": {
    "di_min_hard": 0.55,
    "di_min_easy": 0.50,
    "rescue_margin": 0.05,
    "below_threshold_rescue_hard": true,
    "below_threshold_rescue_easy": true,
    "high_prob_override": 0.80,
    "high_prob_bypass_di_easy": true,
    "consistency": {
      "Pneumonia_requires_one_of": ["Lung Opacity","Consolidation"],
      "apply_if_prob_below": 0.70
    }
  }
}
```

- **Order:** Threshold first → **then** apply DI gating tweaks (log any flips)
- **High-prob override:** if calibrated prob ≥ 0.80, allow bypass of DI for easy labels only

## Step 6: Run Pipeline

```bash
python src/pipelines/run_5k_blend_eval.py \
  --images data/image_list_phaseA_5k_absolute.txt \
  --manifest data/evaluation_manifest_phaseA_5k.csv \
  --device mps \
  --out_root outputs_5k \
  --chexagent_metadata results/hybrid_ensemble_5826.csv \
  --train_ratio 0.8 \
  --image_root ../radiology_report/files \
  --resume
```

## Step 7: Per-label Blend Weight Search + Meta-Calibration + Thresholds

These are inside the orchestrator, but list them explicitly for clarity:

- **Blend weight search (train side)**: `src/blending/search_blend_weights.py`
    - Inputs: `outputs_5k/txr/train_calibrated.csv` (if present), `outputs_5k/linear_probe/train_calibrated.csv`
    - Objective: `--metric fbeta --beta 0.3` *(precision-weighted: β=0.3 for higher precision)*
    - Special cases:
        - **Support Devices**, **Pleural Other** → TXR weight = **0.0** (until TXR provides these heads)
    - Output: `outputs_5k/blend/blend_weights.json`, `outputs_5k/blend/train_blended.csv`
- **Meta-calibration on blended train**: `src/calibration/meta_calibrate.py`
    - Output: `outputs_5k/calibration/meta_platt.json`, `outputs_5k/blend/train_blended_calibrated.csv`
- **Threshold tuning (with floors & optional prevalence guard)**: `src/thresholds/threshold_tuner.py`
    - Floors file:
```bash
echo '{"_default_":0.45,"Fracture":0.65,"Pleural Other":0.70,"Lung Lesion":0.60,"Consolidation":0.60,"Pneumonia":0.55}' > config/minfloors.json
```
    - Optional prevalence guard: allow `--target_prev_json` per label with ±0.10 tolerance
    - Output: `outputs_5k/thresholds/thresholds.json`, `outputs_5k/thresholds/summary.csv`
    - **Note:** tune after meta-calibration only

## Step 8: Final Test Evaluation with DI Gating

**Script**: `src/evaluation/run_test_eval.py` (existing; extend to support blend→meta→threshold→gating)

- Inputs:
    - `outputs_5k/txr/test_calibrated.csv` (optional)
    - `outputs_5k/linear_probe/test_calibrated.csv`
    - `outputs_5k/blend/blend_weights.json`
    - `outputs_5k/calibration/meta_platt.json`
    - `outputs_5k/thresholds/thresholds.json`
    - `outputs_5k/splits/test.csv` (ground truth)
    - `config/gating.json`
    - `results/hybrid_ensemble_5826.csv` (for DI JSON)
- Outputs:
    - `outputs_5k/final/test_probs.csv`
    - `outputs_5k/final/test_preds.csv`
    - `outputs_5k/final/test_metrics.csv`
    - **Extras**: PR-AUC per label, reliability (ECE), and Top-50 FP/FN CSV (filename, GT, pred, DI snippet)
- **Apply No Finding rule** at the very end (set to 1 if all other 13 = 0)

## Step 9: Generate Impressions from CheXagent (Manager-Ready Output)

**Script**: `src/utils/generate_impressions_from_di.py` (new)

- Read `results/hybrid_ensemble_5826.csv` and `outputs_5k/final/test_preds.csv`
- Extract `di_outputs` JSON (contains narrative text from `disease_identification`)
- **Format impression:** Use DI narrative text, or if missing, generate from predicted labels:
  ```
  "Based on the chest X-ray analysis, findings include: [list positive labels]. 
  [DI narrative text if available]"
  ```
- Save to `outputs_5k/final/test_with_impressions.csv`:
  - Columns: `filename`, `image`, `impression`, `[13 CheXpert labels]`, `No Finding`, `[13 predicted labels]`
- **Use CheXagent's `disease_identification` output** as the impression source (already generated, just extract)

## Step 10: Create Manager-Ready Summary Report

**Script**: `src/utils/create_manager_report.py` (new)

- Combine:
    - `outputs_5k/final/test_metrics.csv` (per-label metrics)
    - `outputs_5k/final/test_with_impressions.csv` (predictions + impressions)
    - `outputs_5k/thresholds/summary.csv` (threshold decisions)
- Generate:
    - `outputs_5k/final/MANAGER_REPORT.md` (markdown summary)
    - `outputs_5k/final/MANAGER_SUMMARY.csv` (CSV with key metrics)
    - `outputs_5k/final/sample_predictions.csv` (top 20 examples with impressions)
- **Include:**
    - Overall macro/micro P, R, F1
    - Per-label metrics (table)
    - Comparison to baseline (if available)
    - Sample predictions with impressions
    - Visual summary (optional: PR curves per label)

## Key Fixes (recap)

1. **Idempotency everywhere**: each stage skips if outputs exist (plus `--resume`)
2. **Path conversion**: robust server→local mapping & file-existence checks
3. **Image root resolution**: handle `../radiology_report/files` correctly
4. **Continuous probabilities**: CheXagent **linear probe** provides smooth probs for all 13 labels
5. **Per-label blending** + **meta-calibration** **before** threshold tuning
6. **Thresholds with floors** (and optional prevalence guard) to avoid 0.20 collapse
7. **DI gating via config**: hard vs easy rules, margins, and a Pneumonia consistency rule
8. **Relaxed DI requirements**: hard_di_min: 0.55, easy_di_min: 0.50 (based on past issues)
9. **High-prob override**: ≥0.80 bypasses DI for easy labels (prevents blocking high-confidence predictions)
10. **Precision-weighted tuning**: β=0.3 for F-beta optimization (emphasizes precision)
11. **Impression generation**: Extract from CheXagent `di_outputs` (already available)
12. **Manager-ready output**: Combined metrics + impressions + sample predictions

## Expected Outcomes

- **High precision**: Macro P ≥ 0.50 (hard tail ≥ 0.40)
- **High recall**: Macro R ≥ 0.70 (sentinel labels ≥ 0.85)
- **Balanced F1**: Macro F1 ≥ 0.60
- **All labels working**: No zeros (at least some predictions per label)
- **No unnecessary reruns**: Pipeline skips completed steps
- **Manager-ready**: Impressions + predictions + metrics in one CSV

