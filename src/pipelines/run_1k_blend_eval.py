#!/usr/bin/env python3
"""
Run the full 1k pipeline (70/30 split): TXR + CheXagent linear probe → blend → meta-calibrate → thresholds → DI-gated eval.

This script is idempotent and writes into outputs_* directories; it does NOT touch
the long-running CheXagent process.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, desc: str):
    print("\n" + "=" * 88)
    print(f"🔧 {desc}")
    print("=" * 88)
    print(" ".join(map(str, cmd)))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"❌ Failed: {rc}")
        sys.exit(rc)


def main():
    ap = argparse.ArgumentParser(description="Orchestrate the 1k (70/30) blend+eval pipeline")
    ap.add_argument("--images", default="data/image_list_1000_absolute.txt")
    ap.add_argument("--manifest", default="data/evaluation_manifest_phaseA_matched.csv")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--image_root", default=None, help="Root directory for images when manifests contain relative paths.")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--out_root", default="outputs_1k")
    ap.add_argument("--chexagent_metadata", default="results/hybrid_ensemble_1000_improved.csv",
                    help="CSV with binary_outputs/di_outputs for gating")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    txr_dir = out_root / "txr"
    probe_dir = out_root / "linear_probe"
    blend_dir = out_root / "blend"
    calib_dir = out_root / "calibration"
    thrs_dir = out_root / "thresholds"
    final_dir = out_root / "final"

    for d in [txr_dir, probe_dir, blend_dir, calib_dir, thrs_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # TXR inference
    txr_preds = txr_dir / "txr_predictions_1000.csv"
    run([sys.executable, "src/inference/txr_infer.py", "--images", args.images, "--out_csv", str(txr_preds), "--device", args.device],
        "TXR inference (continuous probs)")

    # Split
    run([sys.executable, "src/data_prep/patient_wise_split.py", "--manifest", args.manifest, "--predictions", str(txr_preds),
         "--train_ratio", str(args.train_ratio), "--output_dir", "data"],
        "Patient-wise 70/30 split")

    gt_train = Path("data/ground_truth_train_70.csv")
    gt_test = Path("data/ground_truth_test_30.csv")

    # Prep calibration inputs
    run([sys.executable, "src/data_prep/prepare_predictions_for_calibration.py", "--predictions", str(txr_preds),
         "--ground_truth", str(gt_train), "--output", str(txr_dir / "train_txr_scores.csv")],
        "Prepare TXR train scores")
    run([sys.executable, "src/data_prep/prepare_predictions_for_calibration.py", "--predictions", str(txr_preds),
         "--ground_truth", str(gt_test), "--output", str(txr_dir / "test_txr_scores.csv")],
        "Prepare TXR test scores")

    # Fit/apply Platt
    run([sys.executable, "src/calibration/fit_label_calibrators.py", "--csv", str(txr_dir / "train_txr_scores.csv"),
         "--out_dir", str(txr_dir / "calibration_txr")],
        "Fit TXR Platt calibrators")
    run([sys.executable, "src/calibration/apply_label_calibrators.py", "--csv", str(txr_dir / "train_txr_scores.csv"),
         "--calib_dir", str(txr_dir / "calibration_txr"), "--out_csv", str(txr_dir / "train_txr_calibrated.csv")],
        "Apply TXR calibrators (train)")
    run([sys.executable, "src/calibration/apply_label_calibrators.py", "--csv", str(txr_dir / "test_txr_scores.csv"),
         "--calib_dir", str(txr_dir / "calibration_txr"), "--out_csv", str(txr_dir / "test_txr_calibrated.csv")],
        "Apply TXR calibrators (test)")

    # Linear probe embeddings (train/test)
    project_root = Path(__file__).parent.parent.parent
    image_root = Path(args.image_root) if args.image_root else None
    if image_root is None:
        default_root = project_root.parent / "radiology_report"
        if default_root.exists():
            image_root = default_root
   
    embed_cmd_train = [sys.executable, "src/embeddings/extract_chexagent_embeddings.py", 
                       "--images_csv", str(gt_train),
                       "--out_npz", str(probe_dir / "train_chexagent_cxr.npz"), 
                       "--device", args.device]
    if image_root:
        embed_cmd_train.extend(["--image_root", str(image_root)])
    
    run(embed_cmd_train, "Extract CheXagent embeddings (train)")
    
    embed_cmd_test = [sys.executable, "src/embeddings/extract_chexagent_embeddings.py",
                      "--images_csv", str(gt_test),
                      "--out_npz", str(probe_dir / "test_chexagent_cxr.npz"),
                      "--device", args.device]
    if image_root:
        embed_cmd_test.extend(["--image_root", str(image_root)])
    
    run(embed_cmd_test, "Extract CheXagent embeddings (test)")

    # Train probe + emit probs
    run([sys.executable, "src/models/train_linear_probe.py", "--train_npz", str(probe_dir / "train_chexagent_cxr.npz"),
         "--train_labels_csv", str(gt_train), "--out_dir", str(probe_dir),
         "--eval_split", f"test,{probe_dir / 'test_chexagent_cxr.npz'},{gt_test}"],
        "Train linear probe and score test")

    # Calibrate probe
    run([sys.executable, "src/calibration/platt_calibrate.py", "--train_probs_csv", str(probe_dir / "train_raw_probs.csv"),
         "--train_labels_csv", str(gt_train), "--out_params_json", str(calib_dir / "linear_probe_platt.json"),
         "--out_calibrated_train_csv", str(probe_dir / "train_calibrated.csv")],
        "Platt-calibrate probe (train)")
    run([sys.executable, "src/calibration/apply_platt.py", "--probs_csv", str(probe_dir / "test_raw_probs.csv"),
         "--params_json", str(calib_dir / "linear_probe_platt.json"), "--out_csv", str(probe_dir / "test_calibrated.csv")],
        "Apply probe calibration (test)")

    # Blend weights on train
    run([sys.executable, "src/blending/search_blend_weights.py", "--probs_csv", f"{txr_dir / 'train_txr_calibrated.csv'},txr",
         "--probs_csv", f"{probe_dir / 'train_calibrated.csv'},probe", "--labels_csv", str(gt_train), "--labels", "chexpert13",
         "--score_prefix", "y_cal_", "--metric", "fbeta", "--beta", "0.5", "--out_weights_json", str(blend_dir / "blend_weights.json"),
         "--out_blended_csv", str(blend_dir / "train_blended.csv")],
        "Search blend weights and create blended train")

    # Meta-calibrate blended train
    run([sys.executable, "src/calibration/meta_calibrate.py", "--blended_train_csv", str(blend_dir / "train_blended.csv"),
         "--labels_csv", str(gt_train), "--labels", "chexpert13", "--out_params_json", str(calib_dir / "meta_platt.json"),
         "--out_calibrated_train_csv", str(blend_dir / "train_blended_calibrated.csv")],
        "Meta-calibrate blended train")

    # Thresholds
    Path("config/minfloors.json").write_text("{\"_default_\": 0.45}\n")
    run([sys.executable, "src/thresholding/tune_thresholds.py", "--calibrated_train_csv", str(blend_dir / "train_blended_calibrated.csv"),
         "--labels", "chexpert13", "--score_prefix", "y_cal_", "--min_thresholds_json", "config/minfloors.json",
         "--out_thresholds_json", str(thrs_dir / "thresholds.json"), "--out_summary_csv", str(thrs_dir / "summary.csv")],
        "Tune thresholds on blended train")

    # Final test eval (DI-gated)
    run([sys.executable, "src/evaluation/run_test_eval.py",
         "--probs_csv", f"{txr_dir / 'test_txr_calibrated.csv'},txr",
         "--probs_csv", f"{probe_dir / 'test_calibrated.csv'},probe",
         "--blend_weights_json", str(blend_dir / "blend_weights.json"),
         "--meta_platt_json", str(calib_dir / "meta_platt.json"),
         "--thresholds_json", str(thrs_dir / "thresholds.json"),
         "--test_labels_csv", str(gt_test),
         "--labels", "chexpert13",
         "--score_prefix", "y_cal_",
         "--meta_prefix", "y_cal_",
         "--gating_config", "config/gating.json",
         "--metadata_csv", args.chexagent_metadata,
         "--out_probs_csv", str(final_dir / "test_probs.csv"),
         "--out_preds_csv", str(final_dir / "test_preds.csv"),
         "--out_metrics_csv", str(final_dir / "test_metrics.csv")],
        "Evaluate blended + DI-gated on test")

    print("\n✅ 1k pipeline complete. See:")
    print(f"  - {final_dir / 'test_metrics.csv'} (per-label)")
    print(f"  - {final_dir / 'test_probs.csv'} and {final_dir / 'test_preds.csv'}")
    print(f"  - {thrs_dir / 'thresholds.json'} (decisions)")


if __name__ == "__main__":
    main()
