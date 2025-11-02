#!/usr/bin/env python3
"""
Run the full 5k pipeline (80/20 split): TXR + CheXagent linear probe → blend → meta-calibrate → thresholds → DI-gated eval → impressions.

This script is IDEMPOTENT: skips steps if outputs already exist.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import pandas as pd
from pathlib import Path


def run(cmd, desc: str, skip_if_exists: Path = None):
    """Run command with optional idempotency check."""
    if skip_if_exists and skip_if_exists.exists() and skip_if_exists.stat().st_size > 0:
        print(f"\n⏭️  Skipping: {desc} (already exists: {skip_if_exists})")
        return
    
    print("\n" + "=" * 88)
    print(f"🔧 {desc}")
    print("=" * 88)
    print(" ".join(map(str, cmd)))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"❌ Failed: {rc}")
        sys.exit(rc)


def main():
    ap = argparse.ArgumentParser(description="Orchestrate the 5k (80/20) blend+eval pipeline")
    ap.add_argument("--images", default="data/image_list_phaseA_5k_absolute.txt")
    ap.add_argument("--manifest", default="data/evaluation_manifest_phaseA_5k.csv")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--image_root", default=None, help="Root directory for images (default: ../radiology_report/files)")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--out_root", default="outputs_5k")
    ap.add_argument("--chexagent_metadata", default="results/hybrid_ensemble_5826.csv",
                    help="CSV with binary_outputs/di_outputs for gating")
    ap.add_argument("--resume", action="store_true", help="Resume partial runs (idempotent mode)")
    args = ap.parse_args()

    project_root = Path(__file__).parent.parent.parent
    out_root = project_root / args.out_root
    txr_dir = out_root / "txr"
    probe_dir = out_root / "linear_probe"
    blend_dir = out_root / "blend"
    calib_dir = out_root / "calibration"
    thrs_dir = out_root / "thresholds"
    splits_dir = out_root / "splits"
    final_dir = out_root / "final"

    for d in [txr_dir, probe_dir, blend_dir, calib_dir, thrs_dir, splits_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Resolve image root
    image_root = None
    if args.image_root:
        image_root = Path(args.image_root)
        if not image_root.is_absolute():
            image_root = project_root / image_root
    else:
        # Auto-detect: ../radiology_report/files
        default_root = project_root.parent / "radiology_report" / "files"
        if default_root.exists():
            image_root = default_root
    
    if image_root:
        image_root_str = str(image_root)
    else:
        image_root_str = None

    # Step 1: TXR inference (skip if exists)
    txr_preds = txr_dir / "txr_predictions_5k.csv"
    run([sys.executable, "src/inference/txr_infer.py", 
         "--images", str(project_root / args.images), 
         "--out_csv", str(txr_preds), 
         "--device", args.device],
        "TXR inference (continuous probs)",
        skip_if_exists=txr_preds if args.resume else None)

    # Step 2: Patient-wise split (80/20)
    gt_train = splits_dir / "train.csv"
    gt_test = splits_dir / "test.csv"
    
    if args.resume and gt_train.exists() and gt_test.exists():
        print(f"\n⏭️  Skipping patient-wise split (already exists)")
    else:
        # Use temporary output directory for patient_wise_split
        temp_split_dir = splits_dir / "temp"
        temp_split_dir.mkdir(exist_ok=True)
        
        run([sys.executable, "src/data_prep/patient_wise_split.py",
             "--manifest", str(project_root / args.manifest),
             "--predictions", str(txr_preds),
             "--train_ratio", str(args.train_ratio),
             "--output_dir", str(temp_split_dir),
             "--seed", "42"],
            f"Patient-wise {int(args.train_ratio*100)}/{int((1-args.train_ratio)*100)} split",
            skip_if_exists=None)
        
        # Move results to final location (patient_wise_split.py saves with hardcoded 70/30 names)
        import shutil
        # patient_wise_split.py saves as ground_truth_train_70.csv and ground_truth_test_30.csv
        # regardless of actual ratio, so we read those
        train_source = temp_split_dir / "ground_truth_train_70.csv"
        test_source = temp_split_dir / "ground_truth_test_30.csv"
        
        if train_source.exists():
            shutil.copy(str(train_source), str(gt_train))
            print(f"✅ Copied train split: {len(pd.read_csv(gt_train))} images")
        else:
            print(f"⚠️  Warning: Train split not found at {train_source}")
        
        if test_source.exists():
            shutil.copy(str(test_source), str(gt_test))
            print(f"✅ Copied test split: {len(pd.read_csv(gt_test))} images")
        else:
            print(f"⚠️  Warning: Test split not found at {test_source}")
        
        # Cleanup temp dir
        try:
            import shutil
            shutil.rmtree(temp_split_dir)
        except Exception as e:
            print(f"⚠️  Could not remove temp dir: {e}")
        
        print(f"\n✅ Patient-wise split saved to {splits_dir}/")

    # Step 3: Prep calibration inputs for TXR
    txr_train_scores = txr_dir / "train_txr_scores.csv"
    txr_test_scores = txr_dir / "test_txr_scores.csv"
    
    run([sys.executable, "src/data_prep/prepare_predictions_for_calibration.py",
         "--predictions", str(txr_preds),
         "--ground_truth", str(gt_train),
         "--output", str(txr_train_scores)],
        "Prepare TXR train scores",
        skip_if_exists=txr_train_scores if args.resume else None)
    
    run([sys.executable, "src/data_prep/prepare_predictions_for_calibration.py",
         "--predictions", str(txr_preds),
         "--ground_truth", str(gt_test),
         "--output", str(txr_test_scores)],
        "Prepare TXR test scores",
        skip_if_exists=txr_test_scores if args.resume else None)

    # Step 4: Fit/apply Platt calibration for TXR
    txr_calib_dir = txr_dir / "calibration_txr"
    txr_train_cal = txr_dir / "train_txr_calibrated.csv"
    txr_test_cal = txr_dir / "test_txr_calibrated.csv"
    
    run([sys.executable, "src/calibration/fit_label_calibrators.py",
         "--csv", str(txr_train_scores),
         "--out_dir", str(txr_calib_dir)],
        "Fit TXR Platt calibrators",
        skip_if_exists=txr_calib_dir / "platt_params.json" if args.resume else None)
    
    run([sys.executable, "src/calibration/apply_label_calibrators.py",
         "--csv", str(txr_train_scores),
         "--calib_dir", str(txr_calib_dir),
         "--out_csv", str(txr_train_cal)],
        "Apply TXR calibrators (train)",
        skip_if_exists=txr_train_cal if args.resume else None)
    
    run([sys.executable, "src/calibration/apply_label_calibrators.py",
         "--csv", str(txr_test_scores),
         "--calib_dir", str(txr_calib_dir),
         "--out_csv", str(txr_test_cal)],
        "Apply TXR calibrators (test)",
        skip_if_exists=txr_test_cal if args.resume else None)

    # Step 4b: Selective heavy TXR inference (5 labels with 6.8GB model)
    txr_heavy_raw = txr_dir / "txr_heavy_predictions.csv"
    run([sys.executable, "src/inference/txr_selective_infer.py",
         "--images", str(project_root / args.images),
         "--output", str(txr_heavy_raw),
         "--device", args.device,
         "--model_weights", "resnet50-res512-all",
         "--batch_size", "16",
         "--num_workers", "2"],
        "Selective TXR heavy inference (5-label ensemble)",
        skip_if_exists=txr_heavy_raw if args.resume else None)

    txr_heavy_train_scores = txr_dir / "train_txr_heavy_scores.csv"
    txr_heavy_test_scores = txr_dir / "test_txr_heavy_scores.csv"

    run([sys.executable, "src/data_prep/prepare_predictions_for_calibration.py",
         "--predictions", str(txr_heavy_raw),
         "--ground_truth", str(gt_train),
         "--output", str(txr_heavy_train_scores)],
        "Prepare TXR heavy train scores",
        skip_if_exists=txr_heavy_train_scores if args.resume else None)

    run([sys.executable, "src/data_prep/prepare_predictions_for_calibration.py",
         "--predictions", str(txr_heavy_raw),
         "--ground_truth", str(gt_test),
         "--output", str(txr_heavy_test_scores)],
        "Prepare TXR heavy test scores",
        skip_if_exists=txr_heavy_test_scores if args.resume else None)

    txr_heavy_calib_dir = txr_dir / "calibration_txr_heavy"
    txr_heavy_train_cal = txr_dir / "train_txr_heavy_calibrated.csv"
    txr_heavy_test_cal = txr_dir / "test_txr_heavy_calibrated.csv"

    run([sys.executable, "src/calibration/fit_label_calibrators.py",
         "--csv", str(txr_heavy_train_scores),
         "--out_dir", str(txr_heavy_calib_dir)],
        "Fit TXR heavy calibrators",
        skip_if_exists=txr_heavy_calib_dir / "platt_params.json" if args.resume else None)

    run([sys.executable, "src/calibration/apply_label_calibrators.py",
         "--csv", str(txr_heavy_train_scores),
         "--calib_dir", str(txr_heavy_calib_dir),
         "--out_csv", str(txr_heavy_train_cal)],
        "Apply TXR heavy calibrators (train)",
        skip_if_exists=txr_heavy_train_cal if args.resume else None)

    run([sys.executable, "src/calibration/apply_label_calibrators.py",
         "--csv", str(txr_heavy_test_scores),
         "--calib_dir", str(txr_heavy_calib_dir),
         "--out_csv", str(txr_heavy_test_cal)],
        "Apply TXR heavy calibrators (test)",
        skip_if_exists=txr_heavy_test_cal if args.resume else None)

    # Step 5: Linear probe embeddings (train/test)
    embed_train_npz = probe_dir / "train_chexagent_cxr.npz"
    embed_test_npz = probe_dir / "test_chexagent_cxr.npz"
    
    embed_cmd_train = [sys.executable, "src/embeddings/extract_chexagent_embeddings.py",
                       "--images_csv", str(gt_train),
                       "--out_npz", str(embed_train_npz),
                       "--device", args.device]
    if image_root_str:
        embed_cmd_train.extend(["--image_root", image_root_str])
    
    run(embed_cmd_train, "Extract CheXagent embeddings (train)",
        skip_if_exists=embed_train_npz if args.resume else None)
    
    embed_cmd_test = [sys.executable, "src/embeddings/extract_chexagent_embeddings.py",
                      "--images_csv", str(gt_test),
                      "--out_npz", str(embed_test_npz),
                      "--device", args.device]
    if image_root_str:
        embed_cmd_test.extend(["--image_root", image_root_str])
    
    run(embed_cmd_test, "Extract CheXagent embeddings (test)",
        skip_if_exists=embed_test_npz if args.resume else None)

    # Step 6: Train linear probe + emit probs
    probe_train_raw = probe_dir / "train_raw_probs.csv"
    probe_test_raw = probe_dir / "test_raw_probs.csv"
    
    run([sys.executable, "src/models/train_linear_probe.py",
         "--train_npz", str(embed_train_npz),
         "--train_labels_csv", str(gt_train),
         "--out_dir", str(probe_dir),
         "--eval_split", f"test,{embed_test_npz},{gt_test}"],
        "Train linear probe and score test",
        skip_if_exists=probe_test_raw if args.resume else None)

    # Step 7: Calibrate probe
    probe_platt_json = calib_dir / "linear_probe_platt.json"
    probe_train_cal = probe_dir / "train_calibrated.csv"
    probe_test_cal = probe_dir / "test_calibrated.csv"
    
    run([sys.executable, "src/calibration/platt_calibrate.py",
         "--train_probs_csv", str(probe_train_raw),
         "--train_labels_csv", str(gt_train),
         "--out_params_json", str(probe_platt_json),
         "--out_calibrated_train_csv", str(probe_train_cal)],
        "Platt-calibrate probe (train)",
        skip_if_exists=probe_train_cal if args.resume else None)
    
    run([sys.executable, "src/calibration/apply_platt.py",
         "--probs_csv", str(probe_test_raw),
         "--params_json", str(probe_platt_json),
         "--out_csv", str(probe_test_cal)],
        "Apply probe calibration (test)",
        skip_if_exists=probe_test_cal if args.resume else None)

    # Step 8: Blend weights on train (precision-weighted: beta=0.3)
    blend_weights_json = blend_dir / "blend_weights.json"
    blend_train_csv = blend_dir / "train_blended.csv"
    
    run([sys.executable, "src/blending/search_blend_weights.py",
         "--probs_csv", f"{txr_train_cal},txr",
         "--probs_csv", f"{probe_train_cal},probe",
         "--probs_csv", f"{txr_heavy_train_cal},txr_heavy",
         "--labels_csv", str(gt_train),
         "--labels", "chexpert13",
         "--score_prefix", "y_cal_",
         "--metric", "fbeta",
         "--beta", "0.3",  # Precision-weighted
         "--out_weights_json", str(blend_weights_json),
         "--out_blended_csv", str(blend_train_csv)],
        "Search blend weights (precision-weighted β=0.3) and create blended train",
        skip_if_exists=blend_train_csv if args.resume else None)

    # Step 9: Meta-calibrate blended train
    meta_platt_json = calib_dir / "meta_platt.json"
    blend_train_cal = blend_dir / "train_blended_calibrated.csv"
    
    run([sys.executable, "src/calibration/meta_calibrate.py",
         "--blended_train_csv", str(blend_train_csv),
         "--labels_csv", str(gt_train),
         "--labels", "chexpert13",
         "--out_params_json", str(meta_platt_json),
         "--out_calibrated_train_csv", str(blend_train_cal)],
        "Meta-calibrate blended train",
        skip_if_exists=blend_train_cal if args.resume else None)

    # Step 10: Threshold tuning (with floors)
    minfloors_json = project_root / "config" / "minfloors.json"
    if not minfloors_json.exists():
        minfloors_json.parent.mkdir(exist_ok=True)
        minfloors_json.write_text('{"_default_":0.45,"Fracture":0.65,"Pleural Other":0.70,"Lung Lesion":0.60,"Consolidation":0.60,"Pneumonia":0.55}\n')
    
    thresholds_json = thrs_dir / "thresholds.json"
    
    run([sys.executable, "src/thresholding/tune_thresholds.py",
         "--calibrated_train_csv", str(blend_train_cal),
         "--labels", "chexpert13",
         "--score_prefix", "y_cal_",
         "--min_thresholds_json", str(minfloors_json),
         "--out_thresholds_json", str(thresholds_json),
         "--out_summary_csv", str(thrs_dir / "summary.csv")],
        "Tune thresholds on blended train (with floors)",
        skip_if_exists=thresholds_json if args.resume else None)

    # Step 11: Final test eval (DI-gated)
    test_probs_csv = final_dir / "test_probs.csv"
    test_preds_csv = final_dir / "test_preds.csv"
    test_metrics_csv = final_dir / "test_metrics.csv"
    
    run([sys.executable, "src/evaluation/run_test_eval.py",
         "--probs_csv", f"{txr_test_cal},txr",
         "--probs_csv", f"{probe_test_cal},probe",
         "--probs_csv", f"{txr_heavy_test_cal},txr_heavy",
         "--blend_weights_json", str(blend_weights_json),
         "--meta_platt_json", str(meta_platt_json),
         "--thresholds_json", str(thresholds_json),
         "--test_labels_csv", str(gt_test),
         "--labels", "chexpert13",
         "--score_prefix", "y_cal_",
         "--meta_prefix", "y_cal_",
         "--gating_config", str(project_root / "config" / "gating.json"),
         "--metadata_csv", str(project_root / args.chexagent_metadata),
         "--out_probs_csv", str(test_probs_csv),
         "--out_preds_csv", str(test_preds_csv),
         "--out_metrics_csv", str(test_metrics_csv)],
        "Evaluate blended + DI-gated on test",
        skip_if_exists=test_metrics_csv if args.resume else None)

    # Step 12: Generate impressions
    test_with_impressions_csv = final_dir / "test_with_impressions.csv"
    
    run([sys.executable, "src/utils/generate_impressions_from_di.py",
         "--chexagent_csv", str(project_root / args.chexagent_metadata),
         "--predictions_csv", str(test_preds_csv),
         "--output", str(test_with_impressions_csv)],
        "Generate impressions from CheXagent DI outputs",
        skip_if_exists=test_with_impressions_csv if args.resume else None)

    # Step 13: Create manager report
    manager_report_md = final_dir / "MANAGER_REPORT.md"
    
    run([sys.executable, "src/utils/create_manager_report.py",
         "--metrics_csv", str(test_metrics_csv),
         "--predictions_csv", str(test_with_impressions_csv),
         "--thresholds_csv", str(thrs_dir / "summary.csv"),
         "--output_dir", str(final_dir)],
        "Create manager-ready summary report",
        skip_if_exists=manager_report_md if args.resume else None)

    print("\n" + "=" * 88)
    print("✅ 5K PIPELINE COMPLETE")
    print("=" * 88)
    print(f"\n📊 Outputs:")
    print(f"  - Metrics: {test_metrics_csv}")
    print(f"  - Predictions: {test_preds_csv}")
    print(f"  - With Impressions: {test_with_impressions_csv}")
    print(f"  - Manager Report: {manager_report_md}")
    print(f"  - Thresholds: {thresholds_json}")


if __name__ == "__main__":
    main()
