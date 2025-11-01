#!/usr/bin/env python3
"""
Create manager-ready summary report combining metrics, predictions, and impressions.
"""

import argparse
import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime


def create_manager_report(
    metrics_csv: Path,
    predictions_csv: Path,
    thresholds_csv: Path,
    output_dir: Path,
) -> None:
    """
    Create comprehensive manager report.
    
    Args:
        metrics_csv: Per-label metrics CSV
        predictions_csv: Predictions with impressions CSV
        thresholds_csv: Threshold summary CSV
        output_dir: Output directory
    """
    print("=" * 80)
    print("CREATING MANAGER-READY REPORT")
    print("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n📂 Loading metrics: {metrics_csv}")
    metrics_df = pd.read_csv(metrics_csv)
    
    print(f"📂 Loading predictions: {predictions_csv}")
    pred_df = pd.read_csv(predictions_csv)
    
    thresholds_df = None
    if thresholds_csv.exists():
        print(f"📂 Loading thresholds: {thresholds_csv}")
        thresholds_df = pd.read_csv(thresholds_csv)
    
    # Calculate overall metrics
    macro_p = metrics_df["precision"].mean()
    macro_r = metrics_df["recall"].mean()
    macro_f1 = metrics_df["f1"].mean()
    
    # Micro metrics (weighted by support)
    total_tp = metrics_df["tp"].sum() if "tp" in metrics_df.columns else 0
    total_fp = metrics_df["fp"].sum() if "fp" in metrics_df.columns else 0
    total_fn = metrics_df["fn"].sum() if "fn" in metrics_df.columns else 0
    
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    
    # Generate markdown report
    report_md = output_dir / "MANAGER_REPORT.md"
    with open(report_md, "w") as f:
        f.write("# CheXpert Label Prediction - Manager Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Images Evaluated:** {len(pred_df)}\n")
        f.write(f"- **Macro Precision:** {macro_p:.3f}\n")
        f.write(f"- **Macro Recall:** {macro_r:.3f}\n")
        f.write(f"- **Macro F1-Score:** {macro_f1:.3f}\n")
        f.write(f"- **Micro Precision:** {micro_p:.3f}\n")
        f.write(f"- **Micro Recall:** {micro_r:.3f}\n")
        f.write(f"- **Micro F1-Score:** {micro_f1:.3f}\n\n")
        
        f.write("---\n\n")
        
        f.write("## Per-Label Performance\n\n")
        f.write("| Label | Precision | Recall | F1-Score |\n")
        f.write("|-------|-----------|--------|----------|\n")
        for _, row in metrics_df.iterrows():
            label = row["label"] if "label" in row else row.get(metrics_df.columns[0], "Unknown")
            p = row.get("precision", 0)
            r = row.get("recall", 0)
            f1 = row.get("f1", 0)
            f.write(f"| {label} | {p:.3f} | {r:.3f} | {f1:.3f} |\n")
        
        f.write("\n---\n\n")
        
        if thresholds_df is not None:
            f.write("## Threshold Decisions\n\n")
            f.write("| Label | Threshold |\n")
            f.write("|-------|-----------|\n")
            for _, row in thresholds_df.iterrows():
                label = row.get("label", "Unknown")
                thresh = row.get("threshold", "N/A")
                f.write(f"| {label} | {thresh} |\n")
            f.write("\n---\n\n")
        
        f.write("## Sample Predictions\n\n")
        sample_size = min(5, len(pred_df))
        for i, (_, row) in enumerate(pred_df.head(sample_size).iterrows(), 1):
            f.write(f"### Sample {i}\n\n")
            if "image" in row:
                f.write(f"**Image:** `{Path(row['image']).name}`\n\n")
            f.write(f"**Impression:** {row.get('impression', 'N/A')}\n\n")
            
            # List positive predictions
            CHEXPERT13 = [
                "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
                "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
                "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
            ]
            positives = []
            for label in CHEXPERT13:
                pred_val = row.get(label, 0)
                if isinstance(pred_val, (int, float)) and pred_val == 1:
                    positives.append(label)
            
            if positives:
                f.write(f"**Predicted Findings:** {', '.join(positives)}\n\n")
            else:
                f.write("**Predicted Findings:** No Finding\n\n")
            
            f.write("---\n\n")
    
    print(f"\n✅ Manager report saved to: {report_md}")
    
    # Generate summary CSV
    summary_csv = output_dir / "MANAGER_SUMMARY.csv"
    summary_data = {
        "metric": ["macro_precision", "macro_recall", "macro_f1", "micro_precision", "micro_recall", "micro_f1"],
        "value": [macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1]
    }
    pd.DataFrame(summary_data).to_csv(summary_csv, index=False)
    print(f"✅ Summary CSV saved to: {summary_csv}")
    
    # Sample predictions CSV
    sample_csv = output_dir / "sample_predictions.csv"
    sample_cols = ["filename", "impression"]
    CHEXPERT13 = [
        "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
        "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
        "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
    ]
    for col in sample_cols + CHEXPERT13 + ["No Finding"]:
        if col in pred_df.columns:
            sample_cols.append(col)
    
    pred_df[sample_cols].head(20).to_csv(sample_csv, index=False)
    print(f"✅ Sample predictions saved to: {sample_csv}")
    
    print("\n" + "=" * 80)
    print("MANAGER REPORT CREATION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Create manager-ready summary report")
    parser.add_argument(
        "--metrics_csv",
        type=Path,
        required=True,
        help="Per-label metrics CSV"
    )
    parser.add_argument(
        "--predictions_csv",
        type=Path,
        required=True,
        help="Predictions with impressions CSV"
    )
    parser.add_argument(
        "--thresholds_csv",
        type=Path,
        default=None,
        help="Threshold summary CSV (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs_5k/final"),
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    metrics_csv = (project_root / args.metrics_csv).resolve()
    predictions_csv = (project_root / args.predictions_csv).resolve()
    thresholds_csv = (project_root / args.thresholds_csv).resolve() if args.thresholds_csv else None
    output_dir = (project_root / args.output_dir).resolve()
    
    if not metrics_csv.exists():
        print(f"❌ Metrics CSV not found: {metrics_csv}")
        sys.exit(1)
    
    if not predictions_csv.exists():
        print(f"❌ Predictions CSV not found: {predictions_csv}")
        sys.exit(1)
    
    create_manager_report(metrics_csv, predictions_csv, thresholds_csv, output_dir)


if __name__ == "__main__":
    main()

