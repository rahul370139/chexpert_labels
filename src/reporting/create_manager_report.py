#!/usr/bin/env python3
"""Generate manager-facing report combining certain-only and binary evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


SUMMARY_KEYS = [
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_accuracy",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "micro_accuracy",
]


def load_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError(f"Metrics file {path} is missing 'label' column")
    return df


def load_summary(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text())
    summary: Dict[str, float] = {}
    for key in SUMMARY_KEYS:
        value = data.get(key)
        if value is not None:
            summary[key] = float(value)
    summary["eval_mode"] = data.get("eval_mode", "unknown")
    return summary


def format_metric(value: Optional[float], fallback: str = "-") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return fallback
    return f"{value:.3f}"


def build_per_label_table(metrics_certain: pd.DataFrame, metrics_binary: Optional[pd.DataFrame]) -> pd.DataFrame:
    renamed_certain = metrics_certain.rename(columns={
        "precision": "precision_certain",
        "recall": "recall_certain",
        "f1": "f1_certain",
        "accuracy": "accuracy_certain",
        "coverage": "coverage_certain",
        "total": "total_certain",
        "threshold": "threshold",
        "gt_positives": "positives_certain",
        "gt_negatives": "negatives_certain",
        "pred_positives": "predicted_pos_certain",
    })

    if metrics_binary is not None:
        renamed_binary = metrics_binary.rename(columns={
            "precision": "precision_binary",
            "recall": "recall_binary",
            "f1": "f1_binary",
            "accuracy": "accuracy_binary",
            "coverage": "coverage_binary",
            "total": "total_binary",
            "gt_positives": "positives_binary",
            "gt_negatives": "negatives_binary",
            "pred_positives": "predicted_pos_binary",
        })
        merged = pd.merge(renamed_certain, renamed_binary, on="label", how="left")
    else:
        merged = renamed_certain.copy()

    merged["coverage_rate"] = merged.apply(
        lambda row: row["coverage_certain"] / row["total_certain"] if row.get("total_certain") else np.nan,
        axis=1,
    )
    merged["prevalence"] = merged.apply(
        lambda row: row["positives_certain"] / row["coverage_certain"] if row.get("coverage_certain") else np.nan,
        axis=1,
    )
    return merged


def render_per_label_markdown(df: pd.DataFrame) -> str:
    headers = [
        "Label",
        "P",
        "R",
        "F1",
        "Accuracy",
        "Coverage",
        "Prevalence",
        "τ",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, row in df.iterrows():
        precision_raw = row.get("precision_certain")
        recall_raw = row.get("recall_certain")
        f1 = row.get("f1_certain")
        positives = row.get("positives_certain", 0)
        predicted_pos = row.get("predicted_pos_certain", 0)
        
        # Handle precision: show "n/a (0 pos)" when TP = FP = 0 (predicted_pos = 0)
        if positives == 0 and predicted_pos == 0:
            precision = "n/a (0 pos)"
        elif isinstance(precision_raw, float) and np.isnan(precision_raw):
            precision = "n/a (0 pos)"
        else:
            precision = format_metric(precision_raw)
        
        recall = format_metric(recall_raw)
        
        if positives == 0 or (isinstance(f1, float) and np.isnan(f1)):
            f1_display = "n/a (0 positives)"
        else:
            f1_display = format_metric(f1)
        accuracy = format_metric(row.get("accuracy_certain"))
        coverage = row.get("coverage_certain")
        total = row.get("total_certain")
        coverage_display = f"{coverage}/{total}" if pd.notna(coverage) and pd.notna(total) else "-"
        prevalence_display = format_metric(row.get("prevalence"))
        tau_display = format_metric(row.get("threshold"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    precision,
                    recall,
                    f1_display,
                    accuracy,
                    coverage_display,
                    prevalence_display,
                    tau_display,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create consolidated manager report")
    parser.add_argument("--metrics_certain", type=Path, required=True)
    parser.add_argument("--metrics_binary", type=Path, required=True)
    parser.add_argument("--macro_micro_json", type=Path, required=True, help="Summary JSON for certain-only eval")
    parser.add_argument("--macro_micro_json_binary", type=Path, default=None, help="Summary JSON for binary eval")
    parser.add_argument("--out_md", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    args = parser.parse_args()

    metrics_certain = load_metrics(args.metrics_certain)
    metrics_binary = load_metrics(args.metrics_binary) if args.metrics_binary.exists() else None

    summary_certain = load_summary(args.macro_micro_json)
    summary_binary = load_summary(args.macro_micro_json_binary) if args.macro_micro_json_binary and args.macro_micro_json_binary.exists() else None

    per_label = build_per_label_table(metrics_certain, metrics_binary)

    # Markdown report
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with args.out_md.open("w", encoding="utf-8") as md:
        md.write("# CheXpert Hybrid Evaluation Report\n\n")
        md.write("## Executive Summary (Certain-Only)\n\n")
        md.write("### Macro Metrics (excluding No Finding)\n\n")
        md.write(f"- Precision: {format_metric(summary_certain.get('macro_precision'))}\n")
        md.write(f"- Recall: {format_metric(summary_certain.get('macro_recall'))}\n")
        md.write(f"- F1: {format_metric(summary_certain.get('macro_f1'))}\n")
        md.write(f"- Accuracy: {format_metric(summary_certain.get('macro_accuracy'))}\n\n")
        md.write("### Micro Metrics (excluding No Finding)\n\n")
        md.write(f"- Precision: {format_metric(summary_certain.get('micro_precision'))}\n")
        md.write(f"- Recall: {format_metric(summary_certain.get('micro_recall'))}\n")
        md.write(f"- F1: {format_metric(summary_certain.get('micro_f1'))}\n")
        md.write(f"- Accuracy: {format_metric(summary_certain.get('micro_accuracy'))}\n\n")

        if summary_binary:
            md.write("## Binary Summary (−1 → 0)\n\n")
            md.write("### Macro Metrics\n\n")
            md.write(f"- Precision: {format_metric(summary_binary.get('macro_precision'))}\n")
            md.write(f"- Recall: {format_metric(summary_binary.get('macro_recall'))}\n")
            md.write(f"- F1: {format_metric(summary_binary.get('macro_f1'))}\n")
            md.write(f"- Accuracy: {format_metric(summary_binary.get('macro_accuracy'))}\n\n")
            md.write("### Micro Metrics\n\n")
            md.write(f"- Precision: {format_metric(summary_binary.get('micro_precision'))}\n")
            md.write(f"- Recall: {format_metric(summary_binary.get('micro_recall'))}\n")
            md.write(f"- F1: {format_metric(summary_binary.get('micro_f1'))}\n")
            md.write(f"- Accuracy: {format_metric(summary_binary.get('micro_accuracy'))}\n\n")
        
        # Add No Finding section if present
        nf_row = per_label[per_label['label'] == 'No Finding']
        if len(nf_row) > 0:
            nf = nf_row.iloc[0]
            md.write("## No Finding Metrics\n\n")
            md.write(f"- Precision: {format_metric(nf.get('precision_certain'))}\n")
            md.write(f"- Recall: {format_metric(nf.get('recall_certain'))}\n")
            md.write(f"- F1: {format_metric(nf.get('f1_certain'))}\n")
            md.write(f"- Accuracy: {format_metric(nf.get('accuracy_certain'))}\n")
            md.write(f"- Coverage: {int(nf.get('coverage_certain', 0))}/{int(nf.get('total_certain', 0))}\n")
            md.write(f"- GT Positives: {int(nf.get('positives_certain', 0))}\n")
            md.write(f"- Pred Positives: {int(nf.get('predicted_pos_certain', 0))}\n\n")

        md.write("## Per-Label Performance (Certain-Only)\n\n")
        # Exclude No Finding from main table, show separately above
        per_label_without_nf = per_label[per_label['label'] != 'No Finding']
        md.write(render_per_label_markdown(per_label_without_nf))
        md.write("\n")

    # Combined CSV (summary + per label)
    summary_rows = []
    for key in SUMMARY_KEYS:
        row = {
            "label": f"summary::{key}",
            "precision_certain": summary_certain.get(key),
            "recall_certain": None,
            "f1_certain": None,
        }
        if summary_binary:
            row.update({
                "precision_binary": summary_binary.get(key),
                "recall_binary": None,
                "f1_binary": None,
            })
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    combined = pd.concat([summary_df, per_label], ignore_index=True, sort=False)
    
    # Round numeric columns to 3 decimals for consistency
    numeric_cols = [col for col in combined.columns if col not in ['label'] and combined[col].dtype in ['float64', 'float32']]
    for col in numeric_cols:
        # Round to 3 decimals and ensure trailing zeros are preserved in CSV
        combined[col] = combined[col].apply(lambda x: round(float(x), 3) if pd.notna(x) else x)
    
    # Use float_format to ensure all floats display with 3 decimals (0.420 not 0.42)
    combined.to_csv(args.out_csv, index=False, float_format='%.3f')

    print(f"✅ Manager report written to {args.out_md}")
    print(f"✅ Manager summary CSV written to {args.out_csv}")


if __name__ == "__main__":
    main()
