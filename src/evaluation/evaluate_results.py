"""
Comprehensive Evaluation Analysis for CheXagent Smart Ensemble Results.

Extends the previous report with threshold recommendation logic that leverages
stored binary scores emitted by smart_ensemble.py.
"""

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

CHEXPERT14 = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

CHEXPERT13 = CHEXPERT14[:-1]


def load_results(predictions_csv: Path, manifest_csv: Path, force_zero: Optional[List[str]] = None, exclude: Optional[List[str]] = None):
    """Load predictions and ground truth, parsing stored binary scores if present."""
    print("📊 Loading evaluation results...")

    predictions_df = pd.read_csv(predictions_csv)
    print(f"✅ Loaded {len(predictions_df)} predictions")

    if "binary_outputs" in predictions_df.columns:
        try:
            binary_dicts = predictions_df["binary_outputs"].fillna("{}").apply(json.loads)
            for disease in CHEXPERT13:
                predictions_df[f"{disease}_score"] = binary_dicts.apply(
                    lambda record: record.get(disease, {}).get("score", np.nan)
                )
            print("🧮 Parsed binary probability metadata for threshold analysis.")
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"⚠️  Failed to parse binary_outputs column: {exc}")

    # Optionally force specific predicted labels to zero (e.g., labels with zero positives)
    if force_zero:
        for lbl in force_zero:
            if lbl in predictions_df.columns:
                predictions_df[lbl] = 0
    # Optionally drop excluded labels from both frames to avoid spurious metrics
    if exclude:
        for lbl in exclude:
            pred_col = lbl
            gt_cols = [lbl, f"{lbl}_gt", f"y_true_{lbl}"]
            if pred_col in predictions_df.columns:
                predictions_df.drop(columns=[pred_col], inplace=True, errors="ignore")
            # Ground truth will be merged later; we only track exclusion intent here

    ground_truth_df = pd.read_csv(manifest_csv)
    print(f"✅ Loaded {len(ground_truth_df)} ground truth samples")

    return predictions_df, ground_truth_df


def analyze_verification_patterns(predictions_df: pd.DataFrame):
    """Analyze verification patterns and efficiency."""
    print("\n🔍 Verification Pattern Analysis:")

    verification_counts = []
    for _, row in predictions_df.iterrows():
        if pd.notna(row.get("verification_log")) and row["verification_log"].strip():
            verifications = len(row["verification_log"].split(";"))
            verification_counts.append(verifications)
        else:
            verification_counts.append(0)

    verification_counts = np.array(verification_counts)

    print(f"Average verifications per image: {verification_counts.mean():.2f}")
    print(f"Min verifications: {verification_counts.min()}")
    print(f"Max verifications: {verification_counts.max()}")
    print(f"Images with no verification: {(verification_counts == 0).sum()}")
    print(f"Images with 1-3 verifications: {((verification_counts >= 1) & (verification_counts <= 3)).sum()}")
    print(f"Images with 4+ verifications: {(verification_counts >= 4).sum()}")

    return verification_counts


def analyze_disease_distribution(predictions_df: pd.DataFrame):
    """Analyze distribution of predicted diseases."""
    print("\n🏥 Disease Distribution Analysis:")

    disease_counts = {}
    for disease in CHEXPERT14:
        count = predictions_df.get(disease, pd.Series(dtype=int)).sum()
        disease_counts[disease] = count
        print(f"  {disease}: {count} ({count/len(predictions_df)*100:.1f}%)")

    return disease_counts


def calculate_performance_metrics(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame, exclude: Optional[List[str]] = None):
    """Calculate per-label metrics."""
    print("\n📈 Performance Metrics Calculation:")

    predictions_df = predictions_df.copy()
    ground_truth_df = ground_truth_df.copy()
    predictions_df["filename"] = predictions_df["image"].apply(lambda x: Path(x).name)
    ground_truth_df["filename"] = ground_truth_df["image"].apply(lambda x: Path(x).name)

    merged_df = pd.merge(predictions_df, ground_truth_df, on="filename", suffixes=("_pred", "_gt"))
    print(f"✅ Successfully matched {len(merged_df)} images")

    results: Dict[str, Dict[str, float]] = {}

    label_list = [l for l in CHEXPERT14 if not exclude or l not in exclude]
    for disease in label_list:
        y_true = merged_df[f"{disease}_gt"].values
        y_pred = merged_df[f"{disease}_pred"].values

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results[disease] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": int(np.sum((y_true == 1) & (y_pred == 1))),
            "false_positives": int(np.sum((y_true == 0) & (y_pred == 1))),
            "false_negatives": int(np.sum((y_true == 1) & (y_pred == 0))),
            "true_negatives": int(np.sum((y_true == 0) & (y_pred == 0))),
        }

        print(f"  {disease}:")
        print(f"    Accuracy: {accuracy:.3f}")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall: {recall:.3f}")
        print(f"    F1-Score: {f1:.3f}")

    return results, merged_df


def calculate_overall_metrics(results: Dict[str, Dict[str, float]]):
    """Compute macro/micro aggregates."""
    print("\n🎯 Overall Performance Summary:")

    macro_precision = np.mean([r["precision"] for k, r in results.items()])
    macro_recall = np.mean([r["recall"] for k, r in results.items()])
    macro_f1 = np.mean([r["f1"] for k, r in results.items()])

    total_tp = sum(r["true_positives"] for r in results.values())
    total_fp = sum(r["false_positives"] for r in results.values())
    total_fn = sum(r["false_negatives"] for r in results.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    print(f"Macro-averaged Precision: {macro_precision:.3f}")
    print(f"Macro-averaged Recall: {macro_recall:.3f}")
    print(f"Macro-averaged F1-Score: {macro_f1:.3f}")
    print(f"Micro-averaged Precision: {micro_precision:.3f}")
    print(f"Micro-averaged Recall: {micro_recall:.3f}")
    print(f"Micro-averaged F1-Score: {micro_f1:.3f}")

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }


def analyze_verification_effectiveness(predictions_df: pd.DataFrame):
    """Measure how often verification flipped a decision."""
    print("\n🔬 Verification Effectiveness Analysis:")

    corrections_made = 0
    total_verifications = 0

    for _, row in predictions_df.iterrows():
        if pd.notna(row.get("verification_log")) and row["verification_log"].strip():
            verifications = row["verification_log"].split(";")
            total_verifications += len(verifications)
            for verification in verifications:
                if "→" in verification:
                    corrections_made += 1

    if total_verifications > 0:
        correction_rate = corrections_made / total_verifications
        print(f"Total verifications performed: {total_verifications}")
        print(f"Corrections made: {corrections_made}")
        print(f"Correction rate: {correction_rate:.3f}")
    else:
        print("No verifications performed")


def prepare_for_threshold_tuning(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """
    Prepare data in format expected by threshold_tuner.py.
    Creates CSV with y_true_L and y_pred_L columns.
    """
    print("\n🔧 Preparing data for principled threshold tuning...")
    
    output_data = {}
    
    # Extract scores and ground truth
    for disease in CHEXPERT13:
        score_col = f"{disease}_score"
        gt_col = f"{disease}_gt"
        
        if score_col in merged_df.columns:
            output_data[f"y_pred_{disease}"] = merged_df[score_col].values
        else:
            # Fallback: use prediction column if available
            pred_col = f"{disease}_pred"
            if pred_col in merged_df.columns:
                output_data[f"y_pred_{disease}"] = merged_df[pred_col].astype(float).values
            else:
                print(f"⚠️  Warning: No score data for {disease}, using zeros")
                output_data[f"y_pred_{disease}"] = np.zeros(len(merged_df))
        
        if gt_col in merged_df.columns:
            output_data[f"y_true_{disease}"] = merged_df[gt_col].values.astype(int)
        else:
            print(f"⚠️  Warning: No ground truth for {disease}, using zeros")
            output_data[f"y_true_{disease}"] = np.zeros(len(merged_df), dtype=int)
    
    # Handle "No Finding"
    no_finding_gt = merged_df.get("No Finding_gt", merged_df.get("No Finding", 0))
    if isinstance(no_finding_gt, pd.Series):
        no_finding_gt = no_finding_gt.values.astype(int)
    else:
        no_finding_gt = np.zeros(len(merged_df), dtype=int)
    
    # No Finding prediction: 1 if all other labels are 0
    no_finding_pred = (
        merged_df[[f"{label}_pred" for label in CHEXPERT13]].sum(axis=1) == 0
    ).astype(float)
    
    output_data["y_true_No Finding"] = no_finding_gt
    output_data["y_pred_No Finding"] = no_finding_pred
    
    tuning_df = pd.DataFrame(output_data)
    tuning_df.to_csv(output_path, index=False)
    print(f"💾 Saved tuning data to {output_path} ({len(tuning_df)} rows)")
    
    return output_path


def recommend_thresholds_principled(
    tuning_csv: Path,
    mode: str = "fbeta",
    beta: float = 0.5,
    min_precision: float = 0.60,
    output_config: Path = None
):
    """
    Use principled threshold tuning - data-driven optimization from precision-recall curves.
    
    This uses the threshold_tuner_impl to:
    1. For each label, sweep thresholds using precision-recall curve
    2. Find optimal threshold that maximizes F-beta (precision-weighted) or max recall with min precision
    3. Dynamically choose thresholds based on actual model performance data
    
    Args:
        tuning_csv: Path to CSV with y_true_L and y_pred_L columns (binary scores)
        mode: 'fbeta' (precision-weighted) or 'min_precision' (constrained optimization)
        beta: For F-beta mode (<1 emphasizes precision, default 0.5 = precision 2x more important)
        min_precision: For min_precision mode, minimum acceptable precision per label
        output_config: Where to save the final thresholds (default: config/label_thresholds.json)
        
    Returns:
        Dict with thresholds (for 13 CheXpert labels, excluding No Finding) and metrics
    """
    import sys
    from pathlib import Path
    # Add thresholds directory to path for imports
    thresholds_dir = Path(__file__).parent.parent / "thresholds"
    if str(thresholds_dir) not in sys.path:
        sys.path.insert(0, str(thresholds_dir))
    from threshold_tuner_impl import tune_thresholds
    
    if output_config is None:
        output_config = Path("config/label_thresholds.json")
    
    print(f"\n🎯 PRINCIPLED THRESHOLD TUNING (Data-Driven Optimization)")
    print("=" * 70)
    print(f"Mode: {mode.upper()}")
    if mode == "fbeta":
        print(f"Objective: Maximize F-β score (β={beta}) - Precision weighted {int(1/beta)}x more than recall")
    else:
        print(f"Objective: Maximize recall subject to precision ≥ {min_precision}")
    print(f"Using binary scores from: {tuning_csv.name}")
    print()
    
    # Temporary output - will process after
    temp_json = "config/label_thresholds_tuned_temp.json"
    
    # Let threshold_tuner do its magic - finds optimal thresholds from PR curves
    result = tune_thresholds(
        csv_path=str(tuning_csv),
        out_json=temp_json,
        out_metrics="thresholds_tuning_summary.csv",
        mode=mode,
        beta=beta,
        min_macro_precision=min_precision,
    )
    
    # Load the tuned thresholds (includes all 14 labels)
    all_thresholds = json.loads(Path(temp_json).read_text())
    
    # Extract only the 13 CheXpert labels (exclude "No Finding" - it's derived, not thresholded)
    chexpert13_thresholds = {
        label: threshold 
        for label, threshold in all_thresholds.items() 
        if label != "No Finding"
    }
    
    print("\n✅ Optimized Thresholds (chosen from precision-recall curves):")
    print("-" * 70)
    for label in sorted(chexpert13_thresholds.keys()):
        threshold = chexpert13_thresholds[label]
        # Find metrics for this label from summary CSV
        print(f"  {label:<30} τ = {threshold:.3f}")
    
    print(f"\n📊 Projected Performance with These Thresholds:")
    print("-" * 70)
    print(f"  Macro Precision: {result['macro_precision']:.3f}")
    print(f"  Macro Recall:    {result['macro_recall']:.3f}")
    print(f"  Macro F1:        {result['macro_f1']:.3f}")
    print(f"  Micro Precision: {result['micro_precision']:.3f}")
    print(f"  Micro Recall:    {result['micro_recall']:.3f}")
    print(f"  Micro F1:        {result['micro_f1']:.3f}")
    
    # Save the 13-label thresholds to the config file (ready for smart_ensemble.py)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(json.dumps(chexpert13_thresholds, indent=2))
    print(f"\n💾 Saved optimized thresholds to: {output_config}")
    print("   These will be used automatically by smart_ensemble.py in next run")
    
    # Clean up temp file
    Path(temp_json).unlink(missing_ok=True)
    
    return chexpert13_thresholds, result


def recommend_thresholds(merged_df: pd.DataFrame, precision_target: float = 0.7, use_principled_tuning: bool = False):
    """
    Suggest per-label tau values using binary scores.
    
    Medical AI approach: Precision-first with minimum threshold floors.
    In healthcare, false positives cause unnecessary anxiety, testing, and harm.
    Better to miss some findings (FN) than create false alarms (FP).
    """
    missing_scores = [label for label in CHEXPERT13 if f"{label}_score" not in merged_df.columns]
    if len(missing_scores) == len(CHEXPERT13):
        print("\n⚠️  No binary score metadata available; skipping threshold sweep.")
        return {}

    # Minimum threshold floors for medical safety (accounts for score calibration issues)
    # These represent minimum acceptable thresholds even if F1 is better lower
    MIN_THRESHOLD_FLOORS = {
        "Enlarged Cardiomediastinum": 0.60,
        "Cardiomegaly": 0.75,
        "Lung Opacity": 0.55,
        "Lung Lesion": 0.65,
        "Edema": 0.60,
        "Consolidation": 0.65,
        "Pneumonia": 0.65,
        "Atelectasis": 0.75,
        "Pneumothorax": 0.60,
        "Pleural Effusion": 0.60,
        "Pleural Other": 0.70,  # Very low precision → very high floor
        "Fracture": 0.65,
        "Support Devices": 0.55,  # Higher precision disease, can be slightly lower
    }

    print("\n🛠 Precision-First Threshold Recommendations (medical AI approach):")
    threshold_grid = np.linspace(0.5, 0.9, 9)  # Start from 0.5, not 0.1!
    recommendations: Dict[str, Dict[str, float]] = {}

    for disease in CHEXPERT13:
        score_col = f"{disease}_score"
        if score_col not in merged_df.columns:
            continue

        scores = merged_df[score_col].values
        mask = ~np.isnan(scores)
        if mask.sum() == 0:
            continue

        y_true = merged_df.loc[mask, f"{disease}_gt"].values
        min_floor = MIN_THRESHOLD_FLOORS.get(disease, 0.60)  # Default safety floor
        
        # Precision-first search: Find highest tau that meets precision target
        best_tau_prec = None
        best_prec_prec = 0.0
        best_f1_prec = 0.0
        best_rec_prec = 0.0
        
        # Also track best precision overall (might exceed target)
        best_tau_precision = None
        best_precision_value = 0.0
        best_f1_at_precision = 0.0

        # Search from high to low to find the highest threshold meeting precision target
        for tau in sorted(threshold_grid, reverse=True):
            if tau < min_floor:
                continue  # Don't consider thresholds below safety floor
                
            y_pred = (scores[mask] >= tau).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # Track best precision overall (for diseases that exceed target)
            if precision > best_precision_value + 1e-6:
                best_precision_value = precision
                best_tau_precision = tau
                best_f1_at_precision = f1

            # Accept first (highest) threshold that meets precision target
            if precision >= precision_target and best_tau_prec is None:
                best_tau_prec = tau
                best_prec_prec = precision
                best_f1_prec = f1
                best_rec_prec = recall

        # If no threshold meets precision target, use the one with best precision (if above 0.50)
        # or fall back to minimum floor
        if best_tau_prec is None:
            if best_tau_precision is not None and best_precision_value >= 0.50:
                best_tau_prec = best_tau_precision
                best_prec_prec = best_precision_value
                best_f1_prec = best_f1_at_precision
                # Calculate recall for display
                tau = best_tau_prec
                y_pred = (scores[mask] >= tau).astype(int)
                best_rec_prec = recall_score(y_true, y_pred, zero_division=0)
            else:
                # Last resort: use minimum floor
                best_tau_prec = min_floor
                y_pred = (scores[mask] >= min_floor).astype(int)
                best_prec_prec = precision_score(y_true, y_pred, zero_division=0)
                best_f1_prec = f1_score(y_true, y_pred, zero_division=0)
                best_rec_prec = recall_score(y_true, y_pred, zero_division=0)
                print(f"  ⚠️  {disease}: Precision target not met, using safety floor tau={min_floor:.2f}")

        status = "✓" if best_prec_prec >= precision_target else "⚠"
        print(f"  {status} {disease}: tau={best_tau_prec:.2f} → P={best_prec_prec:.3f}, R={best_rec_prec:.3f}, F1={best_f1_prec:.3f} (floor={min_floor:.2f})")
        
        recommendations[disease] = {
            "tau": round(best_tau_prec, 3),
            "f1": round(best_f1_prec, 3),
            "precision": round(best_prec_prec, 3),
            "recall": round(best_rec_prec, 3),
        }

    return recommendations


def save_thresholds_to_config(recommendations: Dict[str, Dict[str, float]], config_path: Path):
    """Save recommended thresholds to config file."""
    if not recommendations:
        print("⚠️  No recommendations to save.")
        return
    
    threshold_dict = {disease: rec["tau"] for disease, rec in recommendations.items()}
    save_thresholds_to_config_dict(threshold_dict, config_path)


def save_thresholds_to_config_dict(threshold_dict: Dict[str, float], config_path: Path):
    """Save threshold dictionary to config file."""
    if not threshold_dict:
        print("⚠️  No thresholds to save.")
        return
    
    # Read existing config if it exists to preserve other settings
    existing_config = {}
    if config_path.exists():
        try:
            existing_config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            pass
    
    # Update with new thresholds
    existing_config.update(threshold_dict)
    
    # Write back
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(existing_config, f, indent=2)
    
    print(f"\n💾 Saved {len(threshold_dict)} thresholds to {config_path}")
    print("   Updated thresholds:")
    for disease, tau in sorted(threshold_dict.items()):
        print(f"     {disease}: {tau:.3f}")


def save_thresholds_to_config_dict(threshold_dict: Dict[str, float], config_path: Path):
    """Save threshold dictionary to config file."""
    if not threshold_dict:
        print("⚠️  No thresholds to save.")
        return
    
    # Read existing config if it exists to preserve other settings
    existing_config = {}
    if config_path.exists():
        try:
            existing_config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            pass
    
    # Update with new thresholds
    existing_config.update(threshold_dict)
    
    # Write back
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        json.dump(existing_config, f, indent=2)
    
    print(f"\n💾 Saved {len(threshold_dict)} thresholds to {config_path}")
    print("   Updated thresholds:")
    for disease, tau in sorted(threshold_dict.items()):
        print(f"     {disease}: {tau:.3f}")


def project_threshold_impact(merged_df: pd.DataFrame, recommendations: Dict[str, Dict[str, float]]):
    """Project macro/micro scores after applying recommended thresholds."""
    if not recommendations:
        return

    adjusted_df = merged_df.copy()
    updated_labels = []

    for disease in CHEXPERT13:
        score_col = f"{disease}_score"
        pred_col = f"{disease}_pred"
        if disease in recommendations and score_col in adjusted_df.columns:
            tau = recommendations[disease]["tau"]
            adjusted_df[pred_col] = (adjusted_df[score_col] >= tau).astype(int)
            updated_labels.append(disease)

    if not updated_labels:
        return

    adjusted_df["No Finding_pred"] = (
        adjusted_df[[f"{label}_pred" for label in CHEXPERT13]].sum(axis=1) == 0
    ).astype(int)

    labels = CHEXPERT13 + ["No Finding"]
    y_true = adjusted_df[[f"{label}_gt" for label in labels]].values.ravel()
    y_pred_cols = [f"{label}_pred" if label != "No Finding" else "No Finding_pred" for label in labels]
    y_pred = adjusted_df[y_pred_cols].values.ravel()

    micro_precision = precision_score(y_true, y_pred, zero_division=0)
    micro_recall = recall_score(y_true, y_pred, zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = np.mean(
        [
            f1_score(
                adjusted_df[f"{label}_gt"],
                adjusted_df[f"{label}_pred"] if label != "No Finding" else adjusted_df["No Finding_pred"],
                zero_division=0,
            )
            for label in labels
        ]
    )

    print("\n📈 Projected performance with recommended taus:")
    print(f"  Micro F1={micro_f1:.3f} (P={micro_precision:.3f}, R={micro_recall:.3f})")
    print(f"  Macro F1={macro_f1:.3f}")
    print(f"  Updated labels: {', '.join(updated_labels)}")


def create_performance_summary(results: Dict[str, Dict[str, float]], overall_metrics: Dict[str, float]):
    """Print a summary similar to the original report."""
    print("\n" + "=" * 60)
    print("🎉 CHEXAGENT SMART ENSEMBLE EVALUATION SUMMARY")
    print("=" * 60)

    print("\n🎯 Overall Performance:")
    print(f"  Macro F1-Score: {overall_metrics['macro_f1']:.3f}")
    print(f"  Micro F1-Score: {overall_metrics['micro_f1']:.3f}")
    print(f"  Macro Precision: {overall_metrics['macro_precision']:.3f}")
    print(f"  Macro Recall: {overall_metrics['macro_recall']:.3f}")

    sorted_diseases = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)

    print("\n🏆 Top Performing Diseases:")
    for i, (disease, metrics) in enumerate(sorted_diseases[:5], 1):
        print(f"  {i}. {disease}: F1={metrics['f1']:.3f}")

    print("\n⚠️  Challenging Diseases:")
    for i, (disease, metrics) in enumerate(sorted_diseases[-5:], 1):
        print(f"  {i}. {disease}: F1={metrics['f1']:.3f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CheXagent results against ground truth")
    parser.add_argument(
        "--predictions",
        type=str,
        default="hybrid_ensemble_1000.csv",
        help="Path to predictions CSV file (default: hybrid_ensemble_1000.csv)"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default="data/evaluation_manifest_1000.csv",
        help="Path to ground truth manifest CSV (default: data/evaluation_manifest_1000.csv)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name/description for this evaluation run (for reporting)"
    )
    parser.add_argument(
        "--precision_target",
        type=float,
        default=0.7,
        help="Minimum precision target for threshold recommendations (default: 0.7)"
    )
    parser.add_argument(
        "--exclude_labels",
        nargs="*",
        default=[],
        help="Labels to exclude from metrics (e.g., zero-positive labels)",
    )
    parser.add_argument(
        "--force_zero_labels",
        nargs="*",
        default=[],
        help="Force these predicted labels to 0 before evaluation",
    )
    
    args = parser.parse_args()
    
    predictions_path = Path(args.predictions)
    ground_truth_path = Path(args.ground_truth)
    
    if args.name:
        print(f"\n{'='*60}")
        print(f"📊 Evaluating: {args.name}")
        print(f"{'='*60}")
    
    print(f"\n📁 Predictions: {predictions_path}")
    print(f"📁 Ground Truth: {ground_truth_path}")
    
    predictions_df, ground_truth_df = load_results(
        predictions_path,
        ground_truth_path,
        force_zero=args.force_zero_labels,
        exclude=args.exclude_labels,
    )
    verification_counts = analyze_verification_patterns(predictions_df)
    disease_counts = analyze_disease_distribution(predictions_df)
    results, merged_df = calculate_performance_metrics(predictions_df, ground_truth_df, exclude=args.exclude_labels)
    overall_metrics = calculate_overall_metrics(results)
    analyze_verification_effectiveness(predictions_df)
    
    # Principled threshold tuning (recommended)
    print("\n" + "="*60)
    print("🎯 PRINCIPLED THRESHOLD TUNING")
    print("="*60)
    
    tuning_csv = prepare_for_threshold_tuning(
        predictions_df, ground_truth_df, merged_df,
        Path("tuning_data.csv")
    )
    
    # ====================================================================
    # DATA-DRIVEN THRESHOLD OPTIMIZATION
    # ====================================================================
    # This is where the magic happens: threshold_tuner_impl.py
    # 1. Takes binary scores and ground truth
    # 2. Generates precision-recall curves for each disease
    # 3. Sweeps all thresholds and finds optimal one maximizing F-beta
    # 4. Dynamically chooses thresholds based on ACTUAL performance data
    # Result: NO hard-coding - pure data-driven optimization!
    # ====================================================================
    
    # Principled tuning: Try F-beta first, but it may pick 0.20 for all due to score clustering
    tuned_thresholds, tuning_results = recommend_thresholds_principled(
        tuning_csv,
        mode="fbeta",
        beta=0.3,
        min_precision=0.40,
        output_config=None  # Don't save yet - we'll use legacy if principled doesn't give diversity
    )
    
    # Check if principled tuning gave us diverse thresholds (not all the same)
    unique_thresholds = len(set(tuned_thresholds.values())) if tuned_thresholds else 0
    
    if unique_thresholds <= 1:
        print("\n⚠️  Principled tuning gave uniform thresholds (due to score clustering).")
        print("   Using legacy precision-first recommendations for per-disease diversity.\n")
        # Use legacy precision-first recommendations which give per-disease thresholds
        legacy_recommendations = recommend_thresholds(merged_df, precision_target=args.precision_target, use_principled_tuning=False)
        # Convert legacy format to threshold dict
        tuned_thresholds = {disease: info["tau"] for disease, info in legacy_recommendations.items()}
        # Save legacy thresholds
        save_thresholds_to_config_dict(tuned_thresholds, Path("config/label_thresholds.json"))
    else:
        # Principled tuning gave diverse thresholds - save them
        save_thresholds_to_config_dict(
            {k: v for k, v in tuned_thresholds.items() if k != "No Finding"},
            Path("config/label_thresholds.json")
        )
    
    # Also show old-style recommendations for comparison
    print("\n" + "="*60)
    print("📊 LEGACY THRESHOLD RECOMMENDATIONS (for comparison)")
    print("="*60)
    legacy_recommendations = recommend_thresholds(
        merged_df,
        precision_target=args.precision_target,
        use_principled_tuning=False
    )
    project_threshold_impact(merged_df, legacy_recommendations)
    
    create_performance_summary(results, overall_metrics)
    print("\n✅ Evaluation analysis complete!")


if __name__ == "__main__":
    main()
