"""
Smart Ensemble: hybrid CheXagent inference with per-label thresholds.

This flow always collects binary predictions, derives calibrated scores,
and optionally rescues under-called labels using disease_identification text.

NEW: Supports Platt calibration + precision-first gating (HARD vs EASY labels).
"""

import csv
import json
import re
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add chexagent_repo to path
sys.path.insert(0, str(Path(__file__).parent / "chexagent_repo"))
from model_chexagent.chexagent import CheXagent  # noqa: E402

CHEXPERT13 = [
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
]

# Tuned starting thresholds; override via config/label_thresholds.json.
DEFAULT_THRESHOLDS = {
    "Enlarged Cardiomediastinum": 0.70,
    "Cardiomegaly": 0.85,
    "Lung Opacity": 0.65,
    "Lung Lesion": 0.70,
    "Edema": 0.55,
    "Consolidation": 0.60,
    "Pneumonia": 0.60,
    "Atelectasis": 0.85,
    "Pneumothorax": 0.55,
    "Pleural Effusion": 0.60,
    "Pleural Other": 0.50,
    "Fracture": 0.65,
    "Support Devices": 0.60,
}

# Precision-first label grouping (based on evaluation results)
# HARD: labels with chronic FP issues → require DI confirmation (AND gating)
# EASY: labels with good precision → allow DI boost (OR gating)
HARD_LABELS = {
    "Fracture", "Lung Lesion", "Pleural Other", 
    "Consolidation", "Pneumonia", "Enlarged Cardiomediastinum"
}
EASY_LABELS = {
    "Pleural Effusion", "Edema", "Lung Opacity", 
    "Support Devices", "Pneumothorax", "Cardiomegaly", "Atelectasis"
}

# Precision-oriented policy knobs
CONF_MARGIN = 0.05              # require score >= tau + margin for direct binary positives
DI_STRICT_STRENGTH = 0.70       # minimum DI strength for any rescue
DI_BORDERLINE_WINDOW = 0.15     # only allow DI rescue if |score - tau| <= window

# Disease-specific confidence margins (higher = more conservative, fewer false positives)
# Reduced slightly to balance precision/recall better
DISEASE_CONF_MARGINS = {
    "Enlarged Cardiomediastinum": 0.08,  # Low precision → larger margin (reduced from 0.10)
    "Pleural Other": 0.12,               # Very low precision → largest margin (reduced from 0.15)
    "Lung Lesion": 0.06,                 # Moderate precision issues (reduced from 0.08)
    "Consolidation": 0.06,               # Reduced from 0.07
    "Fracture": 0.06,                    # Reduced from 0.08
    # Default for others is CONF_MARGIN (0.05)
}

NEGATION_TERMS = ["no", "without", "absent", "negative for", "lack of"]
UNCERTAIN_TERMS = ["possible", "equivocal", "uncertain", "cannot rule out", "indeterminate"]
POSITIVE_STRONG = ["definite", "definitive", "marked", "severe", "obvious", "highly suggestive"]
POSITIVE_MODERATE = ["present", "yes", "shows", "demonstrates", "consistent with", "evidence of"]
POSITIVE_WEAK = ["likely", "suggests", "probable", "mild", "slight"]
NEGATIVE_STRONG = ["no evidence", "absent", "not seen", "without any", "ruled out", "free of", 
                    "no indication", "clearly absent", "definitively absent", "no signs of"]
NEGATIVE_WEAK = ["unlikely", "doubtful", "probably not", "minimal", "unlikely to be"]


def load_thresholds(threshold_path: Path) -> Dict[str, float]:
    if threshold_path and threshold_path.exists():
        try:
            data = json.loads(threshold_path.read_text())
            return {label: float(data.get(label, DEFAULT_THRESHOLDS.get(label, 0.5))) for label in CHEXPERT13}
        except json.JSONDecodeError:
            print(f"⚠️  Failed to parse thresholds at {threshold_path}, falling back to defaults.")
    return DEFAULT_THRESHOLDS.copy()


def load_calibration(calib_path: Optional[Path]) -> Optional[Dict[str, Dict[str, float]]]:
    """Load Platt calibration parameters (a, b) per label."""
    if not calib_path or not calib_path.exists():
        return None
    try:
        data = json.loads(calib_path.read_text())
        print(f"✅ Loaded Platt calibration from {calib_path}")
        return data
    except json.JSONDecodeError:
        print(f"⚠️  Failed to parse calibration at {calib_path}")
        return None


def apply_calibration(score: float, label: str, calib_params: Optional[Dict]) -> float:
    """Apply Platt calibration: P(y=1|s) = 1 / (1 + exp(-(a*s + b)))"""
    if not calib_params or label not in calib_params:
        return score
    a = calib_params[label]["a"]
    b = calib_params[label]["b"]
    calibrated = 1.0 / (1.0 + np.exp(-(a * score + b)))
    return float(np.clip(calibrated, 0.0, 1.0))


def parse_numeric_scores(text: str) -> List[float]:
    scores = []
    for match in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*%?", text):
        raw = match.group(1)
        try:
            value = float(raw)
            if "%" in match.group(0):
                value /= 100.0
            if 0.0 <= value <= 1.0:
                scores.append(value)
        except ValueError:
            continue
    return scores


def parse_binary_response(response: str, disease: str) -> Tuple[float, List[str]]:
    text = response.lower()
    score = 0.5
    reasons: List[str] = []

    # Check for explicit negations first (strongest signal)
    # Pattern: "no [disease]" or "without [disease]" or similar
    disease_lower = disease.lower()
    explicit_negation = False
    for neg_term in NEGATION_TERMS:
        # Check for patterns like "no cardiomegaly", "without evidence of edema", etc.
        if f"{neg_term} {disease_lower}" in text or f"{neg_term} evidence of {disease_lower}" in text:
            score = 0.05
            reasons.append("explicit_negation")
            explicit_negation = True
            break

    if not explicit_negation:
        numeric_scores = parse_numeric_scores(text)
        if numeric_scores:
            score = max(numeric_scores)
            reasons.append("numeric")

        # Negative evidence (check after negation patterns)
        if any(term in text for term in NEGATIVE_STRONG):
            score = min(score, 0.05)
            reasons.append("neg_strong")
        elif any(term in text for term in NEGATION_TERMS + NEGATIVE_WEAK):
            score = min(score, 0.20)
            reasons.append("neg_weak")

        # Positive evidence
        if any(term in text for term in POSITIVE_STRONG):
            score = max(score, 0.92)
            reasons.append("pos_strong")
        elif any(term in text for term in POSITIVE_MODERATE):
            score = max(score, 0.70)  # more conservative than 0.80
            reasons.append("pos_moderate")
        elif any(term in text for term in POSITIVE_WEAK):
            score = max(score, 0.60)
            reasons.append("pos_weak")

        # Uncertainty penalty (applies even to positive signals)
        if any(term in text for term in UNCERTAIN_TERMS):
            # More aggressive penalty: reduce score significantly
            if score > 0.60:
                score = max(min(score * 0.7, 0.52), 0.35)  # Reduce by 30% with bounds
            else:
                score = max(min(score, 0.52), 0.30)
            reasons.append("uncertain")

    score = max(0.0, min(score, 1.0))

    if not response.strip():
        reasons.append("blank")
        score = 0.5

    if "error" in text:
        score = 0.5
        reasons.append("error")

    return score, reasons


def parse_di_response(text: str) -> Dict[str, Dict[str, float]]:
    text_lower = text.lower()
    info: Dict[str, Dict[str, float]] = {}

    for disease in CHEXPERT13:
        disease_lower = disease.lower()
        entry = {
            "mentioned": 0,
            "negated": 0,
            "uncertain": 0,
            "strength": 0.0,
        }

        # Check for explicit negation first (strongest negative signal)
        for neg_term in NEGATION_TERMS:
            neg_patterns = [
                f"{neg_term} {disease_lower}",
                f"{neg_term} evidence of {disease_lower}",
                f"{neg_term} signs of {disease_lower}",
            ]
            if any(pattern in text_lower for pattern in neg_patterns):
                entry["negated"] = 1
                entry["strength"] = 0.0
                info[disease] = entry
                continue

        # Positive assertion only if disease is referenced with positive language
        # Avoid treating mere mentions (e.g., echoed options) as evidence
        mentioned = disease_lower in text_lower
        
        if mentioned:
            # Check for strong positive language near the disease mention
            disease_pos = text_lower.find(disease_lower)
            context_window = 50  # characters before/after disease mention to check
            context_start = max(0, disease_pos - context_window)
            context_end = min(len(text_lower), disease_pos + len(disease_lower) + context_window)
            context = text_lower[context_start:context_end]
            
            # Strong positive evidence
            if any(term in context for term in POSITIVE_STRONG):
                entry["mentioned"] = 1
                entry["strength"] = 0.9
            # Moderate positive evidence
            elif any(term in context for term in POSITIVE_MODERATE):
                entry["mentioned"] = 1
                entry["strength"] = 0.7
            # Weak positive evidence
            elif any(term in context for term in POSITIVE_WEAK):
                entry["mentioned"] = 1
                entry["strength"] = 0.5
            # Mere mention without clear positive language → lower strength
            elif any(term in text_lower for term in POSITIVE_STRONG + POSITIVE_MODERATE + POSITIVE_WEAK):
                entry["mentioned"] = 1
                entry["strength"] = 0.4  # Conservative: weak evidence

        # Check for uncertainty (reduces strength if already set)
        for term in UNCERTAIN_TERMS:
            if f"{term} {disease_lower}" in text_lower:
                entry["uncertain"] = 1
                # Reduce strength if uncertain
                if entry["strength"] > 0:
                    entry["strength"] = max(entry["strength"] * 0.6, 0.3)
                else:
                    entry["strength"] = max(entry["strength"], 0.2)

        info[disease] = entry

    return info


def decide_label(
    disease: str,
    binary_score: float,
    thresholds: Dict[str, float],
    di_entry: Dict[str, float],
    use_precision_gating: bool = False,
) -> Tuple[int, List[str]]:
    """
    Decide final label with optional precision-first gating.
    
    Args:
        disease: Disease label
        binary_score: Calibrated binary score (0-1)
        thresholds: Per-label thresholds
        di_entry: DI parsing result
        use_precision_gating: If True, use HARD/EASY label logic
    """
    tau = thresholds.get(disease, 0.5)
    reasons: List[str] = []
    decision = 0

    # Hard negation from DI wins (highest priority)
    if di_entry.get("negated"):
        decision = 0
        reasons.append("di_negation")
        return decision, reasons

    # Get DI signal strength
    di_mentioned = di_entry.get("mentioned", False)
    di_strength = float(di_entry.get("strength", 0.0))
    di_uncertain = di_entry.get("uncertain", False)
    di_strong = di_mentioned and not di_uncertain and di_strength >= DI_STRICT_STRENGTH

    if use_precision_gating:
        # NEW: Precision-first gating based on label difficulty
        t_hi = tau
        t_lo = 0.5 * t_hi  # Gray zone threshold
        
        if disease in HARD_LABELS:
            # HARD labels: require DI confirmation (AND gating)
            if binary_score >= t_hi:
                if di_strong:
                    decision = 1
                    reasons.append("hard_binary_high+di_confirm")
                else:
                    # Allow direct positive only if score is very high
                    margin = DISEASE_CONF_MARGINS.get(disease, CONF_MARGIN)
                    if binary_score >= t_hi + margin:
                        decision = 1
                        reasons.append("hard_binary_very_high")
                    else:
                        decision = 0
                        reasons.append("hard_no_di_confirmation")
            elif t_lo <= binary_score < t_hi:
                # Gray zone: require strong DI
                if di_strong:
                    decision = 1
                    reasons.append("hard_gray_zone+di_rescue")
                else:
                    decision = 0
                    reasons.append("hard_gray_zone_no_di")
            else:
                decision = 0
                reasons.append("hard_below_threshold")
        
        elif disease in EASY_LABELS:
            # EASY labels: allow DI boost (OR gating)
            if binary_score >= t_hi:
                decision = 1
                reasons.append("easy_binary_high")
            elif t_lo <= binary_score < t_hi:
                # Gray zone: DI can rescue
                if di_strong:
                    decision = 1
                    reasons.append("easy_gray_zone+di_rescue")
                else:
                    decision = 0
                    reasons.append("easy_gray_zone_no_di")
            elif di_strong:
                # Below threshold but strong DI can rescue
                decision = 1
                reasons.append("easy_di_rescue")
            else:
                decision = 0
                reasons.append("easy_below_threshold")
        
        else:
            # Fallback for unlabeled diseases (shouldn't happen)
            decision = 1 if binary_score >= tau else 0
            reasons.append("unlabeled_fallback")
    
    else:
        # LEGACY: Original logic with confidence margins
        margin = DISEASE_CONF_MARGINS.get(disease, CONF_MARGIN)
        required_score = tau + margin
        
        if binary_score >= required_score:
            decision = 1
            reasons.append("binary_strong")
        elif tau <= binary_score < required_score:
            # Borderline: allow DI boost for EASY labels only
            if disease in EASY_LABELS and di_strong:
                decision = 1
                reasons.append("borderline+di_boost")
            else:
                decision = 0
                reasons.append("borderline_insufficient")
        elif (tau - DI_BORDERLINE_WINDOW) <= binary_score < tau:
            # Near miss: allow DI rescue for EASY labels
            if disease in EASY_LABELS and di_strong:
                decision = 1
                reasons.append("rescue_di_strong")
            else:
                decision = 0
                reasons.append("below_tau_close")
        else:
            decision = 0
            reasons.append("below_tau")

    return decision, reasons


def smart_ensemble_prediction(
    image_paths: List[Path],
    device: str = "mps",
    thresholds: Dict[str, float] = None,
    calibration_params: Optional[Dict[str, Dict[str, float]]] = None,
    use_precision_gating: bool = False,
) -> List[Dict[str, object]]:
    """
    Run CheXagent inference with optional calibration and precision gating.
    
    Args:
        image_paths: List of image paths
        device: Device for CheXagent ('mps', 'cuda', or 'cpu')
        thresholds: Per-label thresholds
        calibration_params: Optional Platt calibration parameters per label
        use_precision_gating: If True, use HARD/EASY precision-first gating
    """
    print(f"🔧 Initializing CheXagent on device: {device}")
    chex = CheXagent(device=device)
    thresholds = thresholds or DEFAULT_THRESHOLDS
    
    if calibration_params:
        print("📊 Calibration: ENABLED (Platt)")
    if use_precision_gating:
        print("🎯 Precision gating: ENABLED (HARD/EASY labels)")

    results: List[Dict[str, object]] = []

    for index, img_path in enumerate(image_paths, start=1):
        print(f"\nProcessing {index}/{len(image_paths)}: {img_path.name}")

        di_text = chex.disease_identification([str(img_path)], CHEXPERT13)
        di_info = parse_di_response(di_text)
        print(f"  DI response: {di_text[:150]}...")

        binary_outputs: Dict[str, Dict[str, object]] = {}
        final_labels: Dict[str, int] = {}
        log_entries: List[str] = []

        for disease in CHEXPERT13:
            try:
                raw_binary = chex.binary_disease_classification([str(img_path)], disease)
            except Exception as exc:
                raw_binary = f"ERROR: {exc}"
                score = 0.5
                analysis_reasons = ["exception"]
            else:
                score, analysis_reasons = parse_binary_response(raw_binary, disease)
            
            # Apply calibration if available
            score_raw = score
            if calibration_params:
                score = apply_calibration(score, disease, calibration_params)

            decision, decision_reasons = decide_label(
                disease, score, thresholds, di_info[disease], use_precision_gating
            )

            binary_outputs[disease] = {
                "score_raw": round(score_raw, 4),
                "score": round(score, 4),
                "calibrated": calibration_params is not None,
                "threshold": thresholds.get(disease, 0.5),
                "raw": raw_binary,
                "analysis_reasons": analysis_reasons,
                "decision_reasons": decision_reasons,
            }

            final_labels[disease] = int(decision)

            reason_txt = "|".join(decision_reasons) if decision_reasons else "none"
            log_entries.append(
                f"{disease}: score={score:.2f} tau={thresholds.get(disease, 0.5):.2f} -> {decision} ({reason_txt})"
            )

        no_finding = 1 if all(v == 0 for v in final_labels.values()) else 0
        final_labels["No Finding"] = no_finding

        result = {
            "image": str(img_path),
            "initial_response": di_text,
            "verification_log": "; ".join(log_entries),
        }
        result.update(final_labels)
        result["binary_outputs"] = json.dumps(binary_outputs)
        result["di_outputs"] = json.dumps(di_info)

        results.append(result)
        positives = [label for label, value in final_labels.items() if value == 1 and label != "No Finding"]
        print(f"  Final positives: {positives if positives else 'None'}")

    return results


def save_results(results: List[Dict[str, object]], output_path: Path):
    fieldnames = (
        ["image", "initial_response", "verification_log"]
        + CHEXPERT13
        + ["No Finding", "binary_outputs", "di_outputs"]
    )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"📄 Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CheXagent smart ensemble with calibration support")
    parser.add_argument("--images", type=str, required=True, help="Path to image list or directory")
    parser.add_argument("--out_csv", type=str, default="smart_ensemble_results.csv", help="Output CSV path")
    parser.add_argument("--device", type=str, default="mps", help="Device: mps, cuda, or cpu")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="config/label_thresholds.json",
        help="Path to per-label thresholds JSON",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Optional path to Platt calibration params (platt_params.json)",
    )
    parser.add_argument(
        "--use_precision_gating",
        action="store_true",
        help="Enable precision-first HARD/EASY label gating",
    )

    args = parser.parse_args()

    # Import from same directory (works when run as script from project root)
    import sys
    from pathlib import Path
    inference_dir = Path(__file__).parent
    if str(inference_dir) not in sys.path:
        sys.path.insert(0, str(inference_dir))
    from infer_with_chexagent_class import collect_image_paths

    # Resolve paths relative to project root if needed
    images_path = Path(args.images)
    if not images_path.is_absolute() and not images_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        images_path = project_root / images_path
    image_paths = collect_image_paths(images_path)
    
    thresholds_path = Path(args.thresholds)
    if not thresholds_path.is_absolute() and not thresholds_path.exists():
        project_root = Path(__file__).parent.parent.parent
        thresholds_path = project_root / thresholds_path
    thresholds = load_thresholds(thresholds_path)
    
    # Resolve calibration path if provided
    calibration_path = None
    if args.calibration:
        calibration_path = Path(args.calibration)
        if not calibration_path.is_absolute() and not calibration_path.exists():
            project_root = Path(__file__).parent.parent.parent
            calibration_path = project_root / calibration_path
    calibration_params = load_calibration(calibration_path)

    results = smart_ensemble_prediction(
        image_paths, 
        args.device, 
        thresholds,
        calibration_params,
        args.use_precision_gating,
    )
    save_results(results, Path(args.out_csv))

    print(f"\n🎉 Smart ensemble processing complete!")
    print(f"Processed {len(results)} images")
    if calibration_params:
        print("✅ Calibration was applied")
    if args.use_precision_gating:
        print("✅ Precision gating was enabled")
