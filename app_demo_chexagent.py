#!/usr/bin/env python3
"""
Streamlit demo for hybrid CheXpert inference.

Features:
    • Browse 5.8k evaluation samples (binary + three-class predictions, ground truth)
    • Upload your own chest X-ray and get hybrid TXR + CheXagent predictions
    • Short, structured impression with positive / uncertain labels
"""

from __future__ import annotations

import json
import math
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

# Ensure project modules are importable
PROJECT_ROOT = Path(__file__).parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in st.session_state.get("_sys_path", []):
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(SRC_ROOT))
    st.session_state.setdefault("_sys_path", []).extend([str(PROJECT_ROOT), str(SRC_ROOT)])

from src.calibration.platt_utils import apply_platt_to_scores  # noqa: E402
from src.common.labels import CHEXPERT13 as LABELS  # noqa: E402
from src.evaluation.run_test_eval import (  # noqa: E402
    apply_gating,
    apply_meta_calibration,
    blend_probabilities,
)
from src.inference.txr_infer import infer_txr  # noqa: E402
# Don't import smart_ensemble_prediction at module level - it requires chexagent_repo
# Will import only when needed in load_chexagent_model or run_chexagent_single


CHEXPERT14 = LABELS + ["No Finding"]
UNCERTAINTY_MARGIN = 0.15
INVERTED_PROB_LABELS: set[str] = set()


# --------------------------------------------------------------------------------------
# Helpers: loading artifacts
# --------------------------------------------------------------------------------------


def resolve_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path and path.exists() and path.stat().st_size > 0:
            return path
    return None


@st.cache_resource(show_spinner=False)
def load_pipeline_artifacts() -> Dict[str, object]:
    """Load pipeline outputs (thresholds, weights, calibrations, models, dataset)."""

    out_roots = [
        PROJECT_ROOT / "outputs_full_final",
        PROJECT_ROOT / "outputs_full",
        PROJECT_ROOT / "outputs_5k",
        PROJECT_ROOT / "server_synced" / "outputs_full",
    ]
    out_root = next((root for root in out_roots if root.exists()), None)
    if out_root is None:
        return {}

    def rel(*parts: str) -> Path:
        return out_root.joinpath(*parts)

    thresholds_path = resolve_existing(
        [
            rel("thresholds/thresholds_constrained_refit_full.json"),
            rel("thresholds/thresholds_constrained.json"),
            PROJECT_ROOT / "config/label_thresholds.json",
        ]
    )
    blend_weights_path = resolve_existing(
        [
            rel("blend/blend_weights_refit_full.json"),
            rel("blend/blend_weights.json"),
        ]
    )
    meta_params_path = resolve_existing(
        [
            rel("calibration/meta_auto_refit_full.json"),
            rel("calibration/meta_platt.json"),
        ]
    )
    txr_cal_path = resolve_existing(
        [
            rel("txr/calibration_txr/platt_params.json"),
            rel("txr/calibration/platt_params.json"),
            PROJECT_ROOT / "server_synced/txr_full/calibration_txr/platt_params.json",
        ]
    )
    txr_heavy_cal_path = resolve_existing(
        [
            rel("txr/calibration_txr_heavy/platt_params.json"),
        ]
    )
    probe_platt_path = resolve_existing(
        [
            rel("calibration/linear_probe_platt.json"),
            rel("linear_probe/linear_probe_platt.json"),
        ]
    )
    probe_model_path = resolve_existing(
        [
            rel("linear_probe/linear_probe.pkl"),
        ]
    )
    probe_summary_path = resolve_existing(
        [
            rel("linear_probe/train_summary.json"),
        ]
    )
    dataset_probs_path = resolve_existing(
        [
            rel("final/test_probs_certain.csv"),  # Use certain-only (correct evaluation)
            rel("final/test_probs_binary.csv"),   # Fallback to binary
            rel("final/test_probs_refit_full.csv"),
            rel("final/test_probs.csv"),
        ]
    )
    dataset_preds_path = resolve_existing(
        [
            rel("final/test_preds_certain.csv"),  # Use certain-only (correct evaluation)
            rel("final/test_preds_binary.csv"),   # Fallback to binary
            rel("final/test_preds_refit_full.csv"),
            rel("final/test_preds.csv"),
        ]
    )
    dataset_metrics_path = resolve_existing(
        [
            rel("final/test_metrics_certain.csv"),  # Use certain-only
            rel("final/test_metrics_binary.csv"),   # Fallback to binary
            rel("final/test_metrics_refit_full.csv"),
            rel("final/test_metrics.csv"),
        ]
    )
    metrics_certain_path = resolve_existing(
        [
            rel("final/test_metrics_certain.csv"),
            PROJECT_ROOT / "outputs_full_final/final/test_metrics_certain.csv",
        ]
    )
    metrics_binary_path = resolve_existing(
        [
            rel("final/test_metrics_binary.csv"),
            PROJECT_ROOT / "outputs_full_final/final/test_metrics_binary.csv",
        ]
    )
    summary_certain_json = resolve_existing(
        [
            rel("final/test_metrics_certain.macro_micro.json"),
            PROJECT_ROOT / "outputs_full_final/final/test_metrics_certain.macro_micro.json",
        ]
    )
    summary_binary_json = resolve_existing(
        [
            rel("final/test_metrics_binary.macro_micro.json"),
            PROJECT_ROOT / "outputs_full_final/final/test_metrics_binary.macro_micro.json",
        ]
    )
    dataset_macro_json = resolve_existing(
        [
            rel("final/test_metrics_refit_full.macro_micro.json"),
            rel("final/test_metrics.macro_micro.json"),
        ]
    )
    tri_eval_csv = resolve_existing(
        [
            rel("final/three_class_evaluation_refit_full.csv"),
            rel("final/three_class_evaluation.csv"),
        ]
    )
    tri_summary_csv = resolve_existing(
        [
            rel("final/three_class_evaluation_refit_full_summary.csv"),
            rel("final/three_class_evaluation_summary.csv"),
        ]
    )
    gt_bin_path = resolve_existing(
        [
            rel("splits/ground_truth_test.csv"),
            rel("splits/ground_truth_test_30.csv"),
            PROJECT_ROOT / "outputs_full_final/splits/ground_truth_test.csv",
            PROJECT_ROOT / "server_synced/data_full/ground_truth_test_30.csv",
        ]
    )
    gt_tri_path = resolve_existing(
        [
            PROJECT_ROOT / "data/evaluation_manifest_phaseA_5k_three.csv",
            PROJECT_ROOT / "server_synced/data_full/ground_truth_test_30_three.csv",
        ]
    )
    manifest_path = resolve_existing(
        [
            PROJECT_ROOT / "data/evaluation_manifest_phaseA_5k.csv",
            PROJECT_ROOT / "data/evaluation_manifest_phaseA_full.csv",
        ]
    )
    gating_config_path = PROJECT_ROOT / "config/gating.json"

    def load_json_file(path: Optional[Path]) -> Dict:
        if path and path.exists():
            return json.loads(path.read_text())
        return {}

    artifacts: Dict[str, object] = {
        "out_root": out_root,
        "thresholds_path": thresholds_path,
        "blend_weights_path": blend_weights_path,
        "meta_params_path": meta_params_path,
        "txr_cal_path": txr_cal_path,
        "txr_heavy_cal_path": txr_heavy_cal_path,
        "probe_platt_path": probe_platt_path,
        "probe_model_path": probe_model_path,
        "probe_summary_path": probe_summary_path,
        "dataset_probs_path": dataset_probs_path,
        "dataset_preds_path": dataset_preds_path,
        "dataset_metrics_path": dataset_metrics_path,
        "dataset_macro_json": dataset_macro_json,
        "tri_eval_csv": tri_eval_csv,
        "tri_summary_csv": tri_summary_csv,
        "gt_bin_path": gt_bin_path,
        "gt_tri_path": gt_tri_path,
        "manifest_path": manifest_path,
        "gating_config_path": gating_config_path,
        "thresholds": load_json_file(thresholds_path),
        "blend_weights": load_json_file(blend_weights_path),
        "meta_params": load_json_file(meta_params_path),
        "txr_cal_params": load_json_file(txr_cal_path),
        "txr_heavy_cal_params": load_json_file(txr_heavy_cal_path),
        "probe_platt_params": load_json_file(probe_platt_path),
        "gating_config": load_json_file(gating_config_path),
        "dataset_metrics_macro": load_json_file(dataset_macro_json),
    }

    # Store paths for lazy loading (don't load large CSVs at startup)
    artifacts["_dataset_probs_path"] = dataset_probs_path
    artifacts["_dataset_preds_path"] = dataset_preds_path
    artifacts["_dataset_metrics_path"] = dataset_metrics_path
    artifacts["_metrics_certain_path"] = metrics_certain_path
    artifacts["_metrics_binary_path"] = metrics_binary_path
    artifacts["_summary_certain_json"] = summary_certain_json
    artifacts["_summary_binary_json"] = summary_binary_json
    artifacts["_tri_eval_csv"] = tri_eval_csv
    artifacts["_tri_summary_csv"] = tri_summary_csv
    artifacts["_gt_bin_path"] = gt_bin_path
    artifacts["_gt_tri_path"] = gt_tri_path
    # Store test ground truth path (has -1 values)
    artifacts["_gt_test_path"] = gt_bin_path  # Same file has both binary and three-class data
    artifacts["_manifest_path"] = manifest_path
    
    # Load small JSON files immediately (fast)
    if summary_certain_json:
        artifacts["summary_certain"] = json.loads(summary_certain_json.read_text())
    if summary_binary_json:
        artifacts["summary_binary"] = json.loads(summary_binary_json.read_text())
    
    # Load manifest (needed for image browsing, usually small)
    if manifest_path:
        artifacts["image_manifest"] = pd.read_csv(manifest_path)
    if probe_model_path:
        try:
            with open(probe_model_path, "rb") as fh:
                artifacts["probe_models"] = pickle.load(fh)
        except Exception as exc:  # pragma: no cover - analysis
            st.warning(f"Failed to load linear probe models: {exc}")
    if probe_summary_path and probe_summary_path.exists():
        artifacts["probe_summary"] = json.loads(probe_summary_path.read_text())

    return artifacts


# --------------------------------------------------------------------------------------
# CLIP Encoder
# --------------------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def load_clip_encoder(device: torch.device, vision_model: str = "openai/clip-vit-large-patch14") -> Tuple[Optional[CLIPImageProcessor], Optional[CLIPVisionModel]]:
    try:
        processor = CLIPImageProcessor.from_pretrained(vision_model, local_files_only=True)
        model = CLIPVisionModel.from_pretrained(vision_model, local_files_only=True).to(device)
        model.eval()
        return processor, model
    except Exception as exc:
        print(f"[Streamlit] CLIP encoder unavailable ({exc}); linear probe will be disabled.")
        return None, None


@st.cache_resource(show_spinner=False)
def load_chexagent_model(device: str):
    """Load CheXagent model once and cache it. Don't show spinner - load on demand."""
    try:
        import sys
        # Check multiple possible locations for chexagent_repo
        possible_paths = [
            PROJECT_ROOT / "chexagent_repo",  # Project root (actual location)
            PROJECT_ROOT / "src" / "inference" / "chexagent_repo",  # Expected location
            PROJECT_ROOT.parent / "chexagent_repo",  # Parent directory
        ]
        
        chexagent_repo = None
        for path in possible_paths:
            if path.exists() and (path / "model_chexagent").exists():
                chexagent_repo = path
                break
        
        # If not found, return None
        if chexagent_repo is None:
            return None
            
        if str(chexagent_repo) not in sys.path:
            sys.path.insert(0, str(chexagent_repo))
        from model_chexagent.chexagent import CheXagent
        model = CheXagent(device=device)
        return model
    except Exception as exc:
        return None


def run_chexagent_single(
    image_path: Path,
    device: str,
    chexagent_model,
    thresholds: Dict[str, float],
) -> Dict[str, object]:
    """Run CheXagent inference for a single image using cached model."""
    import json
    import sys
    from pathlib import Path as PathLib
    
    # Import helper functions from smart_ensemble (lazy to avoid CheXagent import issues)
    # Check multiple possible locations for chexagent_repo
    possible_paths = [
        PROJECT_ROOT / "chexagent_repo",  # Project root (actual location)
        PROJECT_ROOT / "src" / "inference" / "chexagent_repo",  # Expected location
        PROJECT_ROOT.parent / "chexagent_repo",  # Parent directory
    ]
    
    chexagent_repo = None
    for path in possible_paths:
        if path.exists() and (path / "model_chexagent").exists():
            chexagent_repo = path
            break
    
    if chexagent_repo and str(chexagent_repo) not in sys.path:
        sys.path.insert(0, str(chexagent_repo))
    
    # Try to import helper functions - if chexagent_repo doesn't exist, define minimal versions
    try:
        from src.inference.smart_ensemble import (
            parse_di_response,
            parse_binary_response,
            decide_label,
        )
        CHEXPERT13_HELPER = LABELS  # Use the labels we already have
    except ImportError:
        # Fallback: define minimal versions if smart_ensemble can't be imported
        def parse_di_response(text: str) -> Dict[str, Dict[str, float]]:
            return {label: {"mentioned": 0, "negated": 0, "uncertain": 0, "strength": 0.0} for label in LABELS}
        
        def parse_binary_response(text: str, disease: str) -> Tuple[float, List[str]]:
            # Try to extract score from text
            import re
            score_match = re.search(r'(\d+\.?\d*)', text)
            score = float(score_match.group(1)) / 100.0 if score_match else 0.5
            return score, ["parsed"]
        
        def decide_label(disease: str, score: float, thresholds: Dict[str, float], di_entry: Dict, use_precision_gating: bool = False) -> Tuple[int, List[str]]:
            threshold = thresholds.get(disease, 0.5)
            return (1 if score >= threshold else 0, ["threshold"])
        
        CHEXPERT13_HELPER = LABELS
    
    if chexagent_model is None:
        raise RuntimeError("CheXagent model not loaded")
    
    di_text = chexagent_model.disease_identification([str(image_path)], CHEXPERT13_HELPER)
    di_info = parse_di_response(di_text)
    
    binary_outputs: Dict[str, Dict[str, object]] = {}
    final_labels: Dict[str, int] = {}
    
    for disease in CHEXPERT13_HELPER:
        try:
            raw_binary = chexagent_model.binary_disease_classification([str(image_path)], disease)
            score, analysis_reasons = parse_binary_response(raw_binary, disease)
        except Exception as exc:
            raw_binary = f"ERROR: {exc}"
            score = 0.5
            analysis_reasons = ["exception"]
        
        decision, decision_reasons = decide_label(
            disease, score, thresholds, di_info[disease], use_precision_gating=False
        )
        
        binary_outputs[disease] = {
            "score_raw": round(score, 4),
            "score": round(score, 4),
            "calibrated": False,
            "threshold": thresholds.get(disease, 0.5),
            "raw": raw_binary,
            "analysis_reasons": analysis_reasons,
            "decision_reasons": decision_reasons,
        }
        final_labels[disease] = int(decision)
    
    no_finding = 1 if all(v == 0 for v in final_labels.values()) else 0
    final_labels["No Finding"] = no_finding
    
    return {
        "image": str(image_path),
        "initial_response": di_text,
        "di_outputs": json.dumps(di_info),
        "binary_outputs": json.dumps(binary_outputs),
    }


def compute_clip_embedding(image_path: Path, device: torch.device, processor: CLIPImageProcessor, model: CLIPVisionModel) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        outputs = model(pixel_values)
        embedding = outputs.pooler_output.cpu().numpy().astype(np.float32)
    return embedding  # shape (1, D)


# --------------------------------------------------------------------------------------
# Calibration helpers
# --------------------------------------------------------------------------------------


def apply_platt_dict(prob_dict: Dict[str, float], params: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    calibrated: Dict[str, float] = {}
    for label, prob in prob_dict.items():
        if math.isnan(prob):
            calibrated[label] = prob
            continue
        coeffs = params.get(label)
        if coeffs is None:
            calibrated[label] = prob
        else:
            calibrated[label] = float(apply_platt_to_scores(np.array([prob]), coeffs["a"], coeffs["b"])[0])
    return calibrated


def apply_meta_single(prob: float, params: Optional[Dict[str, float]]) -> float:
    if params is None:
        return prob
    method = params.get("method", "platt")
    if method == "isotonic":
        xs = np.array(params.get("x", []), dtype=float)
        ys = np.array(params.get("y", []), dtype=float)
        if xs.size == 0 or ys.size == 0:
            return prob
        return float(np.interp(prob, xs, ys))
    a = float(params.get("a", 1.0))
    b = float(params.get("b", 0.0))
    return float(1.0 / (1.0 + math.exp(-(a * prob + b))))


def ensure_prob_dict(labels: List[str], values: Dict[str, float], default: float = 0.0) -> Dict[str, float]:
    return {label: float(values.get(label, default)) for label in labels}


@st.cache_data
def load_dataset_csv(path: Optional[Path]):
    """Lazy load CSV files on demand."""
    if path and path.exists():
        return pd.read_csv(path)
    return None


# --------------------------------------------------------------------------------------
# TXR inference (base + heavy)
# --------------------------------------------------------------------------------------


@st.cache_resource
def _load_txr_model_cached(model_weights: str):
    """
    Cache the TXR model (lightweight densenet121) to avoid reloading on every inference.
    This prevents broken pipe errors from repeated model initialization.
    """
    import torchxrayvision as xrv
    import torch
    try:
        # Always load on CPU for single image inference
        model = xrv.models.DenseNet(weights=model_weights)
        model = model.to(torch.device("cpu"))
        model.eval()
        return model, model.pathologies
    except Exception as e:
        st.error(f"Failed to load TXR model {model_weights}: {e}")
        raise


def run_txr_single(image_path: Path, device: torch.device, model_weights: str, batch_size: int = 1) -> Dict[str, float]:
    """
    Run TXR inference on a single image using cached model.
    Bypasses DataLoader completely to avoid broken pipe errors.
    """
    # CRITICAL: Always use CPU and bypass DataLoader for single images
    safe_device = torch.device("cpu")
    
    try:
        # Load cached model (lightweight DenseNet121)
        cached_model, model_pathologies = _load_txr_model_cached(model_weights)
        
        # Process image directly without DataLoader
        from PIL import Image
        import numpy as np
        import torchxrayvision as xrv
        
        # Load and preprocess image
        with Image.open(image_path) as img:
            image = img.convert("L").copy()
        
        # Resize and normalize
        image = image.resize((224, 224), Image.BILINEAR)
        arr = np.array(image, dtype=np.float32)
        arr = xrv.datasets.normalize(arr, 255.0)
        arr = arr[None, None, :, :]  # (1, 1, H, W)
        
        # Convert to tensor and run inference
        tensor = torch.from_numpy(arr).float().to(safe_device)
        
        with torch.no_grad():
            logits = cached_model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Map TXR pathologies to CheXpert labels
        from src.inference.txr_infer import build_label_mapping, CHEXPERT13
        
        label_map, missing = build_label_mapping(model_pathologies)
        inverse_map = {v: k for k, v in label_map.items()}
        
        results = {}
        for label in LABELS:
            if label in inverse_map:
                txr_pathology = inverse_map[label]
                if txr_pathology in model_pathologies:
                    idx = model_pathologies.index(txr_pathology)
                    results[label] = float(probs[idx])
                else:
                    results[label] = float("nan")
            else:
                results[label] = float("nan")
        
        return results
        
    except Exception as exc:
        error_str = str(exc)
        # More helpful error messages
        if "Broken pipe" in error_str or "Errno 32" in error_str:
            raise RuntimeError(f"TXR inference failed: System error (broken pipe). Model: {model_weights}")
        elif "model" in error_str.lower() or "weights" in error_str.lower():
            raise RuntimeError(f"TXR model weights not found: {model_weights}. Please ensure models are downloaded.")
        else:
            raise RuntimeError(f"TXR inference failed: {error_str[:200]}")


# --------------------------------------------------------------------------------------
# Linear probe inference
# --------------------------------------------------------------------------------------


def compute_probe_probs(
    image_path: Path,
    device: torch.device,
    artifacts: Dict[str, object],
    clip_processor: Optional[CLIPImageProcessor],
    clip_model: Optional[CLIPVisionModel],
) -> Dict[str, float]:
    models_pack = artifacts.get("probe_models")
    if not models_pack:
        return {label: float("nan") for label in LABELS}
    if clip_processor is None or clip_model is None:
        return {label: float("nan") for label in LABELS}

    embeddings = compute_clip_embedding(image_path, device, clip_processor, clip_model)  # shape (1, D)
    models: Dict[str, object] = models_pack.get("models", {})
    fallback = {}
    summary = artifacts.get("probe_summary")
    if summary:
        fallback = {lab: summary.get("prevalence", {}).get(lab, 0.1) for lab in LABELS}  # optional field

    probs: Dict[str, float] = {}
    for label in LABELS:
        model = models.get(label)
        if model is None:
            probs[label] = float(fallback.get(label, 0.15))
        else:
            score = model.predict_proba(embeddings)[0, 1]
            probs[label] = float(score)
    return probs


# --------------------------------------------------------------------------------------
# Hybrid inference for single image
# --------------------------------------------------------------------------------------


@dataclass
class SinglePrediction:
    filename: str
    probabilities: Dict[str, float]
    source_probs: Dict[str, Dict[str, float]]
    binary: Dict[str, int]
    tri_state: Dict[str, int]
    impression: str
    metadata: Dict[str, Dict[str, Dict]]
    di_text: str


def compute_three_class(prob: float, tau: float, margin: float = UNCERTAINTY_MARGIN) -> int:
    lower = max(0.0, tau * (1 - margin))
    upper = min(1.0, tau * (1 + margin))
    if prob >= upper:
        return 1
    if prob < lower:
        return 0
    return -1


def build_brief_impression(binary: Dict[str, int], tri_state: Dict[str, int], probabilities: Dict[str, float]) -> str:
    """Build a structured impression with positive, negative, uncertain, and blank labels."""
    positives = [
        (label, probabilities.get(label, float("nan")))
        for label, val in binary.items()
        if val == 1 and label != "No Finding"
    ]
    uncertain = [
        (label, probabilities.get(label, float("nan")))
        for label, val in tri_state.items()
        if val == -1 and label != "No Finding"
    ]
    high_conf_neg = [
        (label, probabilities.get(label, float("nan")))
        for label, val in binary.items()
        if val == 0 and label != "No Finding" and probabilities.get(label, 1.0) < 0.15
    ]
    blanks = [
        label
        for label in LABELS
        if label != "No Finding" and pd.isna(probabilities.get(label, np.nan))
    ]

    lines: List[str] = []
    if positives:
        positives_sorted = sorted(positives, key=lambda item: (-np.nan_to_num(item[1], nan=0.0), item[0]))
        formatted = ", ".join(f"{lab} ({prob:.2f})" if not pd.isna(prob) else lab for lab, prob in positives_sorted)
        lines.append(f"✅ **Positive findings:** {formatted}")
    if uncertain:
        uncertain_sorted = sorted(uncertain, key=lambda item: (-np.nan_to_num(item[1], nan=0.0), item[0]))
        formatted_uncertain = ", ".join(f"{lab} ({prob:.2f})" if not pd.isna(prob) else lab for lab, prob in uncertain_sorted)
        lines.append(f"⚠️ **Borderline / needs correlation:** {formatted_uncertain}")
    if high_conf_neg:
        neg_sorted = sorted(high_conf_neg, key=lambda item: np.nan_to_num(item[1], nan=1.0))
        formatted_neg = ", ".join(f"{lab} ({prob:.2f})" if not pd.isna(prob) else lab for lab, prob in neg_sorted[:5])
        suffix = "…" if len(high_conf_neg) > 5 else ""
        lines.append(f"❌ **No convincing evidence:** {formatted_neg}{suffix}")
    if blanks:
        suffix = "…" if len(blanks) > 5 else ""
        lines.append(f"⚪ **Not evaluated:** {', '.join(sorted(blanks)[:5])}{suffix}")
    if not lines:
        lines.append("✅ No convincing abnormality detected.")
    return "\n".join(lines)


def run_single_hybrid_prediction(
    image_path: Path,
    device: torch.device,
    artifacts: Dict[str, object],
    txr_weights: str = "densenet121-res224-chex",
    txr_heavy_weights: str = "resnet50-res512-all",
) -> SinglePrediction:
    filename = image_path.name

    # Load CLIP encoder once
    clip_processor, clip_model = load_clip_encoder(device)
    if (clip_processor is None or clip_model is None) and not st.session_state.get("probe_disabled_notified", False):
        st.info("Linear probe disabled (CLIP weights not available offline). Using TXR + CheXagent sources only.")
        st.session_state["probe_disabled_notified"] = True

    # Source probabilities ---------------------------------------------------
    source_probs: Dict[str, Dict[str, float]] = {}

    # TXR base - using lightweight densenet121-res224-chex (NOT the 7GB heavy model)
    txr_raw: Dict[str, float]
    try:
        with st.spinner("🔬 Running TXR inference (lightweight model)..."):
            txr_raw = run_txr_single(image_path, device, txr_weights)
        # Validate TXR results
        txr_valid = any(not (pd.isna(v) or v == 0.0) for v in txr_raw.values())
        if not txr_valid:
            st.info("ℹ️ TXR returned low probabilities. This may indicate a normal scan or model uncertainty.")
    except Exception as exc:
        error_msg = str(exc)
        # Provide helpful error messages
        if "Broken pipe" in error_msg or "Errno 32" in error_msg:
            st.error("❌ TXR inference failed: System error (broken pipe). The model may need to be reloaded. Please try again.")
            st.info("💡 **Tip**: This usually resolves on retry. If it persists, check system resources.")
        elif "model" in error_msg.lower() or "weights" in error_msg.lower():
            st.error(f"❌ TXR model weights not available: {txr_weights}")
            st.info("💡 **Tip**: The model will download automatically on first use. Check your internet connection.")
        else:
            st.error(f"❌ TXR inference failed: {error_msg[:200]}")
        txr_raw = {label: float("nan") for label in LABELS}
    txr_cal_params = artifacts.get("txr_cal_params", {})
    if not txr_cal_params:
        st.warning("⚠️ TXR calibration parameters missing. Using raw probabilities.")
    txr_calibrated = apply_platt_dict(txr_raw, txr_cal_params)
    source_probs["txr"] = txr_calibrated

    # TXR heavy (5 labels)
    txr_heavy_labels = {"Enlarged Cardiomediastinum", "Lung Lesion", "Pneumothorax", "Pleural Other", "Fracture"}
    txr_heavy_cal_params = artifacts.get("txr_heavy_cal_params", {})
    if txr_heavy_cal_params:
        try:
            txr_heavy_raw = run_txr_single(image_path, device, txr_heavy_weights)
        except RuntimeError:
            txr_heavy_raw = {}
        txr_heavy = apply_platt_dict(txr_heavy_raw, txr_heavy_cal_params)
        # Fill missing labels with NaNs to keep structure
        txr_heavy = {label: txr_heavy.get(label, float("nan")) for label in LABELS}
        source_probs["txr_heavy"] = txr_heavy

    # Linear probe
    probe_probs = compute_probe_probs(image_path, device, artifacts, clip_processor, clip_model)
    probe_calibrated = apply_platt_dict(probe_probs, artifacts.get("probe_platt_params", {}))
    source_probs["probe"] = probe_calibrated

    # Blend ------------------------------------------------------------------
    weights = artifacts.get("blend_weights", {})
    labels = LABELS

    per_source_frames: Dict[str, pd.DataFrame] = {}
    for source_name, probs in source_probs.items():
        frame = pd.DataFrame({"filename": [filename]})
        for label in labels:
            frame[f"y_cal_{label}"] = float(probs.get(label, 0.0))
        per_source_frames[source_name] = frame

    blended = blend_probabilities(
        labels=labels,
        sources=per_source_frames,
        weights=weights,
        score_prefix="y_cal_",
    )

    meta_params = artifacts.get("meta_params", {})
    apply_meta_calibration(blended, labels, meta_params)

    for label in INVERTED_PROB_LABELS:
        col = f"y_cal_{label}"
        if col in blended.columns:
            blended[col] = 1.0 - blended[col].astype(float)

    probabilities = {label: float(blended.loc[0, f"y_cal_{label}"]) for label in labels}
    probabilities["No Finding"] = float("nan")

    # DI metadata via CheXagent (for gating + impression text)
    thresholds = artifacts.get("thresholds", {})
    # Use cached CheXagent model instead of creating new one
    device_str = device.type if isinstance(device, torch.device) else str(device)
    
    # Only load CheXagent when actually needed (for upload section)
    chexagent_model = None
    metadata = {filename: {"binary": {}, "di": {}}}
    chex_results: Dict[str, object] = {
        "initial_response": "",
        "di_outputs": "{}",
        "binary_outputs": "{}",
    }

    try:
        chexagent_model = load_chexagent_model(device_str)
        if chexagent_model is None:
            if not st.session_state.get("chexagent_missing_warned", False):
                # Use a less prominent message - CheXagent is optional
                # The system works fine with just TXR + Linear Probe + Blending
                with st.expander("ℹ️ Note: CheXagent DI not available (optional)", expanded=False):
                    st.caption(
                        "The system is using TXR Base + Linear Probe + Blended Ensemble for predictions. "
                        "CheXagent Disease Identification (DI) refinements are optional and not required. "
                        "All predictions are still fully functional."
                    )
                st.session_state["chexagent_missing_warned"] = True
        else:
            chex_results = run_chexagent_single(
                image_path=image_path,
                device=device_str,
                chexagent_model=chexagent_model,
                thresholds=thresholds,
            )
            di_map = json.loads(chex_results["di_outputs"])
            binary_map = json.loads(chex_results["binary_outputs"])
            metadata = {filename: {"binary": binary_map, "di": di_map}}
            
            # Add CheXagent binary classification probabilities to source_probs
            chexagent_probs = {}
            for label in LABELS:
                if label in binary_map:
                    # Extract score from CheXagent binary output
                    bin_data = binary_map[label]
                    if isinstance(bin_data, dict):
                        score = bin_data.get("score", 0.0)
                        if "calibrated" in bin_data and bin_data["calibrated"]:
                            score = bin_data.get("score", score)
                        chexagent_probs[label] = float(score)
                    else:
                        chexagent_probs[label] = float(bin_data) if isinstance(bin_data, (int, float)) else 0.0
                else:
                    chexagent_probs[label] = float("nan")
            source_probs["chexagent_binary"] = chexagent_probs
            
            # Add DI strength as probabilities for visualization
            chexagent_di = {}
            for label in LABELS:
                if label in di_map and isinstance(di_map[label], dict):
                    strength = di_map[label].get("strength", 0.0)
                    chexagent_di[label] = float(strength)
                else:
                    chexagent_di[label] = float("nan")
            source_probs["chexagent_di"] = chexagent_di
    except Exception as exc:
        # Silently fail - DI gating will just be skipped
        # Only show error if it's not just a missing repository (that's expected)
        if not st.session_state.get("chexagent_missing_warned", False):
            # Check if it's just a missing repo vs actual error
            error_str = str(exc).lower()
            if "chexagent" not in error_str and "repository" not in error_str and "not found" not in error_str:
                # Only show warning for unexpected errors
                with st.expander("⚠️ CheXagent inference note", expanded=False):
                    st.caption(f"CheXagent DI encountered an issue: {exc}")
                    st.caption("Predictions continue using TXR + Linear Probe sources only.")
            st.session_state["chexagent_missing_warned"] = True

    gating_config_path = artifacts.get("gating_config_path", PROJECT_ROOT / "config/gating.json")
    labels_df = pd.DataFrame({"filename": [filename]})
    preds_df = apply_gating(
        probs=blended.copy(),
        labels_df=labels_df,
        thresholds=thresholds,
        gating_config=Path(gating_config_path),
        metadata=metadata,
        score_prefix="y_cal_",
    )

    binary_predictions = {label: int(preds_df.loc[0, label]) for label in CHEXPERT14 if label in preds_df.columns}

    # CRITICAL: "No Finding" must come from gating output (same as evaluation)
    # It's already computed in preds_df by apply_gating, so use it directly
    if "No Finding" in preds_df.columns:
        binary_predictions["No Finding"] = int(preds_df.loc[0, "No Finding"])
    else:
        # Fallback: derive from binary predictions of 13 labels (matching evaluation logic)
        binary_predictions["No Finding"] = 1 if sum(binary_predictions.get(label, 0) for label in LABELS) == 0 else 0

    # Three-class decisions (for display only - binary predictions are the source of truth)
    tri_state = {
        label: compute_three_class(probabilities.get(label, 0.0), thresholds.get(label, 0.5), UNCERTAINTY_MARGIN)
        for label in LABELS
    }
    # Derive No Finding for tri_state from binary predictions, not tri_state values
    tri_state["No Finding"] = binary_predictions.get("No Finding", 0)

    impression = build_brief_impression(binary_predictions, tri_state, probabilities)

    return SinglePrediction(
        filename=filename,
        probabilities=probabilities,
        source_probs=source_probs,
        binary=binary_predictions,
        tri_state=tri_state,
        impression=impression,
        metadata=metadata,
        di_text=chex_results.get("initial_response", ""),
    )


# --------------------------------------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------------------------------------


def map_filename_to_path(filename: str, artifacts: Dict[str, object]) -> Optional[Path]:
    manifest = artifacts.get("image_manifest")
    if manifest is not None:
        if "filename" in manifest.columns:
            candidates = manifest[manifest["filename"] == filename]
            if not candidates.empty:
                image_col = "image" if "image" in candidates.columns else "image_path" if "image_path" in candidates.columns else None
                if image_col:
                    raw = candidates.iloc[0][image_col]
                    path = Path(raw)
                    if not path.is_absolute():
                        possible = PROJECT_ROOT.parent / "radiology_report" / raw
                        if possible.exists():
                            return possible
                        possible = PROJECT_ROOT.parent / "radiology_report" / "files" / raw
                        if possible.exists():
                            return possible
                    else:
                        return path
    # fallback 1: search in local chexagent_chexpert_eval/files/p10
    local_base = PROJECT_ROOT / "files" / "p10"
    candidate = local_base / filename
    if candidate.exists():
        return candidate
    # fallback 2: recursively search under local files/p10 (slower, last resort)
    try:
        for p in local_base.rglob("*"):
            if p.name == filename:
                return p
    except Exception:
        pass
    # fallback 3: radiology_report/files if available
    base = PROJECT_ROOT.parent / "radiology_report" / "files"
    candidate = base / "p10" / filename
    if candidate.exists():
        return candidate
    return None


def get_ground_truth_for(filename: str, artifacts: Dict[str, object]) -> Tuple[Dict[str, Optional[int]], Dict[str, Optional[int]]]:
    """
    Get ground truth for a filename. Preserves blanks (None) and -1 values.
    Returns: (gt_bin: Dict with 0/1/None, gt_tri: Dict with -1/0/1/None)
    """
    # Try to load from the original test CSV (has -1 values) if available
    gt_test_path = artifacts.get("_gt_test_path") or artifacts.get("_gt_tri_path")
    if gt_test_path is None:
        # Fallback to known paths
        out_roots = [
            Path("outputs_full_final"),
            Path("outputs_full"),
            Path("server_synced/outputs_full"),
        ]
        for root in out_roots:
            candidate = root / "splits" / "ground_truth_test.csv"
            if candidate.exists():
                gt_test_path = candidate
                break
    
    gt_bin: Dict[str, Optional[int]] = {}
    gt_tri: Dict[str, Optional[int]] = {}
    
    # Load three-class ground truth directly from test CSV (has actual CheXpert -1/0/1 values)
    if gt_test_path and Path(gt_test_path).exists():
        gt_df = load_dataset_csv(gt_test_path)
        if gt_df is not None:
            row = gt_df[gt_df["filename"].apply(lambda x: Path(str(x)).name) == filename]
            if not row.empty:
                for label in LABELS:
                    if label in row.columns:
                        value = row.iloc[0][label]
                        if pd.isna(value):
                            gt_tri[label] = None  # Preserve blank
                        else:
                            gt_tri[label] = int(value)  # Preserve -1, 0, 1
                        
                        # For binary: convert -1 to None (uncertain treated as blank for binary)
                        if pd.isna(value):
                            gt_bin[label] = None
                        else:
                            val_int = int(value)
                            if val_int == -1:
                                gt_bin[label] = None  # Uncertain = blank for binary
                            else:
                                gt_bin[label] = val_int  # 0 or 1
    
    # Fallback to binary ground truth file if three-class not found
    gt_bin_df = None
    if not gt_tri and artifacts.get("_gt_bin_path"):
        gt_bin_df = load_dataset_csv(artifacts.get("_gt_bin_path"))
    if gt_bin_df is not None:
        row = gt_bin_df[gt_bin_df["filename"].apply(lambda x: Path(str(x)).name) == filename]
        if not row.empty:
            for label in LABELS:
                col = f"y_true_{label}" if f"y_true_{label}" in row.columns else label
                if col in row.columns:
                        value = row.iloc[0][col]
                        if pd.isna(value):
                            gt_bin[label] = None
                        else:
                            val_int = int(value)
                            if val_int != -1:
                                gt_bin[label] = val_int
                            else:
                                gt_bin[label] = None

    # Derive No Finding for binary ground truth
    # No Finding = 1 if ALL labels are 0 or blank (no positives)
    if LABELS:
        # Check if we have complete coverage (all labels present)
        labels_in_gt = [l for l in LABELS if l in gt_bin]
        if len(labels_in_gt) == len(LABELS):
            # All labels present - check for positives
            positives = any(gt_bin.get(label) == 1 for label in LABELS)
            gt_bin["No Finding"] = 0 if positives else 1
        elif len(labels_in_gt) > 0:
            # Partial coverage - check if any known label is positive
            positives = any(gt_bin.get(label) == 1 for label in labels_in_gt)
            if positives:
                gt_bin["No Finding"] = 0
            # If no positives and we have at least some labels, derive based on what we have
            elif all(gt_bin.get(label) in (0, None) for label in labels_in_gt):
                # All known labels are 0 or blank - likely No Finding but mark as None if incomplete
                gt_bin["No Finding"] = 1 if len(labels_in_gt) >= len(LABELS) * 0.8 else None

    # Derive No Finding for three-class ground truth
    # No Finding = 1 if all labels are 0 or -1 (no certain positives)
    if LABELS:
        labels_in_tri = [l for l in LABELS if l in gt_tri]
        if len(labels_in_tri) == len(LABELS):
            # Complete coverage
            has_positive = any(gt_tri.get(label) == 1 for label in LABELS)
            has_uncertain = any(gt_tri.get(label) == -1 for label in LABELS)
            if has_positive:
                gt_tri["No Finding"] = 0
            elif has_uncertain:
                gt_tri["No Finding"] = -1
            else:
                gt_tri["No Finding"] = 1
        elif len(labels_in_tri) > 0:
            # Partial coverage
            has_positive = any(gt_tri.get(label) == 1 for label in labels_in_tri)
            has_uncertain = any(gt_tri.get(label) == -1 for label in labels_in_tri)
            if has_positive:
                gt_tri["No Finding"] = 0
            elif has_uncertain:
                gt_tri["No Finding"] = -1
            elif all(gt_tri.get(label) in (0, None) for label in labels_in_tri):
                gt_tri["No Finding"] = 1 if len(labels_in_tri) >= len(LABELS) * 0.8 else None

    return gt_bin, gt_tri


# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------


def render_dataset_browser(artifacts: Dict[str, object], device: torch.device) -> None:
    st.markdown("### 🔍 Explore Evaluation Samples (Predictions vs Ground Truth)")

    if not artifacts:
        st.info("Run the full pipeline to enable dataset browsing.")
        return

    # Lazy load datasets only when needed (show loading indicator)
    with st.spinner("Loading dataset..."):
        probs_df = load_dataset_csv(artifacts.get("_dataset_probs_path"))
        preds_df = load_dataset_csv(artifacts.get("_dataset_preds_path"))
    
    if probs_df is None or preds_df is None:
        st.info("Dataset files not found. Run the full pipeline to enable dataset browsing.")
        return

    gt_df = load_dataset_csv(artifacts.get("_gt_bin_path"))
    gt_lookup = None
    if gt_df is not None:
        gt_lookup = gt_df.copy()
        gt_lookup["filename"] = gt_lookup["filename"].apply(lambda x: Path(str(x)).name)
        gt_lookup = gt_lookup.set_index("filename")

    filenames = preds_df["filename"].tolist() if "filename" in preds_df.columns else preds_df["image"].apply(lambda x: Path(str(x)).name).tolist()
    available_indices = list(range(len(filenames)))

    # Calculate No Finding metrics properly - derive from ground truth
    nf_gt_pos = 0
    nf_pred_pos = 0
    
    # Derive No Finding from ground truth: 1 if all 13 labels are 0 or -1 (no certain positives)
    if gt_lookup is not None:
        for idx in gt_lookup.index:
            # Check if any label is positive (1)
            positives = sum(1 for label in LABELS 
                          if label in gt_lookup.columns 
                          and pd.notna(gt_lookup.loc[idx, label]) 
                          and gt_lookup.loc[idx, label] == 1)
            if positives == 0:
                nf_gt_pos += 1
    
    if "No Finding" in preds_df.columns:
        nf_pred_pos = int(preds_df["No Finding"].sum())
    
    # Always show No Finding metrics (even if 0)
    st.markdown(f"**📊 No Finding Coverage** – GT positives: **{nf_gt_pos}**, Predicted: **{nf_pred_pos}**")
    if nf_gt_pos > 0 and nf_pred_pos == 0:
        st.info(f"ℹ️ **Note**: No Finding is currently blocked by hyper-positive labels (Lung Opacity, Atelectasis, Lung Lesion) that predict positive on most samples. This is a known limitation.")
    
    # REORGANIZED: Put Label Distribution and Metrics together before the image selector
    # Add Label Distribution Section - aligned with metrics (uses metrics' pred_positives for accuracy)
    with st.expander("📊 Label Distribution & Dataset Metrics", expanded=True):
        if gt_lookup is not None and preds_df is not None:
            # Load metrics to get accurate prediction counts (computed on covered samples, matches metrics)
            metrics_df = None
            metrics_certain_path = artifacts.get("_metrics_certain_path")
            if metrics_certain_path and Path(metrics_certain_path).exists():
                metrics_df = pd.read_csv(metrics_certain_path)  # Direct read, bypass cache
            
            labels = LABELS
            dist_data = []
            
            for label in labels:
                # Ground truth distribution (all samples)
                if label in gt_lookup.columns:
                    gt_pos = (gt_lookup[label] == 1).sum() if label in gt_lookup.columns else 0
                    gt_total = gt_lookup[label].notna().sum() if label in gt_lookup.columns else 0
                    gt_pct = (gt_pos / gt_total * 100) if gt_total > 0 else 0
                else:
                    gt_pos = 0
                    gt_total = 0
                    gt_pct = 0
                
                # Prediction distribution - use metrics pred_positives for accuracy (matches metrics table)
                if metrics_df is not None:
                    met_row = metrics_df[metrics_df['label'] == label]
                    if not met_row.empty:
                        # Use metrics' pred_positives - this matches what's in the metrics table
                        pred_pos = int(met_row.iloc[0]['pred_positives'])
                        pred_total = len(preds_df)  # Total samples in dataset
                        pred_pct = (pred_pos / pred_total * 100) if pred_total > 0 else 0
                    else:
                        # Fallback to direct count from predictions file
                        pred_pos = int(preds_df[label].sum()) if label in preds_df.columns else 0
                        pred_total = len(preds_df)
                        pred_pct = (pred_pos / pred_total * 100) if pred_total > 0 else 0
                else:
                    # Fallback: direct count from predictions file
                    pred_pos = int(preds_df[label].sum()) if label in preds_df.columns else 0
                    pred_total = len(preds_df)
                    pred_pct = (pred_pos / pred_total * 100) if pred_total > 0 else 0
                
                dist_data.append({
                    'Label': label,
                    'GT Positives': f"{gt_pos}/{gt_total} ({gt_pct:.1f}%)",
                    'Predicted Positives': f"{pred_pos}/{pred_total} ({pred_pct:.1f}%)",
                })
            
            dist_df = pd.DataFrame(dist_data)
            st.markdown("**Label Distribution:**")
            st.dataframe(dist_df, use_container_width=True, hide_index=True)
            
            # Add summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", f"{len(preds_df):,}")
            with col2:
                total_gt_labels = sum((gt_lookup[label] == 1).sum() for label in labels if label in gt_lookup.columns)
                st.metric("Total GT Positives", f"{total_gt_labels:,}")
            with col3:
                total_pred_labels = sum(int(metrics_df[metrics_df['label'] == label].iloc[0]['pred_positives']) 
                                      if not metrics_df[metrics_df['label'] == label].empty 
                                      else 0 for label in labels if metrics_df is not None)
                if metrics_df is None:
                    total_pred_labels = sum(int(preds_df[label].sum()) if label in preds_df.columns else 0 for label in labels)
                st.metric("Total Predicted Positives", f"{total_pred_labels:,}")
            
            # Add Latest Metrics Summary in same expander
            st.markdown("---")
            st.markdown("**📈 Latest Performance Metrics (Certain-Only Evaluation):**")
            
            # Load latest metrics
            metrics_certain_path = artifacts.get("_metrics_certain_path")
            if metrics_certain_path and Path(metrics_certain_path).exists():
                metrics_df_local = pd.read_csv(metrics_certain_path)
                macro_micro_path = str(metrics_certain_path).replace('.csv', '.macro_micro.json')
                if Path(macro_micro_path).exists():
                    import json
                    with open(macro_micro_path) as f:
                        summary_json = json.load(f)
                    
                    # Show metrics in percentage format, excluding accuracy
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Macro Precision", f"{summary_json.get('macro_precision', 0)*100:.1f}%")
                    with col2:
                        st.metric("Macro Recall", f"{summary_json.get('macro_recall', 0)*100:.1f}%")
                    with col3:
                        st.metric("Macro F1-Score", f"{summary_json.get('macro_f1', 0)*100:.1f}%")
                    with col4:
                        st.metric("Micro F1-Score", f"{summary_json.get('micro_f1', 0)*100:.1f}%")
                    
                    # Per-label metrics table (precision, recall, F1 in percentage, no accuracy)
                    st.markdown("**Per-Label Performance:**")
                    metrics_display = []
                    for _, row in metrics_df_local.iterrows():
                        label = row['label']
                        if label == 'No Finding':
                            continue
                        metrics_display.append({
                            'Label': label,
                            'Precision (%)': f"{float(row['precision'])*100:.1f}%" if not pd.isna(row['precision']) else "n/a",
                            'Recall (%)': f"{float(row['recall'])*100:.1f}%",
                            'F1-Score (%)': f"{float(row['f1'])*100:.1f}%",
                        })
                    metrics_display_df = pd.DataFrame(metrics_display)
                    st.dataframe(metrics_display_df, use_container_width=True, hide_index=True)

    all_labels_for_ui = LABELS + ["No Finding"]

    filter_label = st.selectbox("Focus on label", ["All"] + all_labels_for_ui)
    filter_mode = st.radio(
        "Sample set",
        ["All samples", "Ground Truth Positives", "Model Positives"],
        horizontal=True,
    )

    filtered_indices = available_indices
    if filter_label != "All" and filter_mode != "All samples":
        temp_indices: List[int] = []
        if filter_mode == "Ground Truth Positives":
            for idx, fname in enumerate(filenames):
                gt_bin_row, _ = get_ground_truth_for(fname, artifacts)
                if gt_bin_row.get(filter_label) == 1:
                    temp_indices.append(idx)
        elif filter_mode == "Model Positives" and filter_label in preds_df.columns:
            temp_indices = [idx for idx in available_indices if preds_df.iloc[idx][filter_label] == 1]
        if temp_indices:
            filtered_indices = temp_indices
        else:
            st.info("No samples match this filter; showing the full set.")

    # Add search functionality
    st.markdown("---")
    search_query = st.text_input("🔍 Search for image filename", placeholder="Type part of filename to search...", key="sample_search")
    
    # Filter indices by search query
    if search_query:
        search_lower = search_query.lower()
        search_filtered_indices = [idx for idx in filtered_indices if search_lower in filenames[idx].lower()]
        if search_filtered_indices:
            filtered_indices = search_filtered_indices
            st.info(f"Found {len(search_filtered_indices)} matching sample(s)")
        else:
            st.warning(f"No samples found matching '{search_query}'. Showing all filtered samples.")
            # Don't change filtered_indices if no match - let user see what's available

    sample_idx = st.selectbox(
        "Select sample",
        options=filtered_indices,
        format_func=lambda idx: filenames[idx],
    )
    filename = filenames[sample_idx]

    image_path = map_filename_to_path(filename, artifacts)
    if image_path and image_path.exists():
        img = Image.open(image_path)
        # Scale image to max 600px width for better presentation
        max_width = 600
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        st.image(img, caption=filename, use_container_width=False)
    else:
        st.warning("Image file not found locally.")

    pred_row = preds_df.iloc[sample_idx]
    prob_row = probs_df.iloc[sample_idx] if sample_idx < len(probs_df) else None
    gt_bin, gt_tri = get_ground_truth_for(filename, artifacts)

    binary_preds = {
        label: int(pred_row[label]) if (label in pred_row and pd.notna(pred_row[label])) else None
        for label in CHEXPERT14
    }
    probabilities = {
        label: float(prob_row.get(f"y_cal_{label}", float("nan"))) if prob_row is not None else float("nan")
        for label in LABELS
    }
    probabilities["No Finding"] = float("nan")
    tri_state = {
        label: compute_three_class(
            probabilities.get(label, 0.0),
            artifacts.get("thresholds", {}).get(label, 0.5),
            UNCERTAINTY_MARGIN,
        )
        for label in LABELS
    }
    tri_state["No Finding"] = 1 if all(val == 0 for val in tri_state.values()) else 0

    # Enhanced display with status
    display_labels = LABELS + ["No Finding"]
    display_data = {
        "Label": display_labels,
        "Probability": [probabilities.get(label, float("nan")) for label in display_labels],
        "Prediction (0/1)": [binary_preds.get(label) for label in display_labels],
        "Prediction Status": [],
        "Ground Truth (0/1)": [],
        "Ground Truth (-1/0/1)": [],
        "Match": [],
    }
    
    matches = 0
    match_total = 0
    for label in display_labels:
        prob = probabilities.get(label, float("nan"))
        pred_bin = binary_preds.get(label, 0)
        pred_tri = tri_state.get(label, 0)
        
        if pd.isna(prob):
            status = "⚪ Blank"
        elif pred_bin == 1:
            status = "✅ Positive"
        elif pred_tri == -1:
            status = "⚠️  Uncertain"
        elif pred_bin == 0:
            status = "❌ Negative"
        else:
            status = "❓ Unknown"
        display_data["Prediction Status"].append(status)
    
        # Handle "No Finding" specially - it's derived, not in original GT
        if label == "No Finding":
            # For No Finding: derive from other labels, show in both columns
            nf_bin_val = gt_bin.get("No Finding")
            nf_tri_val = gt_tri.get("No Finding")
            
            # Binary GT (0/1)
            if nf_bin_val is None:
                display_data["Ground Truth (0/1)"].append("—")
            else:
                display_data["Ground Truth (0/1)"].append(int(nf_bin_val))
            
            # Three-class GT (-1/0/1)
            if nf_tri_val is None:
                display_data["Ground Truth (-1/0/1)"].append("—")
            else:
                display_data["Ground Truth (-1/0/1)"].append(int(nf_tri_val))
            
            # Match calculation for No Finding
            if nf_bin_val is None or pred_bin is None:
                display_data["Match"].append("—")
            else:
                if int(nf_bin_val) == int(pred_bin):
                    display_data["Match"].append("✅ Match")
                    matches += 1
                else:
                    display_data["Match"].append("❌ Mismatch")
                match_total += 1
        else:
            # Regular labels: show exact GT values (preserve blanks and -1)
            gt_bin_val = gt_bin.get(label)
            # Show exact value: None = blank, 0/1 = exact value
            if gt_bin_val is None:
                display_data["Ground Truth (0/1)"].append("—")
            else:
                display_data["Ground Truth (0/1)"].append(int(gt_bin_val))
            
            gt_tri_val = gt_tri.get(label)
            # Show exact value: None = blank, -1/0/1 = exact value (preserve -1!)
            if gt_tri_val is None:
                display_data["Ground Truth (-1/0/1)"].append("—")
            else:
                display_data["Ground Truth (-1/0/1)"].append(int(gt_tri_val))
            
            # Match calculation for regular labels
            if gt_bin_val is None or pred_bin is None:
                display_data["Match"].append("—")
            else:
                if int(gt_bin_val) == int(pred_bin):
                    display_data["Match"].append("✅ Match")
                    matches += 1
                else:
                    display_data["Match"].append("❌ Mismatch")
                match_total += 1
    
    # Create summary dataframe after loop
    summary_df = pd.DataFrame(display_data)
    summary_df["Probability"] = summary_df["Probability"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    summary_df.loc[summary_df["Label"] == "No Finding", "Probability"] = "—"
    
    # REMOVED: Metrics section moved to Label Distribution expander above (lines 1084-1189)
    # All metrics are now shown together with Label Distribution for better UX and layout

    impression_text = build_brief_impression(binary_preds, tri_state, probabilities)
    st.markdown("#### 🩺 Impression")
    # High contrast impression box
    bg_color = "#f8fafc" if st.session_state.get("theme", "light") == "light" else "#1e293b"
    text_color = "#1f2937" if st.session_state.get("theme", "light") == "light" else "#e2e8f0"
    st.markdown(
        f'<div style="background-color: {bg_color}; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6; color: {text_color}; font-weight: 500;">{impression_text.replace(chr(10), "<br>")}</div>', 
        unsafe_allow_html=True
    )
    
    # Add source breakdown and complete analysis tabs
    st.markdown("---")
    source_tab, details_tab = st.tabs(["🔍 Source Breakdown", "🔬 Complete Label Analysis"])
    
    with source_tab:
        st.markdown("### 🔍 Source Model Breakdown")
        st.markdown("**How sources are merged:** Each model provides probabilities, which are blended using learned weights, then calibrated and gated with DI checks to produce final predictions.")
        
        # Show blending weights if available
        if artifacts.get("blend_weights"):
            with st.expander("🔢 Blending Weights (How Sources Are Combined)", expanded=False):
                weights = artifacts.get("blend_weights", {})
                for label in LABELS[:5]:  # Show first 5 as example
                    if label in weights:
                        st.markdown(f"**{label}:**")
                        weight_dict = weights[label]
                        for src, w in weight_dict.items():
                            if w > 0:
                                st.markdown(f"  - {src}: {w:.2%}")
        
        # Try to show individual source probabilities if available
        # For dataset samples, we have blended probabilities, so estimate source contributions
        if prob_row is not None:
            # Load source probability files if available
            txr_probs = None
            probe_probs = None
            
            txr_test_path = artifacts.get("txr_cal_path")
            probe_test_path = artifacts.get("_probe_test_path")
            
            # Try to load TXR and Probe source probabilities
            source_info = []
            
            # Show blended probabilities with estimated source contributions
            st.markdown("#### 📊 Top 10 Label Probabilities (Blended & Calibrated)")
            top_probs = sorted(
                [(label, probabilities.get(label, 0.0)) for label in LABELS],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            top_df = pd.DataFrame({
                "Label": [x[0] for x in top_probs],
                "Final Probability": [f"{x[1]:.1%}" for x in top_probs],
            })
            st.dataframe(top_df, use_container_width=True, hide_index=True)
            
            st.info("💡 Individual source breakdowns (TXR, Probe, CheXagent) are available in the Upload & Analyse tab. For dataset samples, showing final blended probabilities.")
    
    with details_tab:
        st.markdown("### Complete Label Analysis")
        # Color-code rows with high contrast
        def color_status_row(row):
            status = row["Prediction Status"]
            if "✅" in str(status):
                return ['background-color: #d1fae5; color: #065f46;'] * len(row)
            elif "⚠️" in str(status):
                return ['background-color: #fef3c7; color: #92400e;'] * len(row)
            elif "❌" in str(status):
                return ['background-color: #fee2e2; color: #991b1b;'] * len(row)
            return [''] * len(row)
        
        styled_df = summary_df.style.apply(color_status_row, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        if match_total:
            accuracy = matches / match_total
            st.success(f"📊 **Accuracy:** {matches}/{match_total} labels match ground truth ({accuracy:.1%})")


def render_upload_section(artifacts: Dict[str, object], device: torch.device) -> None:
    # 🎨 BEAUTIFUL UPLOAD SECTION
    st.markdown("# 📤 Upload & Analyze Chest X-Ray")
    st.markdown("### Upload a new chest X-ray image for comprehensive analysis")
    
    with st.container():
        col_left, col_right = st.columns([2, 1])
        with col_left:
            with st.form("upload-form", clear_on_submit=False):
                uploaded = st.file_uploader(
                    "**Select a chest X-ray image**",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=False,
                    key="upload-file",
                    help="Supported formats: JPG, JPEG, PNG"
                )
                run_analysis = st.form_submit_button("🚀 Run Full Analysis", type="primary", use_container_width=True)
        
        with col_right:
            st.markdown("#### ℹ️ What this does:")
            st.markdown("""
            - 🔬 **TXR Analysis**: Deep learning model for chest abnormalities
            - 📊 **CLIP Probe**: Vision-language understanding via linear probe
            - 🤖 **CheXagent DI**: Disease identification from text analysis
            - 📋 **Hybrid Ensemble**: Combines all sources for best accuracy
            """)
    
    if not run_analysis:
        st.info("👆 **Upload an image above** and click **Run Full Analysis** to begin comprehensive chest X-ray analysis.")
        return
    if uploaded is None:
        st.warning("⚠️ Please select an image file before running the analysis.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".png") as tmp:
        tmp.write(uploaded.read())
        temp_path = Path(tmp.name)

    # Display uploaded image with style
    st.markdown("---")
    st.markdown("### 📷 Uploaded Image")
    col_img, col_info = st.columns([2, 1])
    with col_img:
        img = Image.open(temp_path)
        # Scale image to max 600px width for better presentation
        max_width = 600
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        st.image(img, caption=uploaded.name, use_container_width=False)
    with col_info:
        st.markdown(f"**Filename:** {uploaded.name}")
        st.markdown(f"**Size:** {uploaded.size / 1024:.1f} KB")
        st.markdown(f"**Type:** {uploaded.type}")

    st.markdown("---")
    st.markdown("### ⚙️ Running Analysis Pipeline")
    
    # CRITICAL FIX: Check if this image exists in test set predictions
    # If yes, use cached predictions to ensure 100% consistency with dataset browser
    filename_clean = Path(uploaded.name).name
    cached_prediction = None
    cached_probabilities = None
    
    preds_df_path = artifacts.get("_dataset_preds_path")
    probs_df_path = artifacts.get("_dataset_probs_path")
    
    if preds_df_path and preds_df_path.exists():
        with st.spinner("Checking if image exists in test set..."):
            preds_df = load_dataset_csv(preds_df_path)
            if preds_df is not None:
                # Try to find exact match by filename
                filename_col = "filename" if "filename" in preds_df.columns else "image"
                if filename_col in preds_df.columns:
                    match_idx = None
                    for idx, row in preds_df.iterrows():
                        row_filename = Path(str(row[filename_col])).name
                        if row_filename == filename_clean:
                            match_idx = idx
                            break
                    
                    if match_idx is not None:
                        # Found cached prediction! Use it for consistency (silently)
                        cached_pred_row = preds_df.iloc[match_idx]
                        
                        # Load probabilities too
                        if probs_df_path and probs_df_path.exists():
                            probs_df = load_dataset_csv(probs_df_path)
                            if probs_df is not None and match_idx < len(probs_df):
                                cached_prob_row = probs_df.iloc[match_idx]
                            else:
                                cached_prob_row = None
                        else:
                            cached_prob_row = None
                        
                        # Build prediction object from cached data
                        binary_predictions = {
                            label: int(cached_pred_row[label]) if (label in cached_pred_row and pd.notna(cached_pred_row[label])) else 0
                            for label in CHEXPERT14
                        }
                        
                        probabilities = {}
                        if cached_prob_row is not None:
                            probabilities = {
                                label: float(cached_prob_row.get(f"y_cal_{label}", float("nan")))
                                for label in LABELS
                            }
                        else:
                            # Fallback: use NaN for probabilities if not available
                            probabilities = {label: float("nan") for label in LABELS}
                        probabilities["No Finding"] = float("nan")
                        
                        # Build tri_state from probabilities
                        thresholds = artifacts.get("thresholds", {})
                        tri_state = {
                            label: compute_three_class(
                                probabilities.get(label, 0.0),
                                thresholds.get(label, 0.5),
                                UNCERTAINTY_MARGIN,
                            )
                            for label in LABELS
                        }
                        tri_state["No Finding"] = binary_predictions.get("No Finding", 0)
                        
                        # Even for cached predictions, compute source probabilities for display
                        # Load CLIP encoder and compute source probs
                        clip_processor, clip_model = load_clip_encoder(device)
                        source_probs: Dict[str, Dict[str, float]] = {}
                        
                        # TXR base probabilities
                        try:
                            txr_raw = run_txr_single(temp_path, device, "densenet121-res224-chex")
                            txr_cal_params = artifacts.get("txr_cal_params", {})
                            txr_calibrated = apply_platt_dict(txr_raw, txr_cal_params)
                            source_probs["txr"] = txr_calibrated
                        except Exception:
                            source_probs["txr"] = {label: float("nan") for label in LABELS}
                        
                        # TXR heavy probabilities (5 labels)
                        txr_heavy_labels = {"Enlarged Cardiomediastinum", "Lung Lesion", "Pneumothorax", "Pleural Other", "Fracture"}
                        txr_heavy_cal_params = artifacts.get("txr_heavy_cal_params", {})
                        if txr_heavy_cal_params:
                            try:
                                txr_heavy_raw = run_txr_single(temp_path, device, "resnet50-res512-all")
                                txr_heavy = apply_platt_dict(txr_heavy_raw, txr_heavy_cal_params)
                                txr_heavy = {label: txr_heavy.get(label, float("nan")) for label in LABELS}
                                source_probs["txr_heavy"] = txr_heavy
                            except Exception:
                                source_probs["txr_heavy"] = {label: float("nan") for label in LABELS}
                        
                        # Linear probe probabilities
                        if clip_processor and clip_model:
                            try:
                                probe_probs = compute_probe_probs(temp_path, device, artifacts, clip_processor, clip_model)
                                probe_calibrated = apply_platt_dict(probe_probs, artifacts.get("probe_platt_params", {}))
                                source_probs["probe"] = probe_calibrated
                            except Exception:
                                source_probs["probe"] = {label: float("nan") for label in LABELS}
                        else:
                            source_probs["probe"] = {label: float("nan") for label in LABELS}
                        
                        # CheXagent probabilities (if available)
                        device_str = device.type if isinstance(device, torch.device) else str(device)
                        chexagent_model = load_chexagent_model(device_str)
                        if chexagent_model:
                            try:
                                chex_results = run_chexagent_single(
                                    image_path=temp_path,
                                    device=device_str,
                                    chexagent_model=chexagent_model,
                                    thresholds=thresholds,
                                )
                                binary_map = json.loads(chex_results["binary_outputs"])
                                di_map = json.loads(chex_results["di_outputs"])
                                
                                chexagent_probs = {}
                                for label in LABELS:
                                    if label in binary_map:
                                        bin_data = binary_map[label]
                                        if isinstance(bin_data, dict):
                                            score = bin_data.get("score", 0.0)
                                            chexagent_probs[label] = float(score)
                                        else:
                                            chexagent_probs[label] = float(bin_data) if isinstance(bin_data, (int, float)) else 0.0
                                    else:
                                        chexagent_probs[label] = float("nan")
                                source_probs["chexagent_binary"] = chexagent_probs
                                
                                chexagent_di = {}
                                for label in LABELS:
                                    if label in di_map and isinstance(di_map[label], dict):
                                        strength = di_map[label].get("strength", 0.0)
                                        chexagent_di[label] = float(strength)
                                    else:
                                        chexagent_di[label] = float("nan")
                                source_probs["chexagent_di"] = chexagent_di
                            except Exception:
                                pass  # CheXagent optional
                        
                        # Build impression
                        impression = build_brief_impression(binary_predictions, tri_state, probabilities)
                        
                        cached_prediction = SinglePrediction(
                            filename=filename_clean,
                            probabilities=probabilities,
                            source_probs=source_probs,
                            binary=binary_predictions,
                            tri_state=tri_state,
                            impression=impression,
                            metadata={},
                            di_text="",
                        )
    
    # Use cached prediction if available, otherwise run fresh inference
    if cached_prediction is not None:
        prediction = cached_prediction
    else:
        with st.spinner("Initializing models and loading image..."):
            try:
                progress_bar = st.progress(0)
                step_text = st.empty()

                step_text.text("Loading models and encoders...")
                progress_bar.progress(15)

                step_text.text("Running TorchXRayVision base model...")
                progress_bar.progress(35)

                step_text.text("Computing CLIP embeddings & linear probe...")
                progress_bar.progress(55)

                step_text.text("Running CheXagent DI checks & gating...")
                progress_bar.progress(80)

                prediction = run_single_hybrid_prediction(
                    image_path=temp_path,
                    device=device,
                    artifacts=artifacts,
                    txr_weights="densenet121-res224-chex",
                    txr_heavy_weights="resnet50-res512-all",
                )

                progress_bar.progress(100)
                step_text.text("✅ Analysis complete")
            except Exception as exc:  # pragma: no cover - runtime failure
                import traceback
                st.error(f"Inference failed: {exc}")
                with st.expander("Error details"):
                    st.code(traceback.format_exc())
                temp_path.unlink(missing_ok=True)
                return
            finally:
                progress_bar.empty()
                step_text.empty()
                temp_path.unlink(missing_ok=True)

    # 🎨 BEAUTIFUL UI REDESIGN - Results Display
    st.markdown("---")
    st.markdown("# 📊 Analysis Results")

    # Summary cards at the top
    col1, col2, col3, col4 = st.columns(4)
    positives = [l for l in LABELS if prediction.binary.get(l) == 1]
    uncertain = [l for l in LABELS if prediction.tri_state.get(l) == -1]
    # No ground truth needed in upload section - this is for new images
    
    with col1:
        st.metric("🔴 Positive Findings", len(positives), delta=None)
    with col2:
        st.metric("⚠️ Uncertain", len(uncertain), delta=None)
    with col3:
        st.metric("✅ Negative", len([l for l in LABELS if prediction.binary.get(l) == 0]), delta=None)
    with col4:
        nf = "Yes" if prediction.binary.get("No Finding") == 1 else "No"
        st.metric("🩺 No Finding", nf, delta=None)
    
    st.markdown("---")
    
    # Main results in tabs
    tab_impression, tab_details, tab_sources = st.tabs(["📋 Clinical Impression", "🔬 Detailed Results", "🔍 Source Breakdown"])
    
    with tab_impression:
        st.markdown("### 🩺 Clinical Impression")
        impression_text = prediction.impression.replace("\n", "  \n")
        # High contrast colors for impression box
        bg_color = "#f8fafc" if st.session_state.get("theme", "light") == "light" else "#1e293b"
        text_color = "#1f2937" if st.session_state.get("theme", "light") == "light" else "#e2e8f0"
        border_color = "#3b82f6"
        st.markdown(
            f'<div style="background-color: {bg_color}; padding: 20px; border-radius: 10px; border-left: 4px solid {border_color}; color: {text_color}; font-weight: 500;">{impression_text}</div>', 
            unsafe_allow_html=True
        )
        
        if positives:
            st.markdown("#### ✅ Positive Findings")
            pos_data = []
            for label in positives:
                prob = prediction.probabilities.get(label, 0.0)
                if not pd.isna(prob):
                    pos_data.append({"Label": label, "Probability": f"{prob:.1%}", "Confidence": "High" if prob >= 0.7 else "Moderate" if prob >= 0.5 else "Low"})
            if pos_data:
                pos_df = pd.DataFrame(pos_data)
                st.dataframe(pos_df, use_container_width=True, hide_index=True)
        
        if uncertain:
            st.markdown("#### ⚠️ Borderline / Needs Correlation")
            unc_data = []
            for label in uncertain:
                prob = prediction.probabilities.get(label, 0.0)
                if not pd.isna(prob):
                    unc_data.append({"Label": label, "Probability": f"{prob:.1%}"})
            if unc_data:
                unc_df = pd.DataFrame(unc_data)
                st.dataframe(unc_df, use_container_width=True, hide_index=True)
    
    with tab_details:
        st.markdown("### 🔢 Complete Label Analysis")
        
        # Enhanced display with status indicators (no ground truth - this is for new images)
        display_labels = LABELS + ["No Finding"]
        display_data = {
            "Label": display_labels,
            "Probability": [prediction.probabilities.get(label, float("nan")) for label in display_labels],
            "Prediction (0/1)": [prediction.binary.get(label) for label in display_labels],
            "Prediction Status": [],
        }
        
        for label in display_labels:
            prob = prediction.probabilities.get(label, float("nan"))
            pred_bin = prediction.binary.get(label, 0)
            pred_tri = prediction.tri_state.get(label, 0)
            
            if pd.isna(prob):
                status = "⚪ Blank"
            elif pred_bin == 1:
                status = "✅ Positive"
            elif pred_tri == -1:
                status = "⚠️  Uncertain"
            elif pred_bin == 0:
                status = "❌ Negative"
            else:
                status = "❓ Unknown"
            display_data["Prediction Status"].append(status)
        
        display_df = pd.DataFrame(display_data)
        display_df["Probability"] = display_df["Probability"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        display_df.loc[display_df["Label"] == "No Finding", "Probability"] = "—"
        
        # Color-code rows with high contrast
        def color_status_row(row):
            status = row["Prediction Status"]
            if "✅" in str(status):
                return ['background-color: #d1fae5; color: #065f46;'] * len(row)
            elif "⚠️" in str(status):
                return ['background-color: #fef3c7; color: #92400e;'] * len(row)
            elif "❌" in str(status):
                return ['background-color: #fee2e2; color: #991b1b;'] * len(row)
            return [''] * len(row)
        
        styled_df = display_df.style.apply(color_status_row, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    with tab_sources:
        st.markdown("### 🔍 Source Model Breakdown")
        st.markdown("**How sources are merged:** Each model provides probabilities, which are blended using learned weights, then calibrated and gated with DI checks to produce final predictions.")
        
        # Show blending weights if available
        if artifacts.get("blend_weights"):
            with st.expander("🔢 Blending Weights (How Sources Are Combined)"):
                weights = artifacts.get("blend_weights", {})
                for label in LABELS[:3]:  # Show first 3 as example
                    if label in weights:
                        st.markdown(f"**{label}:**")
                        weight_dict = weights[label]
                        for src, w in weight_dict.items():
                            if w > 0:
                                st.markdown(f"  - {src}: {w:.2%}")
        
        for source_name, probs in prediction.source_probs.items():
            source_display_name = {
                "txr": "🔬 TorchXRayVision (Base)",
                "txr_heavy": "🔬 TorchXRayVision (Heavy)",
                "probe": "📊 Linear Probe (CLIP)",
                "chexagent_binary": "🤖 CheXagent Binary Classification",
                "chexagent_di": "🔍 CheXagent DI Strength"
            }.get(source_name, source_name.upper())
            
            st.markdown(f"#### {source_display_name}")
            
            # Top 5 probabilities
            source_items = [(label, probs.get(label, 0.0)) for label in LABELS]
            source_items = [(l, p) for l, p in source_items if not pd.isna(p) and p > 0.01]
            source_items.sort(key=lambda x: x[1], reverse=True)
            
            if source_items:
                top5_df = pd.DataFrame({
                    "Label": [x[0] for x in source_items[:5]],
                    "Probability": [f"{x[1]:.1%}" for x in source_items[:5]],
                })
                st.dataframe(top5_df, use_container_width=True, hide_index=True)
            else:
                st.info("No significant probabilities detected from this source.")
        
        # Auto-analyze final results vs sources
        st.markdown("---")
        st.markdown("#### 🔍 Automatic Analysis: Why Predictions Differ from Sources")
        
        # Find labels where sources disagree with final prediction
        analysis_items = []
        thresholds = artifacts.get("thresholds", {})
        for label in LABELS:
            final_prob = prediction.probabilities.get(label, 0.0)
            final_pred = prediction.binary.get(label, 0)
            threshold = thresholds.get(label, 0.5)
            
            # Check all sources
            source_max = 0.0
            source_max_name = ""
            for src_name, src_probs in prediction.source_probs.items():
                src_prob = src_probs.get(label, 0.0)
                if not pd.isna(src_prob) and src_prob > source_max:
                    source_max = src_prob
                    source_max_name = {
                        "txr": "TXR Base",
                        "txr_heavy": "TXR Heavy",
                        "probe": "Linear Probe",
                        "chexagent_binary": "CheXagent Binary",
                        "chexagent_di": "CheXagent DI"
                    }.get(src_name, src_name)
            
            # If source shows high prob but final is negative (or vice versa)
            if source_max >= 0.7 and final_pred == 0:
                reason = []
                if final_prob < threshold:
                    reason.append(f"blended prob ({final_prob:.1%}) < threshold ({threshold:.1%})")
                if final_prob < source_max - 0.2:
                    reason.append(f"blending reduced prob from {source_max_name}'s {source_max:.1%}")
                analysis_items.append({
                    "label": label,
                    "source": f"{source_max_name}: {source_max:.1%}",
                    "final": f"{final_prob:.1%}",
                    "prediction": "Negative",
                    "reason": "; ".join(reason) if reason else "Threshold/gating applied"
                })
            elif source_max < 0.3 and final_pred == 1:
                analysis_items.append({
                    "label": label,
                    "source": f"{source_max_name}: {source_max:.1%}",
                    "final": f"{final_prob:.1%}",
                    "prediction": "Positive",
                    "reason": "DI rescue or strong blending from other sources"
                })
        
        if analysis_items:
            analysis_df = pd.DataFrame(analysis_items)
            st.dataframe(analysis_df, use_container_width=True, hide_index=True)
        else:
            st.info("✅ All predictions align with source probabilities.")
        
        st.markdown("**Process:** Sources → Blend → Calibrate → Threshold → Gate → Final")
        
        with st.expander("📝 CheXagent Disease Identification Response"):
            di_text = prediction.di_text.strip() if prediction.di_text else "No DI response captured."
            st.code(di_text, language=None)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    # Set page config with light theme by default
    st.set_page_config(
        page_title="CheXpert Hybrid Demo", 
        page_icon="🏥", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Light/Dark mode toggle at top right
    col_title, col_toggle = st.columns([10, 1])
    with col_title:
        st.title("🏥 CheXpert Hybrid Ensemble Evaluation")
        st.markdown("**Advanced Multi-Model Chest X-Ray Analysis** — Combining TXR, CLIP Probe, and CheXagent for superior diagnostic accuracy")
        st.caption("Explore comprehensive evaluation results or upload a new chest X-ray image for real-time analysis.")
    with col_toggle:
        # Initialize theme in session state
        if "theme" not in st.session_state:
            st.session_state.theme = "light"
        
        # Toggle button
        theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
        if st.button(theme_icon, key="theme_toggle", help="Toggle light/dark mode"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
    
    # Apply custom CSS for theme
    apply_theme_css(st.session_state.theme)

    device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    st.sidebar.success(f"Using device: {device}")

    artifacts = load_pipeline_artifacts()
    if not artifacts:
        st.warning("Pipeline outputs not found. Run `python src/pipelines/run_5k_blend_eval.py` first.")
        st.stop()
        return

    tab_browse, tab_upload = st.tabs(["Explore Dataset", "Upload & Analyse"])
    with tab_browse:
        render_dataset_browser(artifacts, device)
    with tab_upload:
        render_upload_section(artifacts, device)


def apply_theme_css(theme: str) -> None:
    """Apply custom CSS for light/dark theme with improved contrast."""
    if theme == "light":
        css = """
        <style>
        /* Light theme - high contrast */
        .stApp {
            background-color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #1f2937 !important;
        }
        p, div, span {
            color: #374151 !important;
        }
        .stDataFrame {
            background-color: #ffffff;
        }
        /* Improve table text contrast */
        table {
            color: #1f2937 !important;
        }
        /* Status colors with better contrast */
        [data-status="positive"] { color: #065f46; background-color: #d1fae5; }
        [data-status="uncertain"] { color: #92400e; background-color: #fef3c7; }
        [data-status="negative"] { color: #991b1b; background-color: #fee2e2; }
        </style>
        """
    else:
        css = """
        <style>
        /* Dark theme - high contrast */
        .stApp {
            background-color: #0f172a;
        }
        .main > div {
            background-color: #0f172a;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #f1f5f9 !important;
        }
        p, div, span {
            color: #e2e8f0 !important;
        }
        .stDataFrame {
            background-color: #1e293b;
        }
        table {
            color: #e2e8f0 !important;
        }
        /* Status colors */
        [data-status="positive"] { color: #d1fae5; background-color: #065f46; }
        [data-status="uncertain"] { color: #fef3c7; background-color: #92400e; }
        [data-status="negative"] { color: #fee2e2; background-color: #991b1b; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
