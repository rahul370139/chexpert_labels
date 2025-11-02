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
from src.inference.smart_ensemble import smart_ensemble_prediction  # noqa: E402


CHEXPERT14 = LABELS + ["No Finding"]
UNCERTAINTY_MARGIN = 0.15


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
        PROJECT_ROOT / "outputs_full",
        PROJECT_ROOT / "outputs_5k",
        PROJECT_ROOT / "server_synced" / "outputs_full",
    ]
    out_root = next((root for root in out_roots if root.exists()), None)
    if out_root is None:
        st.warning("Pipeline outputs not found. Run `python src/pipelines/run_5k_blend_eval.py` first.")
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
            rel("final/test_probs_refit_full.csv"),
            rel("final/test_probs.csv"),
        ]
    )
    dataset_preds_path = resolve_existing(
        [
            rel("final/test_preds_refit_full.csv"),
            rel("final/test_preds.csv"),
        ]
    )
    dataset_metrics_path = resolve_existing(
        [
            rel("final/test_metrics_refit_full.csv"),
            rel("final/test_metrics.csv"),
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
            rel("splits/ground_truth_test_30.csv"),
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

    # Dataset frames (lazy loaded)
    if dataset_probs_path:
        artifacts["dataset_probs"] = pd.read_csv(dataset_probs_path)
    if dataset_preds_path:
        artifacts["dataset_preds"] = pd.read_csv(dataset_preds_path)
    if dataset_metrics_path:
        artifacts["dataset_metrics"] = pd.read_csv(dataset_metrics_path)
    if tri_eval_csv:
        artifacts["dataset_three_class"] = pd.read_csv(tri_eval_csv)
    if tri_summary_csv:
        try:
            artifacts["dataset_three_class_summary"] = pd.read_csv(tri_summary_csv)
        except Exception:
            pass
    if gt_bin_path:
        artifacts["ground_truth_bin"] = pd.read_csv(gt_bin_path)
    if gt_tri_path and gt_tri_path.exists():
        artifacts["ground_truth_tri"] = pd.read_csv(gt_tri_path)
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
def load_clip_encoder(device: torch.device, vision_model: str = "openai/clip-vit-large-patch14") -> Tuple[CLIPImageProcessor, CLIPVisionModel]:
    processor = CLIPImageProcessor.from_pretrained(vision_model)
    model = CLIPVisionModel.from_pretrained(vision_model).to(device)
    model.eval()
    return processor, model


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


# --------------------------------------------------------------------------------------
# TXR inference (base + heavy)
# --------------------------------------------------------------------------------------


def run_txr_single(image_path: Path, device: torch.device, model_weights: str, batch_size: int = 1) -> Dict[str, float]:
    try:
        results = infer_txr(
            images=[image_path],
            model_weights=model_weights,
            device=device,
            batch_size=batch_size,
            num_workers=0,
        )
    except Exception as exc:  # pragma: no cover - runtime failure
        raise RuntimeError(f"TXR inference failed: {exc}")
    record = next(iter(results.values()))
    probs = {label: float(record.get(f"prob_{label}", float("nan"))) for label in LABELS}
    return probs


# --------------------------------------------------------------------------------------
# Linear probe inference
# --------------------------------------------------------------------------------------


def compute_probe_probs(
    image_path: Path,
    device: torch.device,
    artifacts: Dict[str, object],
    clip_processor: CLIPImageProcessor,
    clip_model: CLIPVisionModel,
) -> Dict[str, float]:
    models_pack = artifacts.get("probe_models")
    if not models_pack:
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
    positives = [label for label, val in binary.items() if val == 1 and label != "No Finding"]
    uncertain = [label for label, val in tri_state.items() if val == -1]

    lines: List[str] = []
    if positives:
        positives_sorted = sorted(positives, key=lambda l: -probabilities.get(l, 0.0))
        formatted = ", ".join(f"{lab} ({probabilities.get(lab, 0.0):.2f})" for lab in positives_sorted)
        lines.append(f"Positive findings: {formatted}.")
    if uncertain:
        formatted_uncertain = ", ".join(f"{lab} ({probabilities.get(lab, 0.0):.2f})" for lab in uncertain)
        lines.append(f"Borderline/uncertain: {formatted_uncertain}.")
    if not lines:
        lines.append("No convincing abnormality detected (all findings below thresholds).")
    return " ".join(lines)


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

    # Source probabilities ---------------------------------------------------
    source_probs: Dict[str, Dict[str, float]] = {}

    # TXR base
    try:
        txr_raw = run_txr_single(image_path, device, txr_weights)
    except RuntimeError as exc:  # pragma: no cover - runtime
        st.warning(str(exc))
        txr_raw = {label: float("nan") for label in LABELS}
    txr_cal_params = artifacts.get("txr_cal_params", {})
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

    probabilities = {label: float(blended.loc[0, f"y_cal_{label}"]) for label in labels}

    # DI metadata via CheXagent (for gating + impression text)
    thresholds = artifacts.get("thresholds", {})
    chex_results = smart_ensemble_prediction(
        image_paths=[image_path],
        device=device.type,
        thresholds=thresholds,
        calibration_params=None,
        use_precision_gating=False,
    )[0]
    di_map = json.loads(chex_results["di_outputs"])
    binary_map = json.loads(chex_results["binary_outputs"])
    metadata = {filename: {"binary": binary_map, "di": di_map}}

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

    # Three-class decisions
    tri_state = {
        label: compute_three_class(probabilities.get(label, 0.0), thresholds.get(label, 0.5), UNCERTAINTY_MARGIN)
        for label in LABELS
    }
    tri_state["No Finding"] = 1 if all(val == 0 for val in tri_state.values()) else 0

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
    # fallback: search in radiology_report/files
    base = PROJECT_ROOT.parent / "radiology_report" / "files"
    candidate = base / "p10" / filename
    if candidate.exists():
        return candidate
    return None


def get_ground_truth_for(filename: str, artifacts: Dict[str, object]) -> Tuple[Dict[str, int], Dict[str, int]]:
    gt_bin_df = artifacts.get("ground_truth_bin")
    gt_tri_df = artifacts.get("ground_truth_tri")
    gt_bin: Dict[str, int] = {}
    gt_tri: Dict[str, int] = {}
    if gt_bin_df is not None:
        row = gt_bin_df[gt_bin_df["filename"].apply(lambda x: Path(str(x)).name) == filename]
        if not row.empty:
            for label in LABELS:
                col = f"y_true_{label}" if f"y_true_{label}" in row.columns else label
                if col in row.columns:
                    gt_bin[label] = int(row.iloc[0][col])
    if gt_tri_df is not None:
        row = gt_tri_df[gt_tri_df["filename"].apply(lambda x: Path(str(x)).name) == filename]
        if not row.empty:
            for label in LABELS:
                if label in row.columns:
                    gt_tri[label] = int(row.iloc[0][label])
    return gt_bin, gt_tri


# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------


def render_metrics_section(artifacts: Dict[str, object]) -> None:
    st.markdown("### 📊 Pipeline Performance Summary")

    macro_json = artifacts.get("dataset_metrics_macro") or {}
    macro_all = macro_json.get("macro_all", {})
    macro_nz = macro_json.get("macro_nonzero", {})
    worst = macro_json.get("worst_labels", [])

    cols = st.columns(3)
    cols[0].metric("Micro F1", f"{macro_all.get('micro_f1', 0):.3f}" if macro_all else "n/a")
    cols[1].metric("Macro F1 (all)", f"{macro_all.get('macro_f1', 0):.3f}" if macro_all else "n/a")
    cols[2].metric("Macro F1 (nonzero)", f"{macro_nz.get('macro_f1', 0):.3f}" if macro_nz else "n/a")

    if worst:
        st.markdown("**Lowest F1 Labels:**")
        worst_df = pd.DataFrame(worst)
        st.dataframe(worst_df, use_container_width=True, hide_index=True)

    tri_summary = artifacts.get("dataset_three_class_summary")
    if tri_summary is not None:
        st.markdown("**Three-Class (Certain Only) Macro:**")
        st.table(tri_summary.set_index("metric"))


def render_dataset_browser(artifacts: Dict[str, object], device: torch.device) -> None:
    st.markdown("### 🔍 Explore Evaluation Samples")

    preds_df: Optional[pd.DataFrame] = artifacts.get("dataset_preds")
    probs_df: Optional[pd.DataFrame] = artifacts.get("dataset_probs")

    if preds_df is None or probs_df is None:
        st.info("Run the full pipeline to enable dataset browsing.")
        return

    filenames = preds_df["filename"].tolist() if "filename" in preds_df.columns else preds_df["image"].apply(lambda x: Path(str(x)).name).tolist()
    sample_idx = st.selectbox("Select sample", options=range(len(filenames)), format_func=lambda idx: filenames[idx])
    filename = filenames[sample_idx]

    image_path = map_filename_to_path(filename, artifacts)
    if image_path and image_path.exists():
        st.image(Image.open(image_path), caption=filename, use_column_width=True)
    else:
        st.warning("Image file not found locally.")

    pred_row = preds_df.iloc[sample_idx]
    prob_row = probs_df.iloc[sample_idx] if sample_idx < len(probs_df) else None
    gt_bin, gt_tri = get_ground_truth_for(filename, artifacts)

    binary_preds = {label: int(pred_row[label]) if label in pred_row else None for label in CHEXPERT14}
    probabilities = {label: float(prob_row.get(f"y_cal_{label}", float("nan"))) if prob_row is not None else float("nan") for label in LABELS}

    tri_state = {label: compute_three_class(probabilities.get(label, 0.0), artifacts.get("thresholds", {}).get(label, 0.5), UNCERTAINTY_MARGIN) for label in LABELS}

    summary_df = pd.DataFrame({
        "Label": LABELS,
        "Probability": [probabilities.get(label, float("nan")) for label in LABELS],
        "Binary Prediction": [binary_preds.get(label) for label in LABELS],
        "GT (0/1)": [gt_bin.get(label) if gt_bin else None for label in LABELS],
        "Three-Class": [tri_state.get(label) for label in LABELS],
        "GT (-1/0/1)": [gt_tri.get(label) if gt_tri else None for label in LABELS],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def render_upload_section(artifacts: Dict[str, object], device: torch.device) -> None:
    st.markdown("### 📁 Upload a Chest X-ray")
    uploaded = st.file_uploader("Upload JPG/PNG chest X-ray", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Upload an image to generate hybrid predictions.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.read())
        temp_path = Path(tmp.name)

    st.image(Image.open(temp_path), caption="Uploaded Image", use_column_width=True)

    # Run hybrid inference
    try:
        prediction = run_single_hybrid_prediction(
            image_path=temp_path,
            device=device,
            artifacts=artifacts,
            txr_weights="densenet121-res224-chex",
            txr_heavy_weights="resnet50-res512-all",
        )
    except Exception as exc:  # pragma: no cover - runtime failure
        st.error(f"Inference failed: {exc}")
        return
    finally:
        temp_path.unlink(missing_ok=True)

    st.markdown("#### 🩺 Impression")
    st.success(prediction.impression)

    st.markdown("#### 🔢 Probabilities & Decisions")
    display_df = pd.DataFrame({
        "Label": LABELS,
        "Blended Probability": [prediction.probabilities.get(label, float("nan")) for label in LABELS],
        "Binary Prediction": [prediction.binary.get(label) for label in LABELS],
        "Three-Class": [prediction.tri_state.get(label) for label in LABELS],
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("#### 📚 Source Probabilities")
    for source_name, probs in prediction.source_probs.items():
        st.markdown(f"**{source_name}**")
        source_df = pd.DataFrame({
            "Label": LABELS,
            "Probability": [probs.get(label, float("nan")) for label in LABELS],
        })
        st.dataframe(source_df, use_container_width=True, hide_index=True)

    with st.expander("CheXagent disease-identification response"):
        st.text(prediction.di_text.strip() or "No DI response captured.")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="CheXpert Hybrid Demo", page_icon="🏥", layout="wide")
    st.title("🏥 CheXpert Hybrid Ensemble Demo")
    st.caption("Continuous TXR + CheXagent linear probes + DI gating (binary and three-class outputs)")

    device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    st.sidebar.success(f"Using device: {device}")

    artifacts = load_pipeline_artifacts()
    if not artifacts:
        st.stop()

    render_metrics_section(artifacts)

    col1, col2 = st.columns(2)
    with col1:
        render_dataset_browser(artifacts, device)
    with col2:
        render_upload_section(artifacts, device)


if __name__ == "__main__":
    main()

