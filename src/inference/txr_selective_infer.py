#!/usr/bin/env python3
"""Selective TorchXRayVision inference for a focused set of CheXpert labels.

This utility powers the "heavy" TXR path that rescues low-performing labels
with the 6.8 GB `resnet50-res512-all` weights. It is intentionally idempotent
and writes split-specific CSVs that plug directly into the 5k pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import inspect
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.inference.txr_infer import infer_txr, load_image_paths

try:
    import torchxrayvision as xrv
except ImportError as exc:  # pragma: no cover - dependency issue
    raise SystemExit(
        "torchxrayvision is required for txr_selective_infer.py."
        " Install with `pip install torchxrayvision`."
    ) from exc


DEFAULT_LABELS = [
    "Support Devices",
    "Pleural Other",
    "Fracture",
    "Lung Lesion",
    "Pneumothorax",
]


def resolve_label_subset(labels_arg: Optional[str], labels_list: Optional[List[str]]) -> List[str]:
    if labels_arg:
        labels = [lbl.strip() for lbl in labels_arg.split(",") if lbl.strip()]
        if labels:
            return labels
    if labels_list:
        return labels_list
    return DEFAULT_LABELS


def ensure_txr_model(weights: str):
    """Load TorchXRayVision DenseNet, retrying with force download if supported."""
    try:
        return xrv.models.DenseNet(weights=weights)
    except FileNotFoundError:
        sig = inspect.signature(xrv.models.DenseNet)
        if "force_download" in sig.parameters:
            print(f"⚠️  Weights '{weights}' missing. Forcing re-download…")
            return xrv.models.DenseNet(weights=weights, force_download=True)
        print("⚠️  Weights missing and force_download not supported. Retrying…")
        return xrv.models.DenseNet(weights=weights)


def load_split_manifest(split: Optional[str], output_csv: Path) -> Optional[pd.DataFrame]:
    if not split:
        return None
    split_name = split.lower()
    if split_name not in {"train", "test", "val", "validation"}:
        print(f"⚠️  Unknown split '{split}'; skipping split filtering.")
        return None

    # Locate split CSV near the intended output directory first, then project root
    candidates: List[Path] = []
    out_root = output_csv.parent.parent if output_csv.parent.parent.name else None
    if out_root:
        candidates.append(out_root / "splits" / f"ground_truth_{split_name}.csv")
    candidates.extend([
        project_root / "outputs_full_final" / "splits" / f"ground_truth_{split_name}.csv",
        project_root / "outputs_5k" / "splits" / f"ground_truth_{split_name}.csv",
        project_root / "data" / f"ground_truth_{split_name}.csv",
    ])

    for candidate in candidates:
        if candidate.exists():
            df = pd.read_csv(candidate)
            print(f"📂 Using split manifest: {candidate} ({len(df)} rows)")
            return df

    print(f"⚠️  Split manifest for '{split}' not found. Proceeding without split filtering.")
    return None


def build_metadata(split_df: Optional[pd.DataFrame], images: Iterable[Path]) -> pd.DataFrame:
    if split_df is not None:
        df = split_df.copy()
        if "image" not in df.columns:
            raise ValueError("Split manifest must contain an 'image' column with absolute paths.")
        df["image"] = df["image"].apply(lambda x: str(Path(x)))
        df["filename"] = df.get("filename", df["image"].apply(lambda x: Path(x).name))
        return df

    rows = []
    for path in images:
        rows.append({
            "image": str(path),
            "filename": Path(path).name,
            "study_id": None,
            "subject_id": None,
        })
    return pd.DataFrame(rows)


def run_selective_txr(
    metadata: pd.DataFrame,
    labels: List[str],
    device: torch.device,
    model_weights: str,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Dict[str, float]]:
    paths = [Path(p) for p in metadata["image"].tolist()]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"⚠️  {len(missing)} image(s) missing on disk. They will be skipped.")
    existing_paths = [p for p in paths if p.exists()]
    if not existing_paths:
        print("❌ No images found for selective TXR inference.")
        return {}

    # Ensure weights are present before running full inference
    try:
        txr_results = infer_txr(
            images=existing_paths,
            model_weights=model_weights,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    except FileNotFoundError:
        # Weights not cached yet – force download then retry once
        model = ensure_txr_model(model_weights)
        del model
        txr_results = infer_txr(
            images=existing_paths,
            model_weights=model_weights,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    # Filter down to requested labels + useful metadata from txr_results
    filtered: Dict[str, Dict[str, float]] = {}
    for img_path, record in txr_results.items():
        entry: Dict[str, float] = {}
        for label in labels:
            key = f"prob_{label}"
            if key in record:
                entry[label] = record[key]
            else:
                # Fallback to raw_probs map if present
                raw = record.get("raw_probs")
                if isinstance(raw, str):
                    raw = json.loads(raw)
                if isinstance(raw, dict):
                    # Normalise key spacing/underscores for matching
                    norm_target = label.lower().replace(" ", "")
                    match = None
                    for raw_key, value in raw.items():
                        norm_raw = raw_key.lower().replace(" ", "").replace("_", "")
                        if norm_raw == norm_target:
                            match = value
                            break
                    if match is not None:
                        entry[label] = match
                    else:
                        entry[label] = np.nan
                else:
                    entry[label] = np.nan
        entry["patient_id"] = record.get("patient_id")
        filtered[img_path] = entry
    return filtered


def save_output(metadata: pd.DataFrame, txr_scores: Dict[str, Dict[str, float]], labels: List[str], output_csv: Path) -> None:
    rows = []
    for _, row in metadata.iterrows():
        image_path = str(row["image"])
        entry = {
            "image": image_path,
            "filename": row.get("filename", Path(image_path).name),
            "study_id": row.get("study_id"),
            "subject_id": row.get("subject_id"),
            "patient_id": row.get("patient_id"),
        }
        scores = txr_scores.get(image_path) or txr_scores.get(str(Path(image_path)))
        for label in labels:
            value = float("nan")
            if scores and label in scores:
                value = scores[label]
            entry[f"prob_{label}"] = value
            entry[f"y_pred_{label}"] = value
        rows.append(entry)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved selective TXR probabilities to {output_csv} ({len(df)} rows)")
    for label in labels:
        col = f"prob_{label}"
        if col in df.columns and df[col].notna().any():
            series = df[col].dropna()
            print(f"   {label}: coverage={len(series)}/{len(df)} ({len(series)/max(len(df),1):.1%}), mean={series.mean():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Selective TXR inference for heavy model")
    parser.add_argument("--images", type=Path, required=True, help="Image list file (.txt) or directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV for probabilities")
    parser.add_argument("--split", type=str, default=None, help="Optional split name (train/test/val)")
    parser.add_argument("--labels_subset", type=str, default=None, help="Comma separated label subset")
    parser.add_argument("--labels", nargs="*", default=None, help="(Deprecated) label list override")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--model_weights", type=str, default="resnet50-res512-all")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    output_csv = (project_root / args.output).resolve()
    images_path = (project_root / args.images).resolve()
    if not images_path.exists():
        raise SystemExit(f"Images path not found: {images_path}")

    # Parse labels
    labels = resolve_label_subset(args.labels_subset, args.labels)
    print(f"Target labels: {labels}")

    # Device selection with fallbacks
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Load manifests / metadata
    split_df = load_split_manifest(args.split, output_csv)
    if split_df is None:
        image_list = load_image_paths(images_path)
        metadata = build_metadata(None, image_list)
    else:
        metadata = build_metadata(split_df, [])

    if metadata.empty:
        print("❌ No samples available for selective TXR inference.")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(output_csv, index=False)
        return

    txr_scores = run_selective_txr(
        metadata=metadata,
        labels=labels,
        device=device,
        model_weights=args.model_weights,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    save_output(metadata, txr_scores, labels, output_csv)


if __name__ == "__main__":
    main()
