#!/usr/bin/env python3
"""
Run TorchXRayVision DenseNet inference to obtain continuous CheXpert label probabilities.

This replaces the binary score clustering produced by CheXagent text parsing with
continuous probabilities suitable for calibration and threshold tuning.

Example:
    python txr_infer.py \
        --images data/image_list_1000_absolute.txt \
        --out_csv txr_predictions_1000.csv \
        --device mps
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tv_transforms

try:
    import torchxrayvision as xrv
except ImportError as exc:  # pragma: no cover - dependency issue
    raise SystemExit(
        "torchxrayvision is required for txr_infer.py. "
        "Install with `pip install torchxrayvision`."
    ) from exc


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

CHEXPERT13 = CHEXPERT14[:-1]  # All except "No Finding"


def normalize_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def build_label_mapping(model_pathologies: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    """Map CheXpert label names to TorchXRayVision pathologies."""
    lookups = {normalize_name(p): p for p in model_pathologies}

    # Aliases when naming differs between model and CheXpert vocabulary
    synonyms = {
        "pleural_effusion": ["effusion", "pleural_effusion"],
        "pleural_other": ["pleural_other", "pleural_thickening"],
        "support_devices": ["support_devices", "supportdevice", "devices"],
        "lung_lesion": ["lung_lesion", "lesion"],
        "no_finding": ["no_finding", "none"],
    }

    mapping: Dict[str, str] = {}
    missing: List[str] = []

    for label in CHEXPERT14:
        norm = normalize_name(label)
        candidates = [norm] + synonyms.get(norm, [])
        matched = None
        for cand in candidates:
            if cand in lookups:
                matched = lookups[cand]
                break
        if matched is None:
            missing.append(label)
        else:
            mapping[label] = matched

    return mapping, missing


def extract_patient_id(path_str: str) -> str:
    """Approximate MIMIC-CXR patient identifier from a path."""
    path = Path(path_str)
    parts = path.parts
    for idx, segment in enumerate(parts):
        if segment.startswith("p") and len(segment) > 3 and idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if candidate.startswith("p") and len(candidate) > 3:
                return candidate
    return path.stem


@dataclass
class CXRRecord:
    image_path: Path
    patient_id: str


class TXRDataset(Dataset):
    """Dataset for loading CXR images for TorchXRayVision inference."""

    def __init__(self, image_paths: Sequence[Path]):
        self.records = [
            CXRRecord(image_path=Path(p), patient_id=extract_patient_id(str(p)))
            for p in image_paths
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        try:
            # Open and close file properly to avoid broken pipe
            with Image.open(record.image_path) as img:
                image = img.convert("L").copy()  # Copy to avoid keeping file handle open
            # Process image (outside with block, but inside try)
            image = image.resize((224, 224), Image.BILINEAR)
            arr = np.array(image, dtype=np.float32)
            arr = xrv.datasets.normalize(arr, 255.0)
            arr = arr[None, :, :]  # (1, H, W)

            tensor = torch.from_numpy(arr).float()
            return tensor, str(record.image_path), record.patient_id
        except Exception as e:
            raise RuntimeError(f"Failed to load image {record.image_path}: {e}") from e


def load_image_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in {".txt", ".csv"}:
            paths: List[Path] = []
            for line in input_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = Path(line)
                if not p.exists():
                    print(f"⚠️  Missing image: {line}")
                    continue
                paths.append(p)
            return paths
        # Single image file
        return [input_path]

    if input_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dcm"}
        files = [p for p in input_path.rglob("*") if p.suffix.lower() in exts]
        return sorted(files)

    raise FileNotFoundError(f"Input path not found: {input_path}")


def infer_txr(
    images: Sequence[Path],
    model_weights: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Dict[str, float]]:
    # Validate image paths exist
    valid_images = [Path(img) for img in images if Path(img).exists()]
    if not valid_images:
        raise FileNotFoundError(f"No valid images found in: {images}")
    
    # For single image uploads, force CPU and num_workers=0 to avoid multiprocessing issues
    # This prevents broken pipe errors on Apple Silicon and other platforms
    if len(valid_images) == 1:
        device = torch.device("cpu")
        num_workers = 0
    
    dataset = TXRDataset(valid_images)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )

    model = xrv.models.DenseNet(weights=model_weights)
    model = model.to(device)
    model.eval()

    label_map, missing = build_label_mapping(model.pathologies)
    if missing:
        print(
            "⚠️  Warning: the following CheXpert labels were not found in the "
            f"TorchXRayVision model outputs: {missing}. Their probabilities will be NaN."
        )

    inverse_map = {v: k for k, v in label_map.items()}

    results: Dict[str, Dict[str, float]] = {}
    total = len(dataset)
    processed = 0

    with torch.no_grad():
        for batch, paths, patient_ids in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)

            for i in range(batch.shape[0]):
                path = paths[i]
                patient_id = patient_ids[i]
                logit_vec = logits[i].cpu().numpy()
                prob_vec = probs[i].cpu().numpy()

                record: Dict[str, float] = {
                    "patient_id": patient_id,
                }

                for j, pathology in enumerate(model.pathologies):
                    prob = float(prob_vec[j])
                    logit = float(logit_vec[j])
                    if math.isnan(prob):
                        continue
                    if pathology in inverse_map:
                        label = inverse_map[pathology]
                        record[f"prob_{label}"] = prob
                        record[f"logit_{label}"] = logit
                    record.setdefault("raw_probs", {})[pathology] = prob

                nf_key = "prob_No Finding"
                if math.isnan(record.get(nf_key, float("nan"))):
                    other_probs = [
                        record.get(f"prob_{lbl}", float("nan"))
                        for lbl in CHEXPERT13
                    ]
                    other_probs = [p for p in other_probs if not math.isnan(p)]
                    if other_probs:
                        record[nf_key] = max(0.0, 1.0 - max(other_probs))
                        record["logit_No Finding"] = float("nan")

                record["raw_probs"] = json.dumps(record["raw_probs"])
                results[path] = record

            processed += batch.shape[0]
            print(f"Processed {processed}/{total} images", end="\r", file=sys.stderr)

    # Fill in NaNs for missing labels
    for label in CHEXPERT14:
        key_prob = f"prob_{label}"
        key_logit = f"logit_{label}"
        for rec in results.values():
            rec.setdefault(key_prob, float("nan"))
            rec.setdefault(key_logit, float("nan"))

    return results


def main():
    parser = argparse.ArgumentParser(description="TorchXRayVision inference for CheXpert labels")
    parser.add_argument("--images", type=str, required=True, help="Image list (.txt/.csv) or directory")
    parser.add_argument("--out_csv", type=str, default="txr_predictions.csv", help="Output CSV path")
    parser.add_argument("--model_weights", type=str, default="densenet121-res224-chex", help="TorchXRayVision weights identifier")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", help="cpu | cuda | mps | auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        if device.type == "mps" and not torch.backends.mps.is_available():
            print("⚠️  MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")

    image_paths = load_image_paths(Path(args.images))
    if not image_paths:
        raise SystemExit(f"No images found at {args.images}")

    print(f"🩻 Running TorchXRayVision inference on {len(image_paths)} images")
    print(f"   Device: {device}")
    print(f"   Model weights: {args.model_weights}")

    results = infer_txr(
        images=image_paths,
        model_weights=args.model_weights,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    rows = []
    for path_str, record in results.items():
        row = {"image": path_str}
        row.update({k: v for k, v in record.items() if k != "raw_probs"})
        row["raw_probs"] = record["raw_probs"]
        rows.append(row)

    import pandas as pd  # Local import to avoid making pandas hard dependency if unused elsewhere

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\n✅ Saved {len(df)} predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
