#!/usr/bin/env python3
"""
Fine-tune a TorchXRayVision model on local MIMIC-CXR JPGs using CheXpert labels.

- Reads the standard CheXpert CSV (e.g., files/mimic-cxr-2.0.0-chexpert.csv)
  with values in {-1, 0, 1, blank}. Blanks become NaN and can be ignored.
- Supports uncertainty policy: u_zero (map -1->0), u_one (map -1->1), ignore (mask).
- Patient-wise 80/20 split by default; saves probabilities CSV on the val split.

Example:
  ./venv/bin/python src/models/finetune_torchxray.py \
    --csv files/mimic-cxr-2.0.0-chexpert.csv \
    --root files \
    --labels chexpert13 \
    --uncertainty_policy ignore \
    --model densenet121-res224-chex \
    --epochs 5 --batch_size 32 --device mps \
    --out_dir outputs_full/ft_model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

try:
    import torchxrayvision as xrv
except ImportError as exc:
    raise SystemExit("torchxrayvision is required. pip install torchxrayvision") from exc


CHEXPERT14 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"
]
CHEXPERT13 = CHEXPERT14[:-1]


def get_labels(name: str) -> List[str]:
    key = name.strip().lower()
    if key in {"chexpert13", "13"}:
        return CHEXPERT13
    if key in {"chexpert14", "14"}:
        return CHEXPERT14
    raise ValueError("labels must be chexpert13 or chexpert14")


def extract_patient_id(path: str) -> str:
    p = Path(path)
    parts = p.parts
    for i, seg in enumerate(parts):
        if seg.startswith("p") and len(seg) > 3 and i + 1 < len(parts):
            cand = parts[i + 1]
            if cand.startswith("p") and len(cand) > 3:
                return cand
    return p.stem


class CheXpertDataset(Dataset):
    def __init__(self, df: pd.DataFrame, labels: List[str]):
        self.df = df.reset_index(drop=True)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row["abs_path"])  # absolute path
        img = Image.open(img_path).convert("L").resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        arr = xrv.datasets.normalize(arr, 255.0)
        arr = arr[None, :, :]  # (1,H,W)
        x = torch.from_numpy(arr).float()
        y = torch.from_numpy(row[self.labels].to_numpy(np.float32))
        m = torch.from_numpy(row[[f"mask_{l}" for l in self.labels]].to_numpy(np.float32))
        return x, y, m, row["filename"]


def build_model(weights: str, n_outputs: int) -> nn.Module:
    model = xrv.models.DenseNet(weights=weights)
    # TorchXRayVision DenseNet has .classifier at the end with out_features=len(pathologies).
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, n_outputs)
    return model


def bce_masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss = loss_fn(logits, targets)
    loss = loss * mask  # zero out ignored labels
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, labels: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    model.eval()
    all_probs, all_true, all_names = [], [], []
    with torch.no_grad():
        for x, y, m, names in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_true.append(y.numpy())
            all_names.extend(list(names))
    probs = np.concatenate(all_probs, axis=0)
    ytrue = np.concatenate(all_true, axis=0)
    # Simple threshold 0.5 for quick validation
    ypred = (probs >= 0.5).astype(int)
    metrics = {}
    p, r, f, _ = precision_recall_fscore_support(ytrue, ypred, average="macro", zero_division=0)
    metrics.update({"macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f)})
    p, r, f, _ = precision_recall_fscore_support(ytrue, ypred, average="micro", zero_division=0)
    metrics.update({"micro_precision": float(p), "micro_recall": float(r), "micro_f1": float(f)})
    out = pd.DataFrame({"filename": all_names})
    for i, lab in enumerate(labels):
        out[f"y_pred_{lab}"] = probs[:, i]
    return out, metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune TorchXRayVision on local JPGs")
    ap.add_argument("--csv", type=Path, required=True, help="CheXpert labels CSV (e.g., files/mimic-cxr-2.0.0-chexpert.csv)")
    ap.add_argument("--root", type=Path, required=True, help="Root dir that contains the jpgs under the CSV's relative Path (e.g., files)")
    ap.add_argument("--labels", default="chexpert13")
    ap.add_argument("--uncertainty_policy", choices=["ignore", "u_zero", "u_one"], default="ignore")
    ap.add_argument("--model", default="densenet121-res224-chex")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out_dir", type=Path, default=Path("outputs_full/ft_model"))
    args = ap.parse_args()

    labels = get_labels(args.labels)
    df = pd.read_csv(args.csv)
    # The standard CSV uses 'Path' relative to the dataset root; ensure existence.
    path_col = None
    for cand in ["Path", "path", "image"]:
        if cand in df.columns:
            path_col = cand
            break
    if path_col is None:
        raise SystemExit("CSV must have a 'Path' or 'image' column")
    df["rel_path"] = df[path_col].astype(str)
    df["abs_path"] = df["rel_path"].apply(lambda p: str((args.root / p).resolve()))
    df["filename"] = df["rel_path"].apply(lambda p: Path(p).name)
    df = df[df["abs_path"].apply(lambda p: Path(p).exists())].copy()
    if df.empty:
        raise SystemExit("No images found on disk; check --root")

    # Normalize labels and build masks
    for lab in labels:
        if lab not in df.columns:
            df[lab] = np.nan
        vals = df[lab]
        # Map blanks to NaN
        vals = pd.to_numeric(vals, errors="coerce")
        if args.uncertainty_policy == "u_zero":
            vals = vals.fillna(0.0).replace({-1.0: 0.0})
            mask = np.ones(len(vals), dtype=np.float32)
        elif args.uncertainty_policy == "u_one":
            vals = vals.fillna(0.0).replace({-1.0: 1.0})
            mask = np.ones(len(vals), dtype=np.float32)
        else:  # ignore
            mask = (~vals.isna() & (vals != -1.0)).astype(np.float32).values
            vals = vals.fillna(0.0).replace({-1.0: 0.0})
        df[lab] = vals.astype(np.float32)
        df[f"mask_{lab}"] = mask

    # Patient-wise split 80/20
    df["patient_id"] = df["abs_path"].apply(extract_patient_id)
    patients = df["patient_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(patients)
    split = int(len(patients) * 0.8)
    train_patients = set(patients[:split])
    train_df = df[df["patient_id"].isin(train_patients)].copy()
    val_df = df[~df["patient_id"].isin(train_patients)].copy()

    # Compute pos_weight for BCE per label
    pos_weight = []
    for lab in labels:
        y = train_df[lab].values
        m = train_df[f"mask_{lab}"].values
        pos = (y == 1) & (m == 1)
        neg = (y == 0) & (m == 1)
        pw = (neg.sum() + 1.0) / (pos.sum() + 1.0)
        pos_weight.append(pw)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

    # Datasets
    train_ds = CheXpertDataset(train_df, labels)
    val_ds = CheXpertDataset(val_df, labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Device and model
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(args.device)
    model = build_model(args.model, n_outputs=len(labels)).to(device)

    # Optimiser with different LRs for head/backbone
    head_params = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    optimiser = optim.Adam([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ])

    best_macro_f1 = -1.0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, m, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            logits = model(x)
            loss = bce_masked_loss(logits, y, m, pos_weight=pos_weight.to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_loader))

        # Quick val
        val_probs, val_metrics = evaluate(model, val_loader, device, labels)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_macroF1={val_metrics['macro_f1']:.3f} val_microF1={val_metrics['micro_f1']:.3f}")

        # Save best
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")
            val_probs.to_csv(args.out_dir / "val_probs.csv", index=False)
            (args.out_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))

    print(f"Best val macro F1: {best_macro_f1:.3f}")


if __name__ == "__main__":
    main()

