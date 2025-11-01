#!/usr/bin/env python3
"""
Train one-vs-rest logistic probes on frozen CheXagent vision embeddings.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.common.labels import CHEXPERT13, CHEXPERT14, get_label_list


@dataclass
class SplitSpec:
    name: str
    npz_path: Path
    labels_csv: Path


def _load_embeddings(npz_path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    filenames = data["filenames"]
    embeddings = data["embeddings"].astype(np.float32)
    df = pd.DataFrame({"filename": filenames})
    return df, embeddings


def _merge_embeddings(df_embeddings: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    if "filename" not in labels_df.columns:
        if "image" in labels_df.columns:
            labels_df = labels_df.copy()
            labels_df["filename"] = labels_df["image"].map(lambda x: Path(str(x)).name)
        else:
            raise ValueError("Labels CSV must contain 'filename' or 'image' column.")
    merged = labels_df.merge(df_embeddings, on="filename", how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping filenames between embeddings and labels.")
    return merged


def _train_label_probe(X: np.ndarray, y: np.ndarray) -> Tuple[LogisticRegression | None, np.ndarray]:
    """
    Train logistic regression for a single label. Returns (model, fallback_probs).

    If the label has < 2 unique classes, we skip training and return None along
    with a constant probability vector matching the empirical prevalence.
    """
    unique = np.unique(y)
    prevalence = float(np.mean(y))
    if len(unique) < 2:
        fallback = np.full_like(y, fill_value=prevalence, dtype=np.float32)
        return None, fallback

    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    return model, probs.astype(np.float32)


def _prepare_output_frame(base: pd.DataFrame, probabilities: Dict[str, np.ndarray]) -> pd.DataFrame:
    df = base[["filename"]].copy()
    for label, scores in probabilities.items():
        df[f"y_pred_{label}"] = scores
    return df


def parse_eval_specs(values: Iterable[str]) -> List[SplitSpec]:
    specs: List[SplitSpec] = []
    for value in values:
        parts = value.split(",")
        if len(parts) not in {2, 3}:
            raise ValueError(f"Invalid --eval_split '{value}'. Expected format name,npz_path[,labels_csv].")
        name = parts[0].strip()
        npz_path = Path(parts[1])
        labels_csv = Path(parts[2]) if len(parts) == 3 else None
        if labels_csv is None:
            raise ValueError(f"--eval_split '{value}' missing labels CSV path.")
        specs.append(SplitSpec(name=name, npz_path=npz_path, labels_csv=labels_csv))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CheXagent linear probes on frozen embeddings.")
    parser.add_argument("--train_npz", required=True, help="NPZ file produced by extract_chexagent_embeddings.py")
    parser.add_argument("--train_labels_csv", required=True, help="CSV with ground-truth labels for training split")
    parser.add_argument("--out_dir", default="outputs/linear_probe", help="Directory for models and predictions")
    parser.add_argument("--labels", default="chexpert13", help="Label group (chexpert13 or chexpert14)")
    parser.add_argument(
        "--eval_split",
        action="append",
        default=[],
        help="Optional extra splits to score: name,npz_path,labels_csv (repeat per split)",
    )
    args = parser.parse_args()

    label_list = get_label_list(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_df_embeddings, train_embeddings = _load_embeddings(Path(args.train_npz))
    train_labels_df = pd.read_csv(args.train_labels_csv)
    train_merged = _merge_embeddings(train_df_embeddings, train_labels_df)
    X_train = train_embeddings[train_merged.index.values]

    print(f"Training linear probes on {len(train_merged)} samples with embedding dim {X_train.shape[1]}.")

    models: Dict[str, LogisticRegression] = {}
    train_probs: Dict[str, np.ndarray] = {}
    skipped_labels: List[str] = []

    for label in label_list:
        y_col = label if label in train_merged.columns else f"y_true_{label}"
        if y_col not in train_merged.columns:
            print(f"⚠️  Label '{label}' missing in training CSV; skipping.")
            skipped_labels.append(label)
            continue
        y = train_merged[y_col].astype(int).to_numpy()
        model, probs = _train_label_probe(X_train, y)
        train_probs[label] = probs
        if model is None:
            skipped_labels.append(label)
        else:
            models[label] = model

    # Persist models
    models_path = out_dir / "linear_probe.pkl"
    with open(models_path, "wb") as f:
        pickle.dump({"labels": label_list, "models": models}, f)
    print(f"✅ Saved probe models to {models_path}")

    train_out = _prepare_output_frame(train_merged, train_probs)
    train_out.to_csv(out_dir / "train_raw_probs.csv", index=False)
    print(f"✅ Wrote train probabilities to {out_dir / 'train_raw_probs.csv'}")

    # Score extra splits if requested
    for spec in parse_eval_specs(args.eval_split):
        eval_df_embeddings, eval_embeddings = _load_embeddings(spec.npz_path)
        eval_labels_df = pd.read_csv(spec.labels_csv)
        eval_merged = _merge_embeddings(eval_df_embeddings, eval_labels_df)
        X_eval = eval_embeddings[eval_merged.index.values]
        eval_probs: Dict[str, np.ndarray] = {}
        for label in label_list:
            y_col = label if label in eval_merged.columns else f"y_true_{label}"
            if label not in train_probs:
                continue
            model = models.get(label)
            if model is None:
                # Use training prevalence for this label
                prev = float(np.mean(train_probs[label]))
                eval_probs[label] = np.full(len(eval_merged), fill_value=prev, dtype=np.float32)
                continue
            scores = model.predict_proba(X_eval)[:, 1].astype(np.float32)
            eval_probs[label] = scores
        eval_out = _prepare_output_frame(eval_merged, eval_probs)
        out_csv = out_dir / f"{spec.name}_raw_probs.csv"
        eval_out.to_csv(out_csv, index=False)
        print(f"✅ Wrote {spec.name} probabilities to {out_csv}")

    summary = {
        "n_train": len(train_merged),
        "embedding_dim": int(X_train.shape[1]),
        "trained_labels": sorted(models.keys()),
        "skipped_labels": skipped_labels,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
