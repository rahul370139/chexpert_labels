#!/usr/bin/env python3
"""
Deterministic patient-wise splitting with leakage guards and label summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.common.labels import CHEXPERT13


def _extract_patient_id(path_str: str) -> str:
    """
    Extract a patient identifier from a MIMIC-CXR style path (files/p10/p10000001/...).

    Falls back to the parent directory name or filename stem if the expected pattern
    cannot be found so that downstream code still has a unique grouping key.
    """
    path = Path(path_str)
    for part in path.parts:
        if part.startswith("p") and part[1:].isdigit() and len(part) > 2:
            return part
    # Fall back to parent folder if it looks like a patient folder
    if path.parent.name.startswith("p") and path.parent.name[1:].isdigit():
        return path.parent.name
    return path.stem


def _prepare_manifest(manifest_path: Path, id_column: str | None) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    if "filename" not in df.columns:
        if "image" in df.columns:
            df["filename"] = df["image"].apply(lambda x: Path(x).name)
        elif "filepath" in df.columns:
            df.rename(columns={"filepath": "image"}, inplace=True)
            df["filename"] = df["image"].apply(lambda x: Path(x).name)
        else:
            raise ValueError("Manifest must contain an 'image' (full path) or 'filename' column.")

    candidate_cols = [
        id_column,
        "patient_id",
        "subject_id",
        "patient",
    ]
    col = next((c for c in candidate_cols if c and c in df.columns), None)
    if col:
        df["patient_id"] = df[col].astype(str)
    else:
        source_col = "image" if "image" in df.columns else "filename"
        df["patient_id"] = df[source_col].astype(str).map(_extract_patient_id)
    return df


def _derive_split_counts(n_patients: int, train_frac: float, val_frac: float, test_frac: float) -> Tuple[int, int, int]:
    fractions = np.array([train_frac, val_frac, test_frac], dtype=float)
    if np.any(fractions < 0):
        raise ValueError("Split fractions must be non-negative.")
    if not np.isclose(fractions.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Split fractions must sum to 1.0 (got {fractions.sum():.4f}).")

    raw_counts = fractions * n_patients
    counts = np.floor(raw_counts).astype(int)
    remainder = n_patients - counts.sum()
    if remainder > 0:
        # Distribute the remaining patients to splits with largest fractional parts
        fractional = raw_counts - counts
        for idx in np.argsort(fractional)[::-1][:remainder]:
            counts[idx] += 1
    return tuple(int(x) for x in counts)


def _split_patients(patients: np.ndarray, counts: Tuple[int, int, int], seed: int) -> Tuple[Iterable[str], ...]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(patients)

    train_end = counts[0]
    val_end = train_end + counts[1]
    train_patients = perm[:train_end]
    val_patients = perm[train_end:val_end]
    test_patients = perm[val_end:]
    return train_patients, val_patients, test_patients


def _filter_by_patients(df: pd.DataFrame, patients: Iterable[str]) -> pd.DataFrame:
    mask = df["patient_id"].isin(set(patients))
    return df.loc[mask].copy()


def _summarise_split(name: str, split_df: pd.DataFrame) -> Dict[str, int]:
    summary = {
        "patients": split_df["patient_id"].nunique(),
        "images": len(split_df),
    }
    for label in CHEXPERT13:
        if label in split_df.columns:
            summary[f"pos_{label}"] = int(split_df[label].sum())
    print(f"\n{name} split → {summary['patients']} patients / {summary['images']} images")
    if split_df["patient_id"].duplicated().any():
        print("  ⚠️  Multiple entries per patient (expected for multi-view studies).")
    return summary


def write_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  ↳ saved {path} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic patient-wise data splits.")
    parser.add_argument("--manifest", required=True, help="CSV manifest with at least image/filename columns.")
    parser.add_argument("--train_csv", default="outputs/splits/train.csv")
    parser.add_argument("--val_csv", default="outputs/splits/val.csv")
    parser.add_argument("--test_csv", default="outputs/splits/test.csv")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.0)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id_column", default=None, help="Optional column name containing patient IDs.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    df = _prepare_manifest(manifest_path, args.id_column)
    patients = df["patient_id"].unique()
    counts = _derive_split_counts(len(patients), args.train_frac, args.val_frac, args.test_frac)
    train_patients, val_patients, test_patients = _split_patients(patients, counts, args.seed)

    splits = {
        "train": _filter_by_patients(df, train_patients),
        "val": _filter_by_patients(df, val_patients),
        "test": _filter_by_patients(df, test_patients),
    }

    # Leakage check
    overlap = set(train_patients) & set(test_patients)
    if args.val_frac > 0:
        overlap |= set(train_patients) & set(val_patients)
        overlap |= set(val_patients) & set(test_patients)
    if overlap:
        raise RuntimeError(f"Patient leakage detected across splits: {sorted(overlap)[:5]} ...")

    stats: Dict[str, Dict[str, int]] = {}
    for split_name, split_df in splits.items():
        stats[split_name] = _summarise_split(split_name.capitalize(), split_df)

    # Persist CSVs (skip val if empty)
    output_paths = {
        "train": Path(args.train_csv),
        "val": Path(args.val_csv),
        "test": Path(args.test_csv),
    }
    for split_name, path in output_paths.items():
        split_df = splits[split_name]
        if split_name == "val" and split_df.empty:
            continue
        write_split(split_df, path)

    print("\nLabel prevalence (positives per split):")
    header = ["Label", "Train", "Val", "Test"]
    rows: List[List[str]] = [header]
    for label in CHEXPERT13:
        vals = []
        for split_name in ["train", "val", "test"]:
            val = stats.get(split_name, {}).get(f"pos_{label}", 0)
            vals.append(str(val))
        rows.append([label] + vals)
    col_widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    for row in rows:
        print("  " + "  ".join(row[i].ljust(col_widths[i]) for i in range(len(header))))

    print("\n✅ Patient-wise split complete.")


if __name__ == "__main__":
    main()

