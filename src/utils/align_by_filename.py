"""Utility helpers to align probability tables using the filename column."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_filename_column(df: pd.DataFrame) -> pd.DataFrame:
    if "filename" in df.columns:
        return df
    if "image" in df.columns:
        df = df.copy()
        df["filename"] = df["image"].map(lambda x: Path(str(x)).name)
        return df
    raise ValueError("DataFrame must contain a 'filename' or 'image' column.")


def align_to_reference(reference: list[str], df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_filename_column(df)
    aligned = df.set_index("filename").reindex(reference)
    if aligned.isnull().any().any():
        missing = aligned[aligned.isnull().any(axis=1)].index.tolist()
        raise RuntimeError(f"Predictions missing for {len(missing)} filenames (e.g. {missing[:3]}).")
    return aligned.reset_index()

