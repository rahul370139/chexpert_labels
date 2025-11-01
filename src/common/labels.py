"""Canonical CheXpert label definitions shared across the project."""

from __future__ import annotations

from typing import Dict, List

# 13 CheXpert findings excluding "No Finding"
CHEXPERT13: List[str] = [
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

# 14-label variant including "No Finding"
CHEXPERT14: List[str] = CHEXPERT13 + ["No Finding"]

# Convenience mapping in case we need to normalise aliases from different sources.
CHEXPERT_LABEL_MAP: Dict[str, str] = {label.lower(): label for label in CHEXPERT14}


def normalise_label(label: str) -> str:
    """Normalise an arbitrary label string onto the CheXpert schema."""
    canonical = CHEXPERT_LABEL_MAP.get(label.strip().lower())
    if not canonical:
        raise KeyError(f"Label '{label}' is not part of the CheXpert schema.")
    return canonical


def get_label_list(name: str) -> List[str]:
    """Utility to fetch the requested label set."""
    key = name.strip().lower()
    if key in {"chexpert13", "chexpert_13", "13"}:
        return CHEXPERT13
    if key in {"chexpert14", "chexpert_14", "14"}:
        return CHEXPERT14
    raise ValueError(f"Unknown label group '{name}'. Expected 'chexpert13' or 'chexpert14'.")

# Score column prefixes for consistent naming across pipelines
SCORE_PREFIX: Dict[str, str] = {
    "raw": "y_pred_",  # Raw model predictions
    "cal": "y_cal_",   # Calibrated probabilities
}

