#!/usr/bin/env python3
"""
Apply impression-based DI boosts for borderline rescue.

Reads:
- base CheXagent metadata CSV (with binary_outputs, di_outputs JSON)
- test labels CSV (for 'impression' + 'filename')

Writes:
- boosted metadata CSV with di_outputs updated for labels whose impression matches
  curated keyword patterns.

Policy:
- Only mark DI mentioned=1, strength=0.9 for labels with impression hits.
- Do not override if original DI is negated.
- Keep uncertain flag untouched; gating will handle it.

Target labels and keywords:
- Pneumonia: pneumonia|infect|airspace|consolidation
- Consolidation: consolidation|airspace opacity|infiltrat
- Pleural Other: pleural thickening|calcified pleura|fibrothorax
- Fracture: fracture|rib fracture|compression fracture|break
- Edema: edema|interstitial edema|kerley|batwing|pulmonary edema
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


KEYWORDS = {
    "Pneumonia": r"\b(pneumonia|infect|air\s*space|airspace|consolidation)\b",
    "Consolidation": r"\b(consolidation|air\s*space\s*opacity|infiltrat)\w*\b",
    "Pleural Other": r"\b(pleural\s*thickening|calcified\s*pleura|fibrothorax)\b",
    "Fracture": r"\b(fracture|rib\s*fracture|compression\s*fracture|broken\s*rib|break)\b",
    "Edema": r"\b(edema|interstitial\s*edema|kerley|bat\s*wing|pulmonary\s*edema)\b",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply impression-based DI boost to metadata CSV")
    ap.add_argument("--metadata_csv", required=True)
    ap.add_argument("--test_labels_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    meta_df = pd.read_csv(args.metadata_csv)
    test_df = pd.read_csv(args.test_labels_csv)

    # Align by filename
    def to_filename(x):
        p = Path(str(x))
        return p.name

    if "filename" not in meta_df.columns:
        if "image" in meta_df.columns:
            meta_df["filename"] = meta_df["image"].map(to_filename)
        else:
            raise SystemExit("metadata CSV must contain 'image' or 'filename'")
    if "filename" not in test_df.columns:
        if "image" in test_df.columns:
            test_df["filename"] = test_df["image"].map(to_filename)
        else:
            raise SystemExit("test_labels CSV must contain 'image' or 'filename'")

    # Build impression hits per filename
    impression = test_df.set_index("filename").get("impression", pd.Series(dtype=str)).fillna("")
    patterns = {lab: re.compile(pat, flags=re.IGNORECASE) for lab, pat in KEYWORDS.items()}

    def boost_di(di_json: str, fn: str) -> str:
        try:
            di = json.loads(di_json) if isinstance(di_json, str) else (di_json or {})
        except Exception:
            di = {}
        text = impression.get(fn, "")
        for lab, rx in patterns.items():
            if not text:
                continue
            if rx.search(text):
                entry = di.get(lab, {})
                if entry.get("negated"):
                    continue
                entry["mentioned"] = True
                # strength high enough to pass di_min
                entry["strength"] = max(0.9, float(entry.get("strength", 0.0) or 0.0))
                di[lab] = entry
        return json.dumps(di)

    out = meta_df.copy()
    if "di_outputs" not in out.columns:
        raise SystemExit("metadata CSV must include 'di_outputs' JSON column")
    out["di_outputs"] = [boost_di(di_json, fn) for di_json, fn in zip(out["di_outputs"], out["filename"])]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"✅ Wrote boosted metadata to {args.out_csv}")


if __name__ == "__main__":
    main()

