#!/usr/bin/env python3
"""
Extract frozen vision-tower embeddings for downstream linear probes.

This implementation uses the CLIP/BiomedCLIP vision backbones that underpin
CheXagent.  By default we load ``openai/clip-vit-large-patch14``; pass
``--vision_model microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`` to
match the medical variant used in CheXagent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel


def _discover_image_column(df) -> Tuple[str, List[str]]:
    for column in ("image", "filepath", "path"):
        if column in df.columns:
            return column, df[column].astype(str).tolist()
    if "filename" in df.columns:
        return "filename", df["filename"].astype(str).tolist()
    raise ValueError("Input CSV must contain one of: image, filepath, path, filename.")


def _resolve_paths(paths: Iterable[str], image_root: Path | None) -> List[Path]:
    resolved = []
    for entry in paths:
        p = Path(entry)
        if not p.is_absolute() and image_root:
            p = image_root / p
        resolved.append(p)
    return resolved


def _load_images(paths: List[Path]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        images.append(img)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CLIP/BiomedCLIP embeddings for CheXagent linear probes.")
    parser.add_argument("--images_csv", required=True, help="CSV with an image/filename column.")
    parser.add_argument("--out_npz", required=True, help="Output .npz with 'filenames' and 'embeddings'.")
    parser.add_argument("--image_root", default=None, help="Optional root directory for relative filenames.")
    parser.add_argument("--vision_model", default="openai/clip-vit-large-patch14", help="HuggingFace vision backbone.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.images_csv)
    image_col, raw_paths = _discover_image_column(df)
    image_root = Path(args.image_root) if args.image_root else None
    image_paths = _resolve_paths(raw_paths, image_root)

    for path in image_paths:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

    device = torch.device(args.device)
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print(f"Loading vision backbone '{args.vision_model}' on {device} ({torch_dtype}).")
    processor = CLIPImageProcessor.from_pretrained(args.vision_model)
    vision_model = CLIPVisionModel.from_pretrained(args.vision_model).to(device=device, dtype=torch_dtype)
    vision_model.eval()

    embeddings: List[np.ndarray] = []
    filenames: List[str] = []
    batch_size = max(1, args.batch_size)

    iterator = range(0, len(image_paths), batch_size)
    for start in tqdm(iterator, desc="Extracting embeddings", total=(len(image_paths) + batch_size - 1) // batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_images = _load_images(batch_paths)
        inputs = processor(images=batch_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=device, dtype=torch_dtype)
        with torch.no_grad():
            outputs = vision_model(pixel_values)
            pooled = outputs.pooler_output  # [B, hidden_dim]
        batch_embeddings = pooled.detach().cpu().to(torch.float32).numpy()
        embeddings.append(batch_embeddings)
        filenames.extend([p.name for p in batch_paths])

    stacked = np.concatenate(embeddings, axis=0)
    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, filenames=np.array(filenames), embeddings=stacked)
    print(f"✅ Saved embeddings to {out_path} (shape={stacked.shape})")


if __name__ == "__main__":
    main()

