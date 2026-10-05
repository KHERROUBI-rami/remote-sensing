"""Run inference with a trained checkpoint on one or more 6-band GeoTIFFs.

Examples
--------
    python infer.py --weights best_model.pth path/to/tile.tif
    python infer.py --weights best_model.pth tiles_dir/ --top-k 3 --save-rgb out.png

Inputs may be original 13-band EuroSAT tiles (the 6 training bands are
selected automatically) or tiles already reduced to 6 bands.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dataset import CLASS_NAMES, SELECTED_BANDS, read_6band_tile
from model import load_checkpoint


def collect_paths(inputs: List[str]) -> List[str]:
    paths: List[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for ext in ("*.tif", "*.tiff"):
                paths.extend(sorted(glob.glob(os.path.join(item, ext))))
        else:
            paths.append(item)
    if not paths:
        raise FileNotFoundError("No .tif files found for the given inputs.")
    return paths


@torch.no_grad()
def predict(model, tile_path: str, device, top_k: int = 3) -> Tuple[List[Tuple[str, float]], torch.Tensor]:
    """Return the top-k ``(class_name, probability)`` pairs and the input tensor."""
    x = read_6band_tile(tile_path, SELECTED_BANDS).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0].cpu()
    values, indices = probs.topk(min(top_k, len(CLASS_NAMES)))
    return [(CLASS_NAMES[i], float(v)) for v, i in zip(values, indices)], x[0].cpu()


def save_rgb_preview(tensor: torch.Tensor, title: str, path: str) -> None:
    """Save a true-colour preview. Channels are ordered [B2, B3, B4, ...] so RGB = [2, 1, 0]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = tensor[[2, 1, 0]].permute(1, 2, 0).numpy()
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    plt.figure(figsize=(3, 3))
    plt.imshow(rgb)
    plt.title(title, fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Classify EuroSAT 6-band tiles.")
    p.add_argument("inputs", nargs="+", help=".tif file(s) or folder(s) of .tif files")
    p.add_argument("--weights", default="best_model.pth", help="Path to the trained state_dict.")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--save-rgb", default=None,
                   help="Save an RGB preview of the (last) tile to this PNG path.")
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.weights, device, num_bands=len(SELECTED_BANDS), num_classes=len(CLASS_NAMES))

    for path in collect_paths(args.inputs):
        top, tensor = predict(model, path, device, args.top_k)
        best_name, best_prob = top[0]
        print(f"{os.path.basename(path)} -> {best_name} ({best_prob * 100:.2f}%)")
        for name, prob in top[1:]:
            print(f"    {name}: {prob * 100:.2f}%")
        if args.save_rgb:
            save_rgb_preview(tensor, f"{best_name} ({best_prob * 100:.1f}%)", args.save_rgb)


if __name__ == "__main__":
    main()
