"""EuroSAT multispectral (6-band) dataset utilities.

The EuroSAT "all bands" release stores every 64x64 Sentinel-2 patch as a
13-band GeoTIFF.  This module:

* selects 6 of those bands,
* scales reflectance to ~[0, 1],
* serves train / val / test subsets defined by a *fixed* ``split.json`` so that
  every experiment is evaluated on exactly the same test images,
* optionally applies light, spectrally-safe augmentation (train split only).
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Zero-based channel indices into the 13-band EuroSAT GeoTIFFs.
# IMPORTANT: this is the exact selection the released ``best_model.pth`` was
# trained with.  Changing it requires retraining.  See README ("Band selection").
SELECTED_BANDS: List[int] = [1, 2, 3, 7, 10, 11]

CLASS_NAMES: List[str] = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

REFLECTANCE_SCALE: float = 10000.0  # Sentinel-2 L1C digital numbers -> reflectance
TIF_EXTENSIONS = (".tif", ".tiff")


# --------------------------------------------------------------------------- #
# File discovery and fixed splits
# --------------------------------------------------------------------------- #
def list_samples(root_dir: str) -> Tuple[List[str], List[int], List[str]]:
    """Return ``(paths, labels, class_names)`` in a deterministic (sorted) order.

    Sorting matters: ``split.json`` stores *indices* into this list, so the order
    must be identical on every machine and every run.
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    class_names = sorted(
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    )
    if not class_names:
        raise RuntimeError(f"No class sub-folders found in {root_dir}")

    paths: List[str] = []
    labels: List[int] = []
    for label, cls in enumerate(class_names):
        cls_dir = os.path.join(root_dir, cls)
        for name in sorted(os.listdir(cls_dir)):
            if name.lower().endswith(TIF_EXTENSIONS):
                paths.append(os.path.join(cls_dir, name))
                labels.append(label)

    if not paths:
        raise RuntimeError(f"No .tif files found under {root_dir}")
    return paths, labels, class_names


def create_split(
    num_samples: int,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Dict[str, List[int]]:
    """Create a seeded random train/val/test split (the remainder is test)."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_samples, generator=generator).tolist()
    n_train = int(train_frac * num_samples)
    n_val = int(val_frac * num_samples)
    return {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }


def load_or_create_split(
    split_file: str, num_samples: int, seed: int = 42
) -> Dict[str, List[int]]:
    """Load ``split_file`` if it exists, otherwise create and save it."""
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            split = json.load(f)
        total = sum(len(split[k]) for k in ("train", "val", "test"))
        if total != num_samples:
            raise ValueError(
                f"{split_file} covers {total} samples but the dataset has "
                f"{num_samples}. Delete the file to regenerate it, or point "
                f"--data-dir at the same dataset the split was made for."
            )
        if max(max(split[k]) for k in ("train", "val", "test")) >= num_samples:
            raise ValueError(f"{split_file} contains out-of-range indices.")
        return split

    split = create_split(num_samples, seed=seed)
    parent = os.path.dirname(os.path.abspath(split_file))
    os.makedirs(parent, exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(split, f)
    print(f"Saved new split to {split_file} "
          f"(train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])})")
    return split


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def augment_6band(img: torch.Tensor) -> torch.Tensor:
    """Random flips / 90-degree rotations / sensor noise / per-band gain.

    ``img`` has shape ``(C, H, W)``.  All operations are geometry- or
    radiometry-preserving, so they are safe for multispectral data (no colour
    jitter, which would break the spectral meaning of the channels).
    """
    if torch.rand(1).item() > 0.5:
        img = torch.flip(img, dims=[2])                       # horizontal flip
    if torch.rand(1).item() > 0.5:
        img = torch.flip(img, dims=[1])                       # vertical flip
    k = int(torch.randint(0, 4, (1,)).item())
    img = torch.rot90(img, k=k, dims=[1, 2])                  # 0/90/180/270 deg
    if torch.rand(1).item() > 0.5:
        img = img + torch.randn_like(img) * 0.01              # additive noise
    if torch.rand(1).item() > 0.5:
        gain = torch.empty(img.shape[0]).uniform_(0.9, 1.1).view(-1, 1, 1)
        img = img * gain                                      # per-band gain
    return img


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class EuroSat6BandDataset(Dataset):
    """EuroSAT 6-band dataset restricted to one fixed split.

    Args:
        root_dir:   Folder containing one sub-folder per class with ``.tif`` files.
        split:      ``"train"``, ``"val"``, ``"test"`` or ``"all"``.
        split_file: Path to ``split.json`` (created automatically if missing).
        bands:      Zero-based band indices to keep from the 13-band tiles.
        augment:    Apply :func:`augment_6band`. Defaults to ``True`` for the
                    train split and ``False`` otherwise.
        seed:       Seed used only when ``split_file`` has to be created.

    Each item is ``(tensor[len(bands), H, W] float32, label int)``.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        split_file: str = "split.json",
        bands: Sequence[int] = tuple(SELECTED_BANDS),
        augment: Optional[bool] = None,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be one of: train, val, test, all")

        self.bands = list(bands)
        self.augment = (split == "train") if augment is None else augment

        paths, labels, self.classes = list_samples(root_dir)
        if split == "all":
            indices = list(range(len(paths)))
        else:
            indices = load_or_create_split(split_file, len(paths), seed)[split]

        self.image_paths = [paths[i] for i in indices]
        self.labels = [labels[i] for i in indices]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = read_6band_tile(self.image_paths[idx], self.bands)
        if self.augment:
            img = augment_6band(img)
        return img, self.labels[idx]


def read_6band_tile(path: str, bands: Sequence[int] = tuple(SELECTED_BANDS)) -> torch.Tensor:
    """Read one GeoTIFF and return a normalised ``(len(bands), H, W)`` tensor.

    Accepts either the original 13-band tiles or tiles that were already reduced
    to ``len(bands)`` channels (e.g. exported test samples).
    """
    img = tiff.imread(path).astype(np.float32)
    if img.ndim != 3:
        raise ValueError(f"Expected a 3-D (H, W, C) tile, got {img.shape} for {path}")
    if img.shape[2] == len(bands):
        pass  # already band-selected
    elif img.shape[2] > max(bands):
        img = img[:, :, list(bands)]
    else:
        raise ValueError(f"{path}: cannot select bands {list(bands)} from {img.shape[2]} channels")
    tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (C, H, W)
    return tensor / REFLECTANCE_SCALE


# --------------------------------------------------------------------------- #
# DataLoader helper
# --------------------------------------------------------------------------- #
def get_dataloaders(
    root_dir: str,
    split_file: str = "split.json",
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    bands: Sequence[int] = tuple(SELECTED_BANDS),
    pin_memory: Optional[bool] = None,
) -> Tuple[Dict[str, DataLoader], List[str]]:
    """Build train/val/test loaders from the fixed split. Returns ``(loaders, classes)``."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    datasets = {
        name: EuroSat6BandDataset(root_dir, name, split_file, bands, seed=seed)
        for name in ("train", "val", "test")
    }
    loaders = {
        name: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for name, ds in datasets.items()
    }
    return loaders, datasets["train"].classes
