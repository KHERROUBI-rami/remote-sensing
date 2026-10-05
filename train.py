"""Train the 6-band MobileNetV3 classifier on EuroSAT.

Example
-------
    python train.py --data-dir /path/to/EuroSATallBands/.../tif \
                    --split-file split.json --epochs 10 --output-dir runs/exp1
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, get_dataloaders
from model import MobileNetV3Classifier, count_parameters


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(labels: Sequence[int], preds: Sequence[int], losses: Sequence[float]) -> Dict[str, float]:
    """Loss, accuracy and weighted F1 / recall / precision (all in %, loss raw)."""
    labels, preds = np.asarray(labels), np.asarray(preds)
    kw = dict(average="weighted", zero_division=0)
    return {
        "loss": float(np.mean(losses)) if len(losses) else 0.0,
        "acc": 100.0 * float((preds == labels).mean()),
        "f1": 100.0 * f1_score(labels, preds, **kw),
        "recall": 100.0 * recall_score(labels, preds, **kw),
        "precision": 100.0 * precision_score(labels, preds, **kw),
    }


class EarlyStopping:
    """Stop when validation loss has not improved by ``delta`` for ``patience`` epochs."""

    def __init__(self, patience: int = 7, delta: float = 1e-3) -> None:
        self.patience, self.delta = patience, delta
        self.best_loss: Optional[float] = None
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss, self.counter = val_loss, 0
        else:
            self.counter += 1
            self.stop = self.counter >= self.patience


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm: float = 1.0) -> Dict[str, float]:
    model.train()
    losses: List[float] = []
    preds: List[int] = []
    labels_all: List[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        losses.append(loss.item())
        preds.extend(outputs.argmax(1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    return compute_metrics(labels_all, preds, losses)


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_predictions: bool = False):
    """Evaluate on ``loader``; optionally also return ``(labels, preds)``."""
    model.eval()
    losses: List[float] = []
    preds: List[int] = []
    labels_all: List[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        losses.append(criterion(outputs, labels).item())
        preds.extend(outputs.argmax(1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    metrics = compute_metrics(labels_all, preds, losses)
    return (metrics, labels_all, preds) if return_predictions else metrics


def plot_history(history: Dict[str, list], save_path: Optional[str] = None, show: bool = False) -> None:
    """Plot loss and accuracy curves for train vs. validation."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_confusion_matrix(labels, preds, class_names, title: str = "Confusion matrix (test split)",
                          save_path: Optional[str] = None, show: bool = False) -> np.ndarray:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return cm


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #
def fit(model, loaders: Dict[str, DataLoader], device, epochs: int, lr: float, weight_decay: float,
        label_smoothing: float, patience: int, checkpoint_path: str, verbose: bool = True):
    """Train with ReduceLROnPlateau + early stopping. Saves the best-val-accuracy weights."""
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    stopper = EarlyStopping(patience=patience)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr")}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        va = evaluate(model, loaders["val"], criterion, device)
        scheduler.step(va["loss"])
        stopper(va["loss"])

        saved = va["acc"] > best_val_acc
        if saved:
            best_val_acc = va["acc"]
            torch.save(model.state_dict(), checkpoint_path)

        lr_now = optimizer.param_groups[0]["lr"]
        for key, val in (("train_loss", tr["loss"]), ("train_acc", tr["acc"]),
                         ("val_loss", va["loss"]), ("val_acc", va["acc"]), ("lr", lr_now)):
            history[key].append(val)

        if verbose:
            print(f"Epoch [{epoch:02d}/{epochs}] {'(saved)' if saved else ''}")
            print(f"  TRAIN | loss {tr['loss']:.4f} | acc {tr['acc']:.2f}% | F1 {tr['f1']:.2f}%")
            print(f"  VAL   | loss {va['loss']:.4f} | acc {va['acc']:.2f}% | F1 {va['f1']:.2f}% | lr {lr_now:.2e}")

        if stopper.stop:
            print(f"Early stopping at epoch {epoch} (best val acc {best_val_acc:.2f}%)")
            break

    return history, best_val_acc, criterion


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a 6-band MobileNetV3 on EuroSAT.")
    p.add_argument("--data-dir", required=True,
                   help="Folder with one sub-folder of .tif files per class.")
    p.add_argument("--split-file", default="split.json",
                   help="Fixed train/val/test indices (created on first run if missing).")
    p.add_argument("--output-dir", default=".", help="Where to write best_model.pth, plots and metrics.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=7, help="Early-stopping patience (epochs).")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true",
                   help="Do not load ImageNet weights (random init).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loaders, classes = get_dataloaders(
        args.data_dir, args.split_file, args.batch_size, args.num_workers, args.seed)
    print({k: len(v.dataset) for k, v in loaders.items()}, "| classes:", len(classes))

    model = MobileNetV3Classifier(6, len(classes), pretrained=not args.no_pretrained).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    ckpt = os.path.join(args.output_dir, "best_model.pth")
    history, best_val_acc, criterion = fit(
        model, loaders, device, args.epochs, args.lr, args.weight_decay,
        args.label_smoothing, args.patience, ckpt)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    plot_history(history, os.path.join(args.output_dir, "training_curves.png"))

    # Final, one-off evaluation on the held-out test split using the best weights.
    model.load_state_dict(torch.load(ckpt, map_location=device))
    test, y_true, y_pred = evaluate(model, loaders["test"], criterion, device, return_predictions=True)
    plot_confusion_matrix(y_true, y_pred, classes, save_path=os.path.join(args.output_dir, "confusion_matrix.png"))
    print(f"TEST | loss {test['loss']:.4f} | acc {test['acc']:.2f}% | F1 {test['f1']:.2f}% "
          f"| recall {test['recall']:.2f}% | precision {test['precision']:.2f}%")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"best_val_acc": best_val_acc, "test": test, "history": history, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
