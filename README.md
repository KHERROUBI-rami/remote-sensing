# EuroSAT 6-Band Classification with MobileNetV3 for CubeSat / Edge Deployment

A compact PyTorch pipeline that classifies Sentinel-2 land-use / land-cover patches from **6 spectral bands** using an ImageNet-pretrained **MobileNetV3-Large** adapted to multispectral input. The model has about **3.25 M parameters** (~13 MB in FP32), which makes it a good starting point for on-board processing on size-, power- and compute-constrained platforms such as CubeSats.

| Metric (held-out test split, 2,700 images) | Result |
|---|---|
| Accuracy | **97.37 %** |
| F1-score (weighted) | **97.37 %** |

## Motivation

Downlinking raw imagery from small satellites is bandwidth-limited. Classifying scenes on board and sending only labels, or only the tiles that matter, saves bandwidth and latency. That requires a model that is accurate, small and cheap to run. Standard RGB networks throw away the near-infrared and shortwave-infrared channels that separate vegetation, water, bare soil and built-up areas far better than visible light alone, so this project keeps those channels and pays only a one-layer cost for them.

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) (Helber et al., 2019): 27,000 Sentinel-2 patches, 64 x 64 px, 10 classes (roughly 2,000-3,000 images per class).

`AnnualCrop` · `Forest` · `HerbaceousVegetation` · `Highway` · `Industrial` · `Pasture` · `PermanentCrop` · `Residential` · `River` · `SeaLake`

We use the **13-band "all bands" GeoTIFF release** and keep six channels (zero-based indices `[1, 2, 3, 7, 10, 11]`, defined in `dataset.py` as `SELECTED_BANDS`):

| Channel | Index | Intended band |
|---|---|---|
| 1 | 1 | B2 (Blue) |
| 2 | 2 | B3 (Green) |
| 3 | 3 | B4 (Red) |
| 4 | 7 | B8 (NIR) |
| 5 | 10 | B11 (SWIR-1) |
| 6 | 11 | B12 (SWIR-2) |

> **Band selection — please verify before publishing claims.** In the standard Sentinel-2 13-band ordering (B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12), index 10 is **B10** and index 11 is **B11**, while **B11/B12 are indices 11/12**. The released `best_model.pth` was trained with `[1, 2, 3, 7, 10, 11]`, so unless your copy of the tiles uses a different band order, the model's last two channels are B10 and B11, not B11 and B12. Check one tile (for example, B10 cirrus values are near-constant and very low) and then either correct the table above or retrain with `[1, 2, 3, 7, 11, 12]`. Inference with the provided weights must use the same indices the model was trained with.

**Preprocessing:** reflectance is scaled by `1/10000`. **Training augmentation:** random horizontal/vertical flips, random 90° rotations, Gaussian noise (σ = 0.01) and a per-band gain in [0.9, 1.1]. Colour jitter is deliberately not used, since it would corrupt the spectral meaning of the channels.

**Fixed split:** `split.json` stores train / validation / test indices (80 / 10 / 10 %, seed 42; 21,600 / 2,700 / 2,700 images). The file is created on first run and then reused, so every experiment is scored on identical test images. The indices refer to the **sorted** file listing (see `dataset.list_samples`); regenerate the split by deleting `split.json` whenever the dataset folder changes.

## Model architecture and transfer learning

```
6-band 64x64 input
  -> MobileNetV3-Large features (ImageNet pretrained; first conv widened 3 -> 6 channels)
  -> global average pooling (960)
  -> Linear 960->256, ReLU, Dropout 0.67
  -> Linear 256->128, ReLU, Dropout 0.47
  -> Linear 128->10
```

* **First-layer surgery:** the pretrained 3-channel filters initialise channels 1-3; the extra channels start from the *mean* pretrained filter. This preserves activation scale and lets the network reuse its ImageNet features from epoch 1.
* **Full fine-tuning** of all layers (not just the head), Adam, `lr=1e-4`, `weight_decay=1e-3`.
* **Regularisation:** label smoothing 0.1, heavy dropout in the head, gradient clipping (1.0), `ReduceLROnPlateau` and early stopping (patience 7) on validation loss; the checkpoint with the best validation accuracy is kept.
* Everything runs on CPU or GPU, and the all-convolutional backbone accepts other tile sizes through global pooling.

## Results

Original run, 10 epochs, best epoch by validation accuracy (epoch 7):

| Split | Loss | Accuracy | F1 (weighted) | Recall | Precision |
|---|---|---|---|---|---|
| Validation | 0.5805 | 97.11 % | 97.11 % | 97.11 % | 97.13 % |
| **Test** | 0.5763 | **97.37 %** | **97.37 %** | 97.37 % | 97.38 % |

Losses are higher than usual because of label smoothing. Training accuracy (~94 %) sits *below* validation accuracy because it is measured on augmented images with dropout active. Re-running will land close to, but not exactly on, these numbers (GPU non-determinism, augmentation randomness). `train.py` writes `training_curves.png`, `confusion_matrix.png` and `metrics.json` to the output folder.

## Quick start

### 1. Install

```bash
git clone <your-repo-url> && cd eurosat-6band-mobilenetv3
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

### 2. Get the data

```bash
wget https://madm.dfki.de/files/sentinel/EuroSATallBands.zip
unzip -q EuroSATallBands.zip -d data/EuroSATallBands
# class folders end up in:
# data/EuroSATallBands/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/
```

### 3. Train

```bash
python train.py \
  --data-dir data/EuroSATallBands/ds/images/remote_sensing/otherDatasets/sentinel_2/tif \
  --split-file split.json \
  --output-dir outputs \
  --epochs 10 --batch-size 32 --lr 1e-4
```

Outputs in `outputs/`: `best_model.pth`, `training_curves.png`, `confusion_matrix.png`, `metrics.json`. The script finishes by evaluating the best checkpoint on the test split. Run `python train.py --help` for all options. The first run downloads the torchvision ImageNet weights (use `--no-pretrained` to skip).

### 4. Inference with `best_model.pth`

```bash
# single tile (13-band EuroSAT tile or a tile already reduced to 6 bands)
python infer.py --weights outputs/best_model.pth path/to/tile.tif

# a whole folder, top-3 classes, plus an RGB preview of the last tile
python infer.py --weights outputs/best_model.pth path/to/tiles/ --top-k 3 --save-rgb preview.png
```

From Python:

```python
import torch
from model import load_checkpoint
from infer import predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_checkpoint("outputs/best_model.pth", device)
top3, _ = predict(model, "path/to/tile.tif", device, top_k=3)
print(top3)   # [('Forest', 0.98), ...]
```

### 5. Notebook

`eurosat_6band_mobilenetv3.ipynb` walks through the whole pipeline (setup, dataset, model, training, evaluation, inference) and imports the modules above. Set `DATA_DIR` in the configuration cell, or export `EUROSAT_DIR`.

## Repository layout

```
├── dataset.py                       # EuroSat6BandDataset, fixed-split handling, augmentation
├── model.py                         # MobileNetV3Classifier (6-channel), checkpoint loader
├── train.py                         # training / validation / early stopping / evaluation CLI
├── infer.py                         # single-image and folder inference CLI
├── eurosat_6band_mobilenetv3.ipynb  # end-to-end walkthrough
├── split.json                       # fixed train/val/test indices (commit this)
├── requirements.txt
└── README.md
```

## Limitations

* EuroSAT patches are clean, centre-cropped scenes of a single class. Expect lower accuracy on real satellite passes (clouds, mixed scenes, sensor differences).
* Reported numbers come from a single run and a single split, with no confidence intervals.
* Latency, energy and memory on actual CubeSat hardware have not been measured here.

## Citation

> P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS, 2019.

## License

Add a `LICENSE` file of your choice (MIT or Apache-2.0 are common). EuroSAT and the torchvision ImageNet weights have their own terms.
