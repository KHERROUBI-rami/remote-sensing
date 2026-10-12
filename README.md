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

| Channel | Index | Band |
|---|---|---|
| 1 | 1 | B2 (Blue) |
| 2 | 2 | B3 (Green) |
| 3 | 3 | B4 (Red) |
| 4 | 7 | B8 (NIR) |
| 5 | 10 | B11 (SWIR-1) |
| 6 | 11 | B12 (SWIR-2) |

**Band selection (verified).** The EuroSAT `.tif` tiles do **not** follow the usual Sentinel-2 order: the bands are stored as B01, B02, ..., B12 and then **B8A last** (index 12). That is why B11 and B12 sit at indices 10 and 11, and why `SELECTED_BANDS = [1, 2, 3, 7, 10, 11]` really is B2, B3, B4, B8, B11, B12. The reflectance profile of one Forest tile confirms it: index 9 is almost zero (cirrus, B10), index 10 is mid-range (SWIR-1), index 11 is low (SWIR-2) and index 12 is the brightest NIR-like band (B8A).

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Band | B01 | B02 | B03 | B04 | B05 | B06 | B07 | B08 | B09 | B10 | B11 | B12 | B8A |
| Mean DN (one Forest tile) | 1150 | 831 | 652 | 410 | 659 | 1998 | 2707 | 2612 | 579 | 8 | 1305 | 496 | 3115 |

Check it on your own copy of the data:

```python
import glob, tifffile
f = sorted(glob.glob("path/to/tif/Forest/*.tif"))[0]
a = tifffile.imread(f)
print([round(float(a[:, :, i].mean())) for i in range(a.shape[2])])
```

Do not use `[1, 2, 3, 7, 11, 12]`: with these tiles that would select B12 and B8A and no longer match the trained weights.

**Preprocessing:** reflectance is scaled by `1/10000`. **Training augmentation:** random horizontal/vertical flips, random 90° rotations, Gaussian noise (σ = 0.01) and a per-band gain in [0.9, 1.1]. Colour jitter is deliberately not used, since it would corrupt the spectral meaning of the channels.

**Fixed split:** `data/eurosat_split_indices.json` (pass it with `--split-file`; any path works, the default is `split.json`) stores train / validation / test indices (80 / 10 / 10 %, seed 42; 21,600 / 2,700 / 2,700 images). The file is created on first run and then reused, so every experiment is scored on identical test images. The indices refer to the **sorted** file listing (see `dataset.list_samples`); regenerate the split by deleting `split.json` whenever the dataset folder changes.

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
* Everything runs on CPU or GPU; global pooling lets the backbone accept other tile sizes, although the model was trained on 64 x 64 tiles only.

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
git clone https://github.com/KHERROUBI-rami/remote-sensing.git && cd remote-sensing
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
  --split-file data/eurosat_split_indices.json \
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

### 5. Deploy: ONNX export and INT8 quantisation

The PyTorch checkpoint is converted to ONNX and then quantised with ONNX Runtime, so the model can run on CPU-only edge hardware without PyTorch.

```bash
D=data/EuroSATallBands/ds/images/remote_sensing/otherDatasets/sentinel_2/tif
S=data/eurosat_split_indices.json

# 1) Export to ONNX (batch size dynamic; checks logits against PyTorch)
python export_onnx.py --weights outputs/best_model.pth --output outputs/model_fp32.onnx

# 2) Quantise: static INT8 (calibrated on 500 class-balanced *training* tiles); add dynamic / fp16 if wanted
python quantize.py --onnx outputs/model_fp32.onnx --data-dir $D --split-file $S \
                   --mode static dynamic fp16 --output-dir outputs

# 3) Compare accuracy, size and latency on the held-out TEST split (first model = reference)
python benchmark_onnx.py --models outputs/model_fp32.onnx outputs/model_int8_static.onnx \
                         --data-dir $D --split-file $S --threads 1 --save outputs/benchmark.md

# 4) Inference with any ONNX file
python infer.py --weights outputs/model_int8_static.onnx path/to/tile.tif
```

| Variant | What it does | File size* |
|---|---|---|
| `model_fp32.onnx` | Straight export of `best_model.pth` | 12.4 MB |
| `model_int8_static.onnx` | INT8 weights **and** activations (QDQ, per-channel weights, calibrated) | ~3.6 MB (about 3.5x smaller) |
| `model_int8_dynamic.onnx` | INT8 weights, activations quantised at run time | ~3.3 MB |
| `model_fp16.onnx` | Float16 weights, float32 inputs/outputs | ~6.2 MB |

\* Sizes depend only on the architecture, so they are reproducible; accuracy and latency depend on your data and hardware, so **measure them with `benchmark_onnx.py`** and record them in the table below.

**Quantised results (fill in after running step 3):**

| Model | Test accuracy | Test F1 (weighted) | Latency, 1 CPU thread |
|---|---|---|---|
| FP32 ONNX | _TBD_ | _TBD_ | _TBD_ |
| INT8 static | _TBD_ | _TBD_ | _TBD_ |

Notes on quantising MobileNetV3:

* Post-training INT8 of MobileNetV3 (Hardswish and squeeze-excite blocks) can lose far more accuracy than for plain ResNets, and the loss depends on the calibration settings. If the INT8 accuracy drop is large, try `--calib-method entropy` or `minmax`, `--keep-head-fp32`, `--activation-type int8`, or more calibration tiles (`--calib-per-class`). If none of that is enough, quantisation-aware training is the next step.
* Calibration uses **training** tiles only, with no augmentation. Never calibrate or evaluate on the test split.
* FP16 halves the file size but is typically **not faster on CPU**; it is mainly useful for GPU/NPU targets.
* Normalisation (`/ 10000`) is done outside the graph. Any runtime that feeds the ONNX model must apply the same preprocessing and the same band selection as `dataset.read_6band_tile`.

### 6. Notebook

`eurosat_6band_mobilenetv3.ipynb` walks through the whole pipeline (setup, dataset, model, training, evaluation, inference) and imports the modules above. Set `DATA_DIR` in the configuration cell, or export `EUROSAT_DIR`.

## Repository layout

```
├── dataset.py                       # EuroSat6BandDataset, fixed-split handling, augmentation
├── model.py                         # MobileNetV3Classifier (6-channel), checkpoint loader
├── train.py                         # training / validation / early stopping / evaluation CLI
├── infer.py                         # single-image / folder inference (.pth or .onnx)
├── export_onnx.py                   # best_model.pth -> ONNX (+ parity check vs PyTorch)
├── quantize.py                      # static / dynamic INT8 and FP16 conversion of the ONNX model
├── benchmark_onnx.py                # accuracy, size and latency of ONNX variants on the test split
├── eurosat_6band_mobilenetv3.ipynb  # end-to-end walkthrough (training and evaluation)
├── incubator_3.py                   # earlier smoke test for a TorchScript INT8 model
├── data/eurosat_split_indices.json  # fixed train/val/test indices (commit this)
├── docs/model_technical_report_en.pdf
├── requirements.txt
└── README.md
```

## Limitations

* EuroSAT patches are clean, centre-cropped scenes of a single class. Expect lower accuracy on real satellite passes (clouds, mixed scenes, sensor differences).
* Reported numbers come from a single run and a single split, with no confidence intervals.
* Latency, energy and memory on actual CubeSat hardware have not been measured here. Quantised-model accuracy is only reported once you run `benchmark_onnx.py` (see section 5).

## Citation

> P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS, 2019.

## License

Add a `LICENSE` file of your choice (MIT or Apache-2.0 are common). EuroSAT and the torchvision ImageNet weights have their own terms.
