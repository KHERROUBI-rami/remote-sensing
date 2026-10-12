# EuroSAT 6-Band MobileNetV3 (INT8 ONNX) for CubeSat / Edge Inference

A compact **INT8-quantised ONNX model** that classifies Sentinel-2 land-use patches from **6 spectral bands** (B2, B3, B4, B8, B11, B12). It is a MobileNetV3-Large adapted from 3 to 6 input channels and fine-tuned on EuroSAT, then quantised for CPU-only edge hardware such as CubeSat on-board computers.

**Model file:** [`models/model_int8_static.onnx`](models/model_int8_static.onnx) (3.57 MB)

## Results

Measured on the 2,700-image test split of EuroSAT (`data/eurosat_split_indices.json`), batch size 1, Google Colab CPU, **1 thread**, ONNX Runtime 1.31.0.

| Model | Size | Accuracy | F1 (weighted) | Latency per tile (median / p95) |
|---|---|---|---|---|
| FP32 ONNX (reference, not included) | 12.40 MB | 97.74 % | 97.74 % | 5.31 / 6.49 ms |
| **INT8 static ONNX (this model)** | **3.57 MB** | **96.74 %** | **96.74 %** | **1.40 / 2.09 ms** |

INT8 quantisation makes the model **3.5x smaller** and about **3.8x faster** on CPU, for a drop of **1.0 accuracy point**.

The original PyTorch training log reported 97.37 % test accuracy for the FP32 model. The split file was created in that Colab run, while this repository lists the dataset files in sorted order, so the exact images in each split can differ slightly (the FP32 model scores 97.74 % here). Numbers come from a single run and have no confidence intervals. Latency was measured on a cloud CPU, not on CubeSat hardware.

## Model details

| | |
|---|---|
| Input | `input`, float32, shape `(batch, 6, 64, 64)` (batch size is dynamic) |
| Output | `output`, float32 logits, shape `(batch, 10)` |
| Preprocessing | pick the 6 bands below, then divide by `10000`. **Normalisation is not inside the graph.** |
| Format | ONNX opset 17, static INT8 (QDQ), per-channel weights |
| Quantisation | calibrated on 500 class-balanced **training** tiles (percentile method) |
| Parameters | about 3.25 M |

**Classes** (output index order): `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`.

**Bands.** EuroSAT `.tif` tiles have 13 bands stored as B01 ... B12 followed by **B8A last**. This differs from the usual Sentinel-2 order (where B8A is index 8), so B11 and B12 are at indices 10 and 11, not 11 and 12. The model uses these zero-based indices:

| Channel | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Band | B2 (Blue) | B3 (Green) | B4 (Red) | B8 (NIR) | B11 (SWIR-1) | B12 (SWIR-2) |
| Index in the `.tif` | 1 | 2 | 3 | 7 | 10 | 11 |

Using any other band selection (for example `[1, 2, 3, 7, 11, 12]`, which would pick B12 and B8A) or scaling gives wrong predictions without raising an error.

## Quick start

```bash
git clone https://github.com/KHERROUBI-rami/remote-sensing.git && cd remote-sensing
pip install onnxruntime numpy tifffile
```

```python
import numpy as np, onnxruntime as ort, tifffile

CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
           "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
BANDS = [1, 2, 3, 7, 10, 11]

sess = ort.InferenceSession("models/model_int8_static.onnx", providers=["CPUExecutionProvider"])

tile = tifffile.imread("path/to/tile.tif").astype(np.float32)        # (64, 64, 13) EuroSAT tile
x = tile[:, :, BANDS].transpose(2, 0, 1)[None] / 10000.0              # (1, 6, 64, 64)
logits = sess.run(None, {"input": x.astype(np.float32)})[0][0]

print(CLASSES[int(logits.argmax())])
```

Or with the script in this repository (prints the top-3 classes):

```bash
python infer.py --weights models/model_int8_static.onnx path/to/tile.tif
```

## Reproduce from scratch

The model was produced with the scripts in this repository:

```bash
pip install -r requirements.txt
wget https://madm.dfki.de/files/sentinel/EuroSATallBands.zip && unzip -q EuroSATallBands.zip -d data/EuroSATallBands
D=data/EuroSATallBands/ds/images/remote_sensing/otherDatasets/sentinel_2/tif
S=data/eurosat_split_indices.json

python train.py --data-dir $D --split-file $S --output-dir outputs                  # PyTorch model
python export_onnx.py --weights outputs/best_model.pth --output outputs/model_fp32.onnx
python quantize.py --onnx outputs/model_fp32.onnx --data-dir $D --split-file $S --mode static --output-dir outputs
python benchmark_onnx.py --models outputs/model_fp32.onnx outputs/model_int8_static.onnx \
       --data-dir $D --split-file $S --threads 1
```

Retraining gives numbers close to, not identical to, those above (GPU non-determinism, augmentation randomness).

## Limitations

* EuroSAT patches are clean, centre-cropped scenes of a single class. Expect lower accuracy on real satellite passes (clouds, mixed scenes, other sensors).
* Trained on 64 x 64 tiles only.
* INT8 accuracy depends on the calibration data; do not re-quantise without re-measuring on a held-out split.

## Repository contents

```
models/model_int8_static.onnx    the published model
dataset.py, model.py, train.py   training pipeline (PyTorch)
export_onnx.py, quantize.py      ONNX export and INT8 quantisation
benchmark_onnx.py                accuracy / size / latency comparison
infer.py                         inference for .pth and .onnx models
eurosat_6band_mobilenetv3.ipynb  end-to-end notebook
data/eurosat_split_indices.json  fixed train / val / test split
docs/                            technical report
```

## License and citation

Released under the MIT License (see `LICENSE`). EuroSAT and the torchvision ImageNet weights used for initialisation have their own terms.

> P. Helber, B. Bischke, A. Dengel, D. Borth. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS, 2019.
