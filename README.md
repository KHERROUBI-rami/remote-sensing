# Model Execution & Technical Data Specifications

Classification model operating guide and technical data specifications.

## 1. Model Overview

| Item | Details |
|---|---|
| **Architecture** | MobileNetV3, optimized and quantized (TorchScript / INT8) |
| **Model file** | `static_int8_model.pt` (or the weights file `.pth`) |
| **Base training data** | EuroSAT dataset, based on the Sentinel-2 satellite |

## 2. Required Input Data Specifications

| Parameter | Value used in training | Technical notes |
|---|---|---|
| **Data source** | Sentinel-2 L2A | Surface reflectance readings |
| **Spectral bands** | 6 bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2) | Band order in code: `BANDS_SELECTED = [1, 2, 3, 7, 10, 11]` (indices follow the Sentinel-2 band array) |
| **Spatial dimensions** | 64 x 64 pixels | The code automatically resizes images of other sizes |
| **Normalization** | Divide by `10000.0` | Converts raw reflectance values to the range 0 to 1 |

## 3. Inference Instructions

1. **Set the paths:** update `PATH_MODEL` to the location of the weights file, and `path_test` to the path of the TIF image to be tested.
2. **Input:** the input image should preferably be TIF / GeoTIFF and contain the spectral bands listed above.

## 4. Technical Notice: Other Satellites (Cross-Satellite Domain Shift)

If you want to test the model on imagery from other satellites (such as Landsat, WorldView, or proprietary sensors), it will **not** give accurate results directly without fine-tuning, for the following reasons:

- **Spectral mismatch:** band wavelengths differ between satellites even when band names are similar.
- **Scaling factor mismatch:** not all satellites use the `10000.0` divisor for reflectance.
- **Spatial resolution mismatch:** a different pixel size affects the patterns the model has learned.

## 5. Fine-Tuning Requirements for a New Satellite

To apply the model to data from another satellite:

1. **Re-match the bands:** order the input bands by physical wavelength, not by channel number.
2. **Adjust normalization:** set the offset and divisor according to the value range of the new satellite.
3. **Fine-tune:** freeze the encoder layers and retrain the final classifier layers on a sample of the new satellite's data, using a small learning rate (e.g., 10^-4).

---

*This document accompanies the inference code and the model weights file.*
