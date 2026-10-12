"""Quantise / convert the exported FP32 ONNX model for edge deployment.

Modes
-----
static   INT8 post-training quantisation (QDQ format, per-channel weights) using
         a class-balanced calibration set drawn from the *training* split.
         This is the main edge-deployment option (smallest and fastest on INT8 hardware).
dynamic  INT8 weights, activations quantised on the fly. No calibration data needed.
fp16     Weights stored in float16 (inputs/outputs stay float32). About half the size;
         usually *not* faster on CPU.

Example
-------
    python quantize.py --onnx outputs/model_fp32.onnx --data-dir <tif dir> \
        --split-file split.json --mode static --output-dir outputs

Always compare the result with ``benchmark_onnx.py`` on the held-out test split:
post-training quantisation of MobileNetV3 (Hardswish + squeeze-excite blocks) can
lose a lot of accuracy, and the calibration method matters.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from typing import List, Optional

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quant_pre_process,
                                      quantize_dynamic, quantize_static)

from dataset import EuroSat6BandDataset


class BalancedCalibrationReader(CalibrationDataReader):
    """Feeds ``samples_per_class`` un-augmented training tiles per class, one tile at a time."""

    def __init__(self, dataset: EuroSat6BandDataset, input_name: str,
                 samples_per_class: int = 50, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        labels = np.asarray(dataset.labels)
        chosen: List[int] = []
        for cls in np.unique(labels):
            idx = np.flatnonzero(labels == cls)
            chosen.extend(rng.choice(idx, size=min(samples_per_class, len(idx)), replace=False).tolist())
        rng.shuffle(chosen)

        self.input_name = input_name
        self.batches = [dataset[i][0].unsqueeze(0).numpy().astype(np.float32) for i in chosen]
        self.num_samples = len(self.batches)
        self._iter = iter(self.batches)

    def get_next(self):
        batch = next(self._iter, None)
        return None if batch is None else {self.input_name: batch}

    def rewind(self) -> None:
        self._iter = iter(self.batches)


def preprocess(src: str, workdir: str) -> str:
    """ONNX Runtime's recommended pre-processing (shape inference / graph optimisation)."""
    dst = os.path.join(workdir, "preprocessed.onnx")
    try:
        quant_pre_process(src, dst, skip_symbolic_shape=True)
        return dst
    except Exception as err:  # pre-processing is optional
        print(f"Pre-processing skipped ({type(err).__name__}: {err})")
        return src


def head_nodes(model_path: str) -> List[str]:
    """Names of the final classifier layers (Gemm/MatMul), used with --keep-head-fp32."""
    graph = onnx.load(model_path).graph
    return [n.name for n in graph.node if n.op_type in {"Gemm", "MatMul"}][-1:]


def quantize_static_int8(src: str, dst: str, data_dir: str, split_file: str,
                         samples_per_class: int, method: str, activation: str,
                         per_channel: bool, keep_head_fp32: bool, seed: int) -> None:
    train_ds = EuroSat6BandDataset(data_dir, split="train", split_file=split_file,
                                   augment=False, seed=seed)
    with tempfile.TemporaryDirectory() as tmp:
        prepared = preprocess(src, tmp)
        input_name = onnx.load(prepared).graph.input[0].name
        reader = BalancedCalibrationReader(train_ds, input_name, samples_per_class, seed)
        print(f"Calibrating with {reader.num_samples} training tiles ({method} method)...")

        exclude = head_nodes(prepared) if keep_head_fp32 else []
        quantize_static(
            model_input=prepared,
            model_output=dst,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8 if activation == "uint8" else QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            calibrate_method={"minmax": CalibrationMethod.MinMax,
                              "percentile": CalibrationMethod.Percentile,
                              "entropy": CalibrationMethod.Entropy}[method],
            nodes_to_exclude=exclude,
        )


def quantize_dynamic_int8(src: str, dst: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        prepared = preprocess(src, tmp)
        quantize_dynamic(prepared, dst, weight_type=QuantType.QUInt8)


def convert_fp16(src: str, dst: str) -> None:
    try:
        from onnxconverter_common import float16
    except ImportError as err:
        raise SystemExit("FP16 conversion needs `pip install onnxconverter-common`.") from err
    model = float16.convert_float_to_float16(onnx.load(src), keep_io_types=True)
    onnx.save(model, dst)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quantise the exported EuroSAT ONNX model.")
    p.add_argument("--onnx", default="outputs/model_fp32.onnx", help="FP32 ONNX produced by export_onnx.py.")
    p.add_argument("--mode", choices=["static", "dynamic", "fp16"], nargs="+", default=["static"],
                   help="One or more conversions to run.")
    p.add_argument("--output-dir", default="outputs")
    # static-mode options
    p.add_argument("--data-dir", help="EuroSAT tif folder (required for static mode).")
    p.add_argument("--split-file", default="split.json")
    p.add_argument("--calib-per-class", type=int, default=50,
                   help="Calibration tiles per class (10 classes -> 500 tiles by default).")
    p.add_argument("--calib-method", choices=["minmax", "percentile", "entropy"], default="percentile",
                   help="Activation-range estimator. Percentile/entropy are more robust to outliers.")
    p.add_argument("--activation-type", choices=["uint8", "int8"], default="uint8",
                   help="uint8 (U8S8) suits x86 CPUs; try int8 (S8S8) for ARM targets.")
    p.add_argument("--per-tensor", action="store_true", help="Per-tensor instead of per-channel weights.")
    p.add_argument("--keep-head-fp32", action="store_true",
                   help="Leave the final classifier layer in float (often recovers accuracy).")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not os.path.exists(args.onnx):
        raise SystemExit(f"{args.onnx} not found - run export_onnx.py first.")
    os.makedirs(args.output_dir, exist_ok=True)

    for mode in args.mode:
        if mode == "static":
            if not args.data_dir:
                raise SystemExit("--data-dir is required for static quantisation (calibration data).")
            dst = os.path.join(args.output_dir, "model_int8_static.onnx")
            quantize_static_int8(args.onnx, dst, args.data_dir, args.split_file, args.calib_per_class,
                                 args.calib_method, args.activation_type, not args.per_tensor,
                                 args.keep_head_fp32, args.seed)
        elif mode == "dynamic":
            dst = os.path.join(args.output_dir, "model_int8_dynamic.onnx")
            quantize_dynamic_int8(args.onnx, dst)
        else:
            dst = os.path.join(args.output_dir, "model_fp16.onnx")
            convert_fp16(args.onnx, dst)

        src_mb = os.path.getsize(args.onnx) / 1024 ** 2
        dst_mb = os.path.getsize(dst) / 1024 ** 2
        print(f"[{mode}] {dst}: {dst_mb:.2f} MB  (FP32 {src_mb:.2f} MB, {src_mb / dst_mb:.1f}x smaller)")

    print("\nNext: python benchmark_onnx.py --models ... to measure accuracy and latency on the test split.")


if __name__ == "__main__":
    main()
