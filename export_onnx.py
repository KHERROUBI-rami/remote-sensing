"""Export a trained ``best_model.pth`` to ONNX and verify it against PyTorch.

Example
-------
    python export_onnx.py --weights outputs/best_model.pth --output outputs/model_fp32.onnx

The exported graph takes a float32 tensor ``input`` of shape ``(batch, 6, 64, 64)``
(batch size is dynamic) and returns raw class logits ``output`` of shape
``(batch, 10)``.  Normalisation (``/ 10000``) is NOT part of the graph - feed it
the same pre-processed tensors used for training (see ``dataset.read_6band_tile``).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch

from dataset import CLASS_NAMES, SELECTED_BANDS
from model import load_checkpoint


def export(model: torch.nn.Module, output: str, opset: int = 17, tile_size: int = 64) -> None:
    """Export ``model`` (eval mode, CPU) to a single-file ONNX graph."""
    model = model.cpu().eval()
    dummy = torch.randn(1, len(SELECTED_BANDS), tile_size, tile_size)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    tmp = output + ".tmp"

    kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    with torch.no_grad():
        try:
            # Legacy TorchScript exporter: simplest graph, best ORT-quantisation support.
            # (Deprecated since PyTorch 2.9; kept because it is the most robust path today.)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                torch.onnx.export(model, dummy, tmp, dynamo=False, **kwargs)
        except TypeError:
            # Older torch without the ``dynamo`` argument already uses the legacy exporter.
            torch.onnx.export(model, dummy, tmp, **kwargs)
        except Exception as err:  # legacy exporter removed in a future torch release
            print(f"Legacy exporter failed ({type(err).__name__}); retrying with the torch.export exporter.")
            torch.onnx.export(model, (dummy,), tmp, dynamo=True,
                              input_names=["input"], output_names=["output"],
                              dynamic_shapes={"x": {0: "batch"}}, opset_version=opset)

    # Re-save as one self-contained file (no external ``.data`` sidecar).
    proto = onnx.load(tmp)
    onnx.checker.check_model(proto)
    onnx.save(proto, output, save_as_external_data=False)
    os.remove(tmp)
    sidecar = tmp + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)


def verify(model: torch.nn.Module, onnx_path: str, n: int = 16, tile_size: int = 64,
           atol: float = 1e-4) -> None:
    """Check that ONNX Runtime reproduces the PyTorch logits."""
    model = model.cpu().eval()
    # Realistic reflectance-like inputs (training data is scaled to roughly 0..0.5).
    x = torch.rand(n, len(SELECTED_BANDS), tile_size, tile_size) * 0.4
    with torch.no_grad():
        ref = model(x).numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]

    max_diff = float(np.abs(ref - out).max())
    agree = float((ref.argmax(1) == out.argmax(1)).mean()) * 100
    print(f"Verification on {n} random tiles: max |logit diff| = {max_diff:.2e}, "
          f"top-1 agreement = {agree:.1f}%")
    if max_diff > atol:
        raise RuntimeError(f"ONNX output deviates from PyTorch by {max_diff:.2e} (> {atol:.0e}).")

    # Dynamic batch axis really works?
    single = sess.run(None, {sess.get_inputs()[0].name: x[:1].numpy()})[0]
    assert single.shape == (1, len(CLASS_NAMES)), single.shape


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Export best_model.pth to ONNX.")
    p.add_argument("--weights", default="outputs/best_model.pth", help="Trained PyTorch state_dict.")
    p.add_argument("--output", default="outputs/model_fp32.onnx", help="Destination .onnx file.")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--tile-size", type=int, default=64, help="Spatial size of the dummy input.")
    p.add_argument("--no-verify", action="store_true", help="Skip the PyTorch-vs-ONNX parity check.")
    args = p.parse_args(argv)

    model = load_checkpoint(args.weights, "cpu", num_bands=len(SELECTED_BANDS),
                            num_classes=len(CLASS_NAMES))
    export(model, args.output, args.opset, args.tile_size)
    size_mb = os.path.getsize(args.output) / 1024 ** 2
    print(f"Saved {args.output} ({size_mb:.2f} MB, opset {args.opset})")
    if not args.no_verify:
        verify(model, args.output, tile_size=args.tile_size)


if __name__ == "__main__":
    main()
