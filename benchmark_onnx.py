"""Compare ONNX variants (FP32 / INT8 / FP16) on the fixed test split.

Reports, per model: file size, test accuracy, weighted F1, accuracy drop versus the
first model listed, and single-tile CPU latency.  Always evaluates on the held-out
test split from ``split.json`` (never on training data).

Example
-------
    python benchmark_onnx.py \
        --models outputs/model_fp32.onnx outputs/model_int8_static.onnx \
        --data-dir <tif dir> --split-file split.json --threads 1 --save outputs/benchmark.md
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import onnxruntime as ort
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import EuroSat6BandDataset


def make_session(path: str, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def input_dtype(sess: ort.InferenceSession):
    return np.float16 if "float16" in sess.get_inputs()[0].type else np.float32


def evaluate(sess: ort.InferenceSession, loader: DataLoader) -> Dict[str, float]:
    name, dtype = sess.get_inputs()[0].name, input_dtype(sess)
    preds: List[int] = []
    labels: List[int] = []
    for x, y in loader:
        logits = sess.run(None, {name: x.numpy().astype(dtype)})[0]
        preds.extend(np.argmax(logits, axis=1).tolist())
        labels.extend(y.tolist())
    preds, labels = np.asarray(preds), np.asarray(labels)
    return {
        "acc": 100.0 * float((preds == labels).mean()),
        "f1": 100.0 * float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "n": int(len(labels)),
    }


def latency_ms(sess: ort.InferenceSession, runs: int, tile: int = 64, bands: int = 6) -> Dict[str, float]:
    """Batch-1 latency with warm-up; returns median and 95th percentile in milliseconds."""
    name, dtype = sess.get_inputs()[0].name, input_dtype(sess)
    x = (np.random.rand(1, bands, tile, tile) * 0.4).astype(dtype)
    for _ in range(max(10, runs // 10)):
        sess.run(None, {name: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        times.append((time.perf_counter() - t0) * 1000)
    return {"median": float(np.median(times)), "p95": float(np.percentile(times, 95))}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Benchmark ONNX models on the EuroSAT test split.")
    p.add_argument("--models", nargs="+", required=True, help="ONNX files; the first is the reference.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--split-file", default="split.json")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--max-samples", type=int, default=0, help="Limit evaluated tiles (0 = whole split).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threads", type=int, default=1, help="CPU threads (1 mimics a small edge CPU).")
    p.add_argument("--latency-runs", type=int, default=200)
    p.add_argument("--save", default=None, help="Write the results table (.md) and a .json next to it.")
    args = p.parse_args(argv)

    ds = EuroSat6BandDataset(args.data_dir, split=args.split, split_file=args.split_file, augment=False)
    if args.max_samples:
        ds.image_paths, ds.labels = ds.image_paths[:args.max_samples], ds.labels[:args.max_samples]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(ds)} '{args.split}' tiles, {args.threads} CPU thread(s)\n")

    rows = []
    for path in args.models:
        sess = make_session(path, args.threads)
        m = evaluate(sess, loader)
        lat = latency_ms(sess, args.latency_runs)
        rows.append({"model": os.path.basename(path), "size_mb": os.path.getsize(path) / 1024 ** 2,
                     **m, "lat_median_ms": lat["median"], "lat_p95_ms": lat["p95"]})
        print(f"{rows[-1]['model']}: acc {m['acc']:.2f}%  F1 {m['f1']:.2f}%  "
              f"{lat['median']:.2f} ms/tile")

    ref = rows[0]
    lines = ["| Model | Size (MB) | Accuracy (%) | F1 weighted (%) | Δ acc vs ref | Latency median / p95 (ms) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['size_mb']:.2f} | {r['acc']:.2f} | {r['f1']:.2f} | "
                     f"{r['acc'] - ref['acc']:+.2f} | {r['lat_median_ms']:.2f} / {r['lat_p95_ms']:.2f} |")
    table = "\n".join(lines)
    print("\n" + table)
    print(f"\n({ref['n']} {args.split} tiles; batch-1 latency, {args.threads} thread(s), "
          f"onnxruntime {ort.__version__})")

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        with open(args.save, "w") as f:
            f.write(table + "\n")
        with open(os.path.splitext(args.save)[0] + ".json", "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
