#!/usr/bin/env python3
"""Train a tiny BiLSTM boundary tagger and export ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("himotoki_split/models/default.onnx"),
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="Optional cap for smoke runs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("output/neural_train_metrics.json"),
    )
    args = ap.parse_args()

    from himotoki_split.labels import boundary_f1, segments_to_bio
    from himotoki_split.model import load_jsonl_dataset
    from himotoki_split.neural import OnnxBoundaryModel, train_bilstm_and_export

    texts, segs = load_jsonl_dataset(args.input)
    if args.limit > 0:
        texts, segs = texts[: args.limit], segs[: args.limit]
    if not texts:
        print("No training examples", file=sys.stderr)
        return 1

    info = train_bilstm_and_export(
        texts,
        segs,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    model = OnnxBoundaryModel.load(args.output)
    f1s = []
    exact = 0
    for text, gold_segs in zip(texts, segs):
        gold = segments_to_bio(text, gold_segs)
        pred = model.predict(text)
        f1s.append(boundary_f1(gold, pred.labels)[2])
        if pred.segments == list(gold_segs):
            exact += 1
    metrics = {
        **info,
        "train_boundary_f1": sum(f1s) / len(f1s),
        "train_exact_seg": exact / len(texts),
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
