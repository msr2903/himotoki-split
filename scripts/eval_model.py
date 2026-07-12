#!/usr/bin/env python3
"""Evaluate a boundary model on a holdout JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-m",
        "--model",
        type=Path,
        default=Path("himotoki_split/models/default.npz"),
    )
    ap.add_argument(
        "-i",
        "--holdout",
        type=Path,
        default=Path("data/labels/holdout.jsonl"),
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/eval_metrics.json"),
    )
    ap.add_argument(
        "--backend",
        choices=("auto", "linear", "onnx"),
        default="auto",
        help="Which student backend to evaluate",
    )
    args = ap.parse_args()

    from himotoki_split.labels import boundary_f1, segments_to_bio
    from himotoki_split.model import load_jsonl_dataset, load_model

    if not args.holdout.is_file():
        print(f"Missing holdout: {args.holdout}", file=sys.stderr)
        return 1

    texts, segs = load_jsonl_dataset(args.holdout)
    if not texts:
        print("Empty holdout", file=sys.stderr)
        return 1

    predict = None
    backend_used = args.backend

    if args.backend in ("auto", "onnx") and str(args.model).endswith(".onnx"):
        from himotoki_split.neural import OnnxBoundaryModel

        model = OnnxBoundaryModel.load(args.model)
        predict = model.predict
        backend_used = "onnx"
    elif args.backend == "onnx":
        from himotoki_split.neural import OnnxBoundaryModel

        model = OnnxBoundaryModel.load(args.model)
        predict = model.predict
        backend_used = "onnx"
    else:
        model = load_model(args.model)
        predict = model.predict
        backend_used = "linear"

    f1s = []
    exact = 0
    confs = []
    for text, gold_segs in zip(texts, segs):
        gold = segments_to_bio(text, gold_segs)
        pred = predict(text)
        f1s.append(boundary_f1(gold, pred.labels)[2])
        confs.append(pred.confidence)
        if pred.segments == list(gold_segs):
            exact += 1

    metrics = {
        "n": len(texts),
        "backend": backend_used,
        "model": str(args.model),
        "holdout": str(args.holdout),
        "boundary_f1": sum(f1s) / len(f1s),
        "exact_seg": exact / len(texts),
        "mean_confidence": sum(confs) / len(confs),
        "exact_count": exact,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
