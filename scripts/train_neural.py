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
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap for smoke runs (0 = all)",
    )
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument(
        "--val-size",
        type=int,
        default=2000,
        help="In-train validation slice size (from end of train set)",
    )
    ap.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Truncate char sequences during training (0 = no truncate)",
    )
    ap.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("output/neural_train_metrics.json"),
    )
    args = ap.parse_args()

    from himotoki_split.model import load_jsonl_dataset
    from himotoki_split.neural import train_bilstm_and_export

    texts, segs = load_jsonl_dataset(args.input)
    if args.limit > 0:
        texts, segs = texts[: args.limit], segs[: args.limit]
    if not texts:
        print("No training examples", file=sys.stderr)
        return 1

    val_texts = None
    val_segs = None
    train_texts, train_segs = texts, segs
    if args.val_size > 0 and len(texts) > args.val_size + 100:
        val_texts = texts[-args.val_size :]
        val_segs = segs[-args.val_size :]
        # Still train on full set including val slice (silver data);
        # val metrics are progress signals, not early-stop holdout.
        train_texts, train_segs = texts, segs

    print(
        f"Training BiLSTM on {len(train_texts)} sentences "
        f"(epochs={args.epochs}, batch={args.batch_size}, device={args.device})"
    )
    info = train_bilstm_and_export(
        train_texts,
        train_segs,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        val_texts=val_texts,
        val_segments=val_segs,
        max_len=args.max_len,
    )
    metrics = {
        **info,
        "limit": args.limit,
        "val_size": args.val_size if val_texts is not None else 0,
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "epoch_logs"}, indent=2))
    if metrics.get("epoch_logs"):
        last = metrics["epoch_logs"][-1]
        print(f"Final epoch: {json.dumps(last)}")
    print(f"Wrote {args.output} and {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
