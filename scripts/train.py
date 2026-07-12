#!/usr/bin/env python3
"""Train a boundary model from JSONL silver labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="JSONL from scripts/dump_labels.py",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .npz model path",
    )
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=400)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap (0 = all)",
    )
    ap.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("output/train_metrics.json"),
    )
    args = ap.parse_args()

    from himotoki_split.model import load_jsonl_dataset, train_boundary_model

    texts, segs = load_jsonl_dataset(args.input)
    if args.limit > 0:
        texts, segs = texts[: args.limit], segs[: args.limit]
    if not texts:
        print("No training examples", file=sys.stderr)
        return 1

    print(f"Training linear SGD on {len(texts)} sentences (max_iter={args.max_iter})")
    model = train_boundary_model(texts, segs, C=args.C, max_iter=args.max_iter)
    model.save(args.output)

    metrics = {
        "model": str(args.output),
        "n_sentences": len(texts),
        "C": args.C,
        "max_iter": args.max_iter,
        "meta": model.meta,
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
