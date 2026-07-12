#!/usr/bin/env python3
"""Train a boundary model from JSONL silver labels."""

from __future__ import annotations

import argparse
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
    ap.add_argument("--max-iter", type=int, default=300)
    args = ap.parse_args()

    from himotoki_split.labels import boundary_f1, segments_to_bio
    from himotoki_split.model import load_jsonl_dataset, train_boundary_model

    texts, segs = load_jsonl_dataset(args.input)
    if not texts:
        print("No training examples", file=sys.stderr)
        return 1

    model = train_boundary_model(texts, segs, C=args.C, max_iter=args.max_iter)
    model.save(args.output)

    # Quick train-set boundary F1
    tp = []
    for text, gold_segs in zip(texts, segs):
        gold = segments_to_bio(text, gold_segs)
        pred = model.predict(text)
        tp.append(boundary_f1(gold, pred.labels))
    p = sum(x[0] for x in tp) / len(tp)
    r = sum(x[1] for x in tp) / len(tp)
    f = sum(x[2] for x in tp) / len(tp)
    print(f"Saved {args.output}")
    print(f"Train boundary P/R/F1: {p:.3f}/{r:.3f}/{f:.3f} on {len(texts)} sents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
