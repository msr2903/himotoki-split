#!/usr/bin/env python3
"""Merge shard JSONL label files into train + frozen holdout splits."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, List


def iter_jsonl(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="Input JSONL files (shards or monolithic)",
    )
    ap.add_argument(
        "--train-out",
        type=Path,
        default=Path("data/labels/train.jsonl"),
    )
    ap.add_argument(
        "--holdout-out",
        type=Path,
        default=Path("data/labels/holdout.jsonl"),
    )
    ap.add_argument("--holdout-size", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    by_text = {}
    skipped = 0
    for row in iter_jsonl(args.inputs):
        text = row.get("text")
        segs = row.get("segments")
        if not text or not isinstance(segs, list) or not segs:
            skipped += 1
            continue
        if "".join(segs) != text:
            skipped += 1
            continue
        by_text[text] = {
            "text": text,
            "segments": segs,
            "teacher": row.get("teacher", "himotoki"),
        }

    rows: List[dict] = list(by_text.values())
    if not rows:
        print("No valid rows", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    holdout_n = min(args.holdout_size, max(0, len(rows) // 10), len(rows))
    # Prefer exactly holdout_size when enough data
    if len(rows) > args.holdout_size:
        holdout_n = args.holdout_size
    holdout = rows[:holdout_n]
    train = rows[holdout_n:]

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.holdout_out.parent.mkdir(parents=True, exist_ok=True)

    with args.train_out.open("w", encoding="utf-8") as f:
        for row in train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.holdout_out.open("w", encoding="utf-8") as f:
        for row in holdout:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"unique={len(rows)} skipped={skipped} "
        f"train={len(train)} -> {args.train_out} "
        f"holdout={len(holdout)} -> {args.holdout_out} (seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
