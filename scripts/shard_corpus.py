#!/usr/bin/env python3
"""Split a sentence file into shard text files for parallel dump_labels."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", type=Path, required=True)
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data/corpus/shards"),
    )
    ap.add_argument("--num-shards", type=int, default=8)
    args = ap.parse_args()

    lines = [
        ln.strip()
        for ln in args.input.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = [[] for _ in range(args.num_shards)]
    for i, ln in enumerate(lines):
        shards[i % args.num_shards].append(ln)
    for i, rows in enumerate(shards):
        path = args.output_dir / f"part-{i:03d}.txt"
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        print(f"{path}: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
