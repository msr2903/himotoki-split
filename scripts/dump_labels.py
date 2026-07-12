#!/usr/bin/env python3
"""Dump Himotoki silver labels to JSONL for distillation."""

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
        help="Text file with one sentence per line",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max sentences (0 = all)",
    )
    args = ap.parse_args()

    try:
        from himotoki_split.fallback import dump_labels, himotoki_available
    except ImportError:
        print("Failed to import himotoki-split", file=sys.stderr)
        return 1

    if not himotoki_available():
        print(
            "Himotoki is not installed. pip install 'himotoki-split[teacher]'",
            file=sys.stderr,
        )
        return 1

    lines = [
        ln.strip()
        for ln in args.input.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if args.limit > 0:
        lines = lines[: args.limit]

    rows = dump_labels(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} examples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
