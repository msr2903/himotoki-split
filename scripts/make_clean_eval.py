#!/usr/bin/env python3
"""Build a cleaner holdout slice from noisy Wikipedia silver labels."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Markup / noise heuristics for wiki extract leftovers
_BAD_SUBSTR = (
    "]]",
    "[[",
    "{{",
    "}}",
    "&amp;",
    "&lt;",
    "&gt;",
    "&quot;",
    "http://",
    "https://",
    "Category:",
    "File:",
    "Image:",
    "|",
    "==",
)
_JP_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")
_ASCII_LETTER = re.compile(r"[A-Za-z]")


def jp_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_JP_RE.findall(text)) / len(text)


def is_clean(text: str, *, min_len: int, max_len: int, min_jp: float) -> bool:
    n = len(text)
    if n < min_len or n > max_len:
        return False
    if any(tok in text for tok in _BAD_SUBSTR):
        return False
    # Too many ASCII letters usually means romanization dump / markup
    if len(_ASCII_LETTER.findall(text)) > max(3, n // 10):
        return False
    if jp_ratio(text) < min_jp:
        return False
    # Unbalanced brackets / quotes often leftover wiki noise
    if text.count("「") != text.count("」"):
        return False
    if text.count("(") != text.count(")"):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/labels/holdout.jsonl"),
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/labels/holdout_clean.jsonl"),
    )
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--min-jp", type=float, default=0.55)
    ap.add_argument("--target", type=int, default=800)
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"Missing input: {args.input}", file=sys.stderr)
        return 1

    kept = []
    total = 0
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            text = row.get("text") or ""
            if not is_clean(
                text,
                min_len=args.min_len,
                max_len=args.max_len,
                min_jp=args.min_jp,
            ):
                continue
            kept.append(row)
            if args.target > 0 and len(kept) >= args.target:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in kept:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "scanned": total,
        "kept": len(kept),
        "min_len": args.min_len,
        "max_len": args.max_len,
        "min_jp": args.min_jp,
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
