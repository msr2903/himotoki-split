#!/usr/bin/env python3
"""Dump Himotoki silver labels to JSONL for distillation.

Supports sharding and resume for large corpora.

Examples:
  python scripts/dump_labels.py -i data/corpus/sentences.txt -o data/labels/raw.jsonl
  python scripts/dump_labels.py -i data/corpus/sentences.txt -o data/labels/shards/part-000.jsonl \\
      --shard-id 0 --num-shards 8 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Set


def _load_done(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if text:
                done.add(text)
    return done


def _select_shard(lines: list[str], shard_id: int, num_shards: int) -> list[str]:
    if num_shards <= 1:
        return lines
    if not (0 <= shard_id < num_shards):
        raise SystemExit(f"--shard-id must be in [0, {num_shards})")
    return [ln for i, ln in enumerate(lines) if i % num_shards == shard_id]


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
        help="Optional max sentences after sharding (0 = all)",
    )
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip texts already present in the output JSONL",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log progress every N newly labeled sentences",
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
    lines = _select_shard(lines, args.shard_id, args.num_shards)
    if args.limit > 0:
        lines = lines[: args.limit]

    done = _load_done(args.output) if args.resume else set()
    todo = [ln for ln in lines if ln not in done]
    print(
        f"shard={args.shard_id}/{args.num_shards} "
        f"input={len(lines)} done={len(done)} todo={len(todo)}"
    )
    if not todo:
        print("Nothing to do")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Label in chunks so resume is useful and progress is visible.
    # dump_labels warms Himotoki once per call; call once with streaming instead.
    import himotoki
    from himotoki_split.fallback import split_with_himotoki

    himotoki.warm_up()
    t0 = time.perf_counter()
    written = 0
    errors = 0
    mode = "a" if args.resume and args.output.is_file() else "w"
    with args.output.open(mode, encoding="utf-8") as f:
        for i, sentence in enumerate(todo):
            try:
                segs = split_with_himotoki(sentence)
                if "".join(segs) != sentence:
                    # Keep only consistent reconstructions
                    errors += 1
                    continue
                row = {
                    "text": sentence,
                    "segments": segs,
                    "teacher": "himotoki",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    print(f"warn: failed on {sentence[:40]!r}: {exc}", file=sys.stderr)
            if args.progress_every and written and written % args.progress_every == 0:
                elapsed = time.perf_counter() - t0
                rate = written / elapsed if elapsed > 0 else 0
                remain = len(todo) - (i + 1)
                eta = remain / rate if rate > 0 else 0
                print(
                    f"  labeled={written} errors={errors} "
                    f"{rate:.1f} sent/s ETA={eta/60:.1f}m",
                    flush=True,
                )
                f.flush()

    elapsed = time.perf_counter() - t0
    print(
        f"Wrote {written} new examples to {args.output} "
        f"({errors} errors, {elapsed:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
