"""CLI for himotoki-split."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="himotoki-split",
        description="Lightweight Japanese split-only tokenizer",
    )
    parser.add_argument("text", nargs="?", help="Text to split")
    parser.add_argument(
        "-m",
        "--model",
        help="Path to .npz model (default: packaged model)",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Never call Himotoki on low confidence",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.96,
        help="Fallback threshold (default: 0.96, Phase B calibrated)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of pipe-separated segments",
    )
    args = parser.parse_args(argv)

    text = args.text
    if text is None:
        text = sys.stdin.read().rstrip("\n")

    from himotoki_split.split import split

    result = split(
        text,
        model_path=args.model,
        fallback=not args.no_fallback,
        min_confidence=args.min_confidence,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "text": result.text,
                    "segments": result.segments,
                    "confidence": result.confidence,
                    "source": result.source,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(" | ".join(result.segments))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
