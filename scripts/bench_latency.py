#!/usr/bin/env python3
"""Latency / throughput microbench for himotoki-split backends."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


SAMPLES = [
    "猫が食べる",
    "今日は天気がいい",
    "学校で勉強しています",
    "彼女は東京大学で日本語を勉強している。",
    "これらの問題を解決するためには、より詳細な分析が必要である。",
]


def _bench(predict, texts, rounds: int, warmup: int) -> dict:
    for _ in range(warmup):
        for t in texts:
            predict(t)
    times = []
    n_chars = 0
    for _ in range(rounds):
        t0 = time.perf_counter()
        for t in texts:
            predict(t)
            n_chars += len(t)
        times.append(time.perf_counter() - t0)
    total_sents = rounds * len(texts)
    mean_batch = statistics.mean(times)
    return {
        "rounds": rounds,
        "sentences_per_round": len(texts),
        "total_sentences": total_sents,
        "total_chars": n_chars,
        "mean_batch_s": mean_batch,
        "median_batch_s": statistics.median(times),
        "sentences_per_s": total_sents / sum(times),
        "chars_per_s": n_chars / sum(times),
        "mean_ms_per_sentence": 1000.0 * sum(times) / total_sents,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/bench_latency.json"),
    )
    ap.add_argument(
        "--holdout-sample",
        type=Path,
        default=Path("data/labels/holdout_clean.jsonl"),
        help="Optional JSONL; uses first 100 texts if present",
    )
    args = ap.parse_args()

    texts = list(SAMPLES)
    if args.holdout_sample.is_file():
        from himotoki_split.model import load_jsonl_dataset

        h_texts, _ = load_jsonl_dataset(args.holdout_sample)
        texts = h_texts[:100] or texts

    results = {"n_texts": len(texts), "backends": {}}

    # Linear
    from himotoki_split.model import load_default_model

    linear = load_default_model()
    results["backends"]["linear"] = _bench(
        linear.predict, texts, args.rounds, args.warmup
    )

    # ONNX if available
    onnx_path = Path("himotoki_split/models/default.onnx")
    if onnx_path.is_file():
        try:
            from himotoki_split.neural import OnnxBoundaryModel

            onnx = OnnxBoundaryModel.load(onnx_path)
            results["backends"]["onnx"] = _bench(
                onnx.predict, texts, args.rounds, args.warmup
            )
        except Exception as exc:
            results["backends"]["onnx_error"] = str(exc)

    # Public split() API (may prefer ONNX)
    from himotoki_split.split import reset_model_cache, split

    reset_model_cache()
    results["backends"]["split_api"] = _bench(
        lambda t: split(t, fallback=False), texts, args.rounds, args.warmup
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
