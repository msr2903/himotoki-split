#!/usr/bin/env python3
"""Sweep min_confidence for hybrid model + Himotoki fallback."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-m",
        "--model",
        type=Path,
        default=Path("himotoki_split/models/default.onnx"),
    )
    ap.add_argument(
        "-i",
        "--holdout",
        type=Path,
        default=Path("data/labels/holdout.jsonl"),
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/fallback_calibration.json"),
    )
    ap.add_argument(
        "--thresholds",
        default="0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
        help="Comma-separated confidence thresholds",
    )
    ap.add_argument(
        "--max-teacher-rate",
        type=float,
        default=0.30,
        help="Prefer thresholds with Himotoki call rate <= this",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional sentence cap")
    args = ap.parse_args()

    from himotoki_split.fallback import himotoki_available, split_with_himotoki
    from himotoki_split.model import load_jsonl_dataset, load_model

    if not args.holdout.is_file():
        print(f"Missing holdout: {args.holdout}", file=sys.stderr)
        return 1

    texts, gold_segs = load_jsonl_dataset(args.holdout)
    if args.limit > 0:
        texts, gold_segs = texts[: args.limit], gold_segs[: args.limit]
    if not texts:
        print("Empty holdout", file=sys.stderr)
        return 1

    if str(args.model).endswith(".onnx"):
        from himotoki_split.neural import OnnxBoundaryModel

        model = OnnxBoundaryModel.load(args.model)
        backend = "onnx"
    else:
        model = load_model(args.model)
        backend = "linear"

    teacher_ok = himotoki_available()
    if not teacher_ok:
        print(
            "WARNING: Himotoki not available; hybrid exact-seg will equal model-only",
            file=sys.stderr,
        )

    # Precompute model predictions once
    preds = []
    for text in texts:
        pred = model.predict(text)
        preds.append(pred)

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    rows = []
    for t in thresholds:
        exact = 0
        teacher_calls = 0
        for text, gold, pred in zip(texts, gold_segs, preds):
            segs = pred.segments
            if pred.confidence < t and teacher_ok:
                teacher_calls += 1
                try:
                    segs = split_with_himotoki(text)
                except Exception:
                    segs = pred.segments
            if list(segs) == list(gold):
                exact += 1
        n = len(texts)
        rows.append(
            {
                "min_confidence": t,
                "exact_seg": exact / n,
                "exact_count": exact,
                "teacher_call_rate": teacher_calls / n,
                "teacher_calls": teacher_calls,
            }
        )

    # Model-only baseline
    model_exact = sum(
        1 for pred, gold in zip(preds, gold_segs) if list(pred.segments) == list(gold)
    ) / len(texts)

    eligible = [r for r in rows if r["teacher_call_rate"] <= args.max_teacher_rate]
    pool = eligible if eligible else rows
    best = max(pool, key=lambda r: (r["exact_seg"], -r["teacher_call_rate"]))

    out = {
        "backend": backend,
        "model": str(args.model),
        "holdout": str(args.holdout),
        "n": len(texts),
        "himotoki_available": teacher_ok,
        "max_teacher_rate": args.max_teacher_rate,
        "model_only_exact_seg": model_exact,
        "sweep": rows,
        "recommended_min_confidence": best["min_confidence"],
        "recommended": best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
