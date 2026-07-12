#!/usr/bin/env python3
"""Active-query training loop for himotoki-split.

1. Score a sample of *train* sentences for uncertainty (not holdout).
2. Take the top-K most uncertain.
3. Label with Himotoki teacher (oracle) — same distribution as silver data.
4. Write feedback JSONL + build an upsampled augmented train set.
5. Optionally retrain ONNX and eval (caller can do that, or --train).

Usage:
  export HIMOTOKI_DB_PATH=~/.himotoki/himotoki.db
  python scripts/active_query_train.py --query-k 500 --upsample 15 --train
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _uncertainty(model, text: str) -> Dict[str, Any]:
    from himotoki_split.labels import LABEL_B

    pred = model.predict(text)
    probs = model.predict_proba_b(text)
    ents = []
    for p in probs:
        p = float(min(max(p, 1e-6), 1 - 1e-6))
        ents.append(-(p * math.log(p) + (1 - p) * math.log(1 - p)))
    mean_entropy = sum(ents) / len(ents) if ents else 0.0
    score = (1.0 - float(pred.confidence)) + 0.5 * mean_entropy
    return {
        "text": text,
        "model_segments": list(pred.segments),
        "model_confidence": float(pred.confidence),
        "mean_entropy": mean_entropy,
        "query_score": score,
        "labels": list(pred.labels),
    }


def _score_pool(
    model,
    texts: Sequence[str],
    *,
    sample_n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pool = list(texts)
    if sample_n > 0 and len(pool) > sample_n:
        pool = rng.sample(pool, sample_n)
    scored = []
    for i, text in enumerate(pool):
        if not text:
            continue
        scored.append(_uncertainty(model, text))
        if (i + 1) % 500 == 0:
            print(f"  scored {i+1}/{len(pool)}", flush=True)
    scored.sort(key=lambda r: r["query_score"], reverse=True)
    return scored


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train-labels",
        type=Path,
        default=Path("data/labels/train.jsonl"),
    )
    ap.add_argument(
        "--holdout",
        type=Path,
        default=Path("data/labels/holdout.jsonl"),
        help="Excluded from query pool (frozen eval)",
    )
    ap.add_argument(
        "--feedback-out",
        type=Path,
        default=Path("demo/data/feedback_active.jsonl"),
    )
    ap.add_argument(
        "--augmented-out",
        type=Path,
        default=Path("data/labels/train_active.jsonl"),
    )
    ap.add_argument("--sample-n", type=int, default=8000)
    ap.add_argument("--query-k", type=int, default=500)
    ap.add_argument("--upsample", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--model",
        type=Path,
        default=Path("himotoki_split/models/default.onnx"),
    )
    ap.add_argument(
        "--train",
        action="store_true",
        help="Retrain ONNX on augmented labels after querying",
    )
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--onnx-out",
        type=Path,
        default=Path("himotoki_split/models/default.onnx"),
    )
    ap.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("output/active_query_metrics.json"),
    )
    args = ap.parse_args()

    from himotoki_split.fallback import himotoki_available, split_with_himotoki
    from himotoki_split.neural import OnnxBoundaryModel
    from himotoki_split.split import reset_model_cache

    if not himotoki_available():
        print("Himotoki teacher required (set HIMOTOKI_DB_PATH)", file=sys.stderr)
        return 1
    if not args.train_labels.is_file():
        print(f"Missing train labels: {args.train_labels}", file=sys.stderr)
        return 1

    holdout_texts = set()
    if args.holdout.is_file():
        for row in _load_jsonl(args.holdout):
            t = row.get("text")
            if t:
                holdout_texts.add(t)

    train_rows = _load_jsonl(args.train_labels)
    train_by_text = {}
    for row in train_rows:
        t = row.get("text")
        segs = row.get("segments")
        if t and isinstance(segs, list) and "".join(segs) == t:
            if t not in holdout_texts:
                train_by_text[t] = segs

    texts = list(train_by_text.keys())
    print(
        f"Train pool: {len(texts)} (excluded {len(holdout_texts)} holdout texts)"
    )

    reset_model_cache()
    model = OnnxBoundaryModel.load(args.model)
    print(f"Scoring sample_n={args.sample_n} …")
    scored = _score_pool(
        model, texts, sample_n=args.sample_n, seed=args.seed
    )
    top = scored[: args.query_k]
    print(f"Querying top {len(top)} uncertain examples with Himotoki oracle")

    feedback: List[Dict[str, Any]] = []
    n_correct = 0
    n_accept = 0
    n_fail = 0
    for i, item in enumerate(top):
        text = item["text"]
        try:
            teacher = split_with_himotoki(text)
        except Exception as exc:
            n_fail += 1
            print(f"  teacher fail [{i}]: {exc}", flush=True)
            continue
        if "".join(teacher) != text:
            n_fail += 1
            continue
        if teacher == item["model_segments"]:
            action = "accept"
            segs = item["model_segments"]
            n_accept += 1
        else:
            action = "correct"
            segs = teacher
            n_correct += 1
        feedback.append(
            {
                "ts": _utc(),
                "text": text,
                "action": action,
                "model_segments": item["model_segments"],
                "model_confidence": item["model_confidence"],
                "query_score": item["query_score"],
                "mean_entropy": item["mean_entropy"],
                "segments": segs,
                "teacher": "himotoki",
                "source": "active_query",
            }
        )
        if (i + 1) % 50 == 0:
            print(
                f"  labeled {i+1}/{len(top)} "
                f"(correct={n_correct} accept={n_accept} fail={n_fail})",
                flush=True,
            )

    args.feedback_out.parent.mkdir(parents=True, exist_ok=True)
    with args.feedback_out.open("w", encoding="utf-8") as f:
        for row in feedback:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {args.feedback_out} ({len(feedback)} rows)")

    # Build augmented train: original + upsampled corrections (+ mild accept)
    aug: List[Dict[str, Any]] = []
    for text, segs in train_by_text.items():
        aug.append(
            {"text": text, "segments": segs, "teacher": "himotoki", "via": "silver"}
        )

    added = 0
    for row in feedback:
        segs = row["segments"]
        text = row["text"]
        copies = args.upsample if row["action"] == "correct" else max(1, args.upsample // 5)
        for _ in range(copies):
            aug.append(
                {
                    "text": text,
                    "segments": segs,
                    "teacher": "himotoki",
                    "via": f"active_{row['action']}",
                }
            )
            added += 1

    rng = random.Random(args.seed)
    rng.shuffle(aug)
    args.augmented_out.parent.mkdir(parents=True, exist_ok=True)
    with args.augmented_out.open("w", encoding="utf-8") as f:
        for row in aug:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"Wrote {args.augmented_out}: {len(aug)} rows "
        f"(+{added} upsampled active examples)"
    )

    summary: Dict[str, Any] = {
        "sample_n": args.sample_n,
        "query_k": args.query_k,
        "scored": len(scored),
        "feedback": len(feedback),
        "n_correct": n_correct,
        "n_accept": n_accept,
        "n_fail": n_fail,
        "upsample": args.upsample,
        "augmented_n": len(aug),
        "feedback_path": str(args.feedback_out),
        "augmented_path": str(args.augmented_out),
        "top_query_scores": [round(x["query_score"], 4) for x in top[:10]],
        "mean_top_confidence": (
            sum(x["model_confidence"] for x in top) / len(top) if top else 0
        ),
    }

    if args.train:
        print(
            f"Retraining ONNX on augmented set ({len(aug)} rows, "
            f"epochs={args.epochs})…"
        )
        from himotoki_split.model import load_jsonl_dataset
        from himotoki_split.neural import train_bilstm_and_export

        texts_t, segs_t = load_jsonl_dataset(args.augmented_out)
        # val slice from end
        val_n = min(2000, max(0, len(texts_t) // 20))
        val_texts = texts_t[-val_n:] if val_n else None
        val_segs = segs_t[-val_n:] if val_n else None
        info = train_bilstm_and_export(
            texts_t,
            segs_t,
            args.onnx_out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device="cpu",
            val_texts=val_texts,
            val_segments=val_segs,
            max_len=256,
        )
        summary["train"] = info
        reset_model_cache()
        print(f"Wrote {args.onnx_out}")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "train"}, indent=2))
    print(f"Wrote {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
