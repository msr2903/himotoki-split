"""BIO label helpers for character-level word boundaries."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

# B = begin token, I = continuation
LABEL_B = 0
LABEL_I = 1
LABEL_NAMES = ("B", "I")


def segments_to_bio(text: str, segments: Sequence[str]) -> List[int]:
    """Convert surface segments into per-character B/I labels.

    Raises ValueError if segments do not exactly reconstitute ``text``.
    """
    joined = "".join(segments)
    if joined != text:
        raise ValueError(
            f"segments do not match text: {joined!r} != {text!r}"
        )
    labels: List[int] = []
    for seg in segments:
        if not seg:
            raise ValueError("empty segment is not allowed")
        labels.append(LABEL_B)
        labels.extend([LABEL_I] * (len(seg) - 1))
    if len(labels) != len(text):
        raise ValueError("label length mismatch")
    return labels


def bio_to_segments(text: str, labels: Sequence[int]) -> List[str]:
    """Decode B/I labels into surface segments."""
    if len(text) != len(labels):
        raise ValueError("text/label length mismatch")
    if not text:
        return []
    segments: List[str] = []
    start = 0
    for i in range(1, len(text)):
        if labels[i] == LABEL_B:
            segments.append(text[start:i])
            start = i
    segments.append(text[start:])
    return segments


def boundary_f1(
    gold_labels: Sequence[int], pred_labels: Sequence[int]
) -> Tuple[float, float, float]:
    """Token-boundary F1 treating B positions as positive class."""
    if len(gold_labels) != len(pred_labels):
        raise ValueError("length mismatch")
    tp = fp = fn = 0
    for g, p in zip(gold_labels, pred_labels):
        g_b = g == LABEL_B
        p_b = p == LABEL_B
        if p_b and g_b:
            tp += 1
        elif p_b and not g_b:
            fp += 1
        elif g_b and not p_b:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def iter_jsonl_examples(path: str) -> Iterable[Tuple[str, List[str]]]:
    """Yield (text, segments) from a JSONL dump file."""
    import json
    from pathlib import Path

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        yield row["text"], list(row["segments"])
