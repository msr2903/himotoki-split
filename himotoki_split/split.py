"""Public split API."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

from himotoki_split.model import BoundaryModel, load_default_model, load_model

PathLike = Union[str, Path]

_MODEL: Optional[BoundaryModel] = None


@dataclass
class SplitResult:
    text: str
    segments: List[str]
    confidence: float
    source: str  # "model" | "himotoki" | "passthrough"
    labels: List[int] = field(default_factory=list)


def _get_model(model_path: Optional[PathLike] = None) -> BoundaryModel:
    global _MODEL
    if model_path is not None:
        return load_model(model_path)
    if _MODEL is None:
        _MODEL = load_default_model()
    return _MODEL


def reset_model_cache() -> None:
    global _MODEL
    _MODEL = None


def split(
    text: str,
    *,
    model_path: Optional[PathLike] = None,
    fallback: bool = True,
    min_confidence: float = 0.55,
) -> SplitResult:
    """Split Japanese text into word-like surface segments.

    Args:
        text: Input sentence (no length hard-cap here; keep inputs reasonable).
        model_path: Optional path to a ``.npz`` boundary model.
        fallback: If True and confidence < ``min_confidence``, try Himotoki.
        min_confidence: Threshold for accepting the distilled model output.
    """
    if not text:
        return SplitResult(
            text=text, segments=[], confidence=1.0, source="passthrough"
        )

    model = _get_model(model_path)
    pred = model.predict(text)

    if (
        fallback
        and pred.confidence < min_confidence
    ):
        try:
            from himotoki_split.fallback import (
                himotoki_available,
                split_with_himotoki,
            )

            if himotoki_available():
                segs = split_with_himotoki(text)
                return SplitResult(
                    text=text,
                    segments=segs,
                    confidence=1.0,
                    source="himotoki",
                )
        except Exception:
            # Keep model result if fallback fails (missing DB, etc.)
            pass

    return SplitResult(
        text=text,
        segments=pred.segments,
        confidence=pred.confidence,
        source="model",
        labels=pred.labels,
    )


def split_texts(
    texts: Sequence[str],
    **kwargs,
) -> List[SplitResult]:
    return [split(t, **kwargs) for t in texts]
