"""Optional Himotoki teacher / fallback integration."""

from __future__ import annotations

from typing import List, Optional, Sequence


def himotoki_available() -> bool:
    try:
        import himotoki  # noqa: F401

        return True
    except ImportError:
        return False


def split_with_himotoki(text: str) -> List[str]:
    """Return top-path surface segments from Himotoki."""
    import himotoki
    from himotoki.segment import segment_text
    from himotoki.db.connection import get_session

    session = get_session()
    paths = segment_text(session, text, limit=1)
    if not paths:
        return [text] if text else []
    segments, _score = paths[0]
    return [seg.text for seg in segments]


def dump_labels(
    sentences: Sequence[str],
    *,
    warm: bool = True,
) -> List[dict]:
    """Label sentences with Himotoki for distillation datasets."""
    import himotoki

    if warm:
        himotoki.warm_up()

    rows: List[dict] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        segs = split_with_himotoki(sentence)
        rows.append(
            {
                "text": sentence,
                "segments": segs,
                "teacher": "himotoki",
            }
        )
    return rows
