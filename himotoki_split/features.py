"""Character window features for the boundary classifier."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# Script buckets
_SCRIPT_OTHER = 0
_SCRIPT_KANJI = 1
_SCRIPT_HIRA = 2
_SCRIPT_KATA = 3
_SCRIPT_ASCII = 4
_SCRIPT_DIGIT = 5
_SCRIPT_PUNCT = 6
_N_SCRIPTS = 7

WINDOW = 3  # chars left/right
CHAR_HASH_BUCKETS = 256
HASH_OFFSETS = (-2, -1, 0, 1, 2)


def char_script(ch: str) -> int:
    o = ord(ch)
    if 0x30 <= o <= 0x39:
        return _SCRIPT_DIGIT
    if 0x41 <= o <= 0x5A or 0x61 <= o <= 0x7A:
        return _SCRIPT_ASCII
    if 0x3040 <= o <= 0x309F:
        return _SCRIPT_HIRA
    if 0x30A0 <= o <= 0x30FF:
        return _SCRIPT_KATA
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
        return _SCRIPT_KANJI
    if ch in "。、．，！？!?,.「」『』（）()[]【】・…〜ー―-":
        return _SCRIPT_PUNCT
    return _SCRIPT_OTHER


def _one_hot(index: int, size: int) -> List[float]:
    row = [0.0] * size
    if 0 <= index < size:
        row[index] = 1.0
    return row


def _char_hash(ch: str) -> int:
    # Stable across processes; prefer identity for BMP
    return (ord(ch) * 2654435761) % CHAR_HASH_BUCKETS


def feature_dim() -> int:
    script_win = (2 * WINDOW + 1) * _N_SCRIPTS
    script_pairs = 2 * (_N_SCRIPTS * _N_SCRIPTS)
    flags = 6
    char_hashes = len(HASH_OFFSETS) * CHAR_HASH_BUCKETS
    bigram_hashes = 2 * CHAR_HASH_BUCKETS  # (prev,cur) and (cur,next) hashed
    return script_win + script_pairs + flags + char_hashes + bigram_hashes


def featurize_char(text: str, i: int) -> np.ndarray:
    """Feature vector for character index ``i`` in ``text``."""
    feats: List[float] = []
    n = len(text)
    scripts = [char_script(ch) for ch in text]

    for off in range(-WINDOW, WINDOW + 1):
        j = i + off
        if 0 <= j < n:
            feats.extend(_one_hot(scripts[j], _N_SCRIPTS))
        else:
            feats.extend([0.0] * _N_SCRIPTS)

    prev = scripts[i - 1] if i > 0 else _SCRIPT_OTHER
    nxt = scripts[i + 1] if i + 1 < n else _SCRIPT_OTHER
    center = scripts[i]
    feats.extend(_one_hot(prev * _N_SCRIPTS + center, _N_SCRIPTS * _N_SCRIPTS))
    feats.extend(_one_hot(center * _N_SCRIPTS + nxt, _N_SCRIPTS * _N_SCRIPTS))

    feats.append(1.0 if i == 0 else 0.0)
    feats.append(1.0 if i == n - 1 else 0.0)
    feats.append(1.0 if i > 0 and scripts[i] == scripts[i - 1] else 0.0)
    feats.append(1.0 if i + 1 < n and scripts[i] != scripts[i + 1] else 0.0)
    feats.append(1.0 if i > 0 and scripts[i - 1] == _SCRIPT_KANJI and scripts[i] == _SCRIPT_HIRA else 0.0)
    feats.append(1.0 if i > 0 and scripts[i - 1] == _SCRIPT_HIRA and scripts[i] == _SCRIPT_KANJI else 0.0)

    for off in HASH_OFFSETS:
        j = i + off
        if 0 <= j < n:
            feats.extend(_one_hot(_char_hash(text[j]), CHAR_HASH_BUCKETS))
        else:
            feats.extend([0.0] * CHAR_HASH_BUCKETS)

    # hashed char bigrams
    if i > 0:
        bg = (_char_hash(text[i - 1]) * 31 + _char_hash(text[i])) % CHAR_HASH_BUCKETS
    else:
        bg = 0
    feats.extend(_one_hot(bg, CHAR_HASH_BUCKETS))
    if i + 1 < n:
        bg2 = (_char_hash(text[i]) * 31 + _char_hash(text[i + 1])) % CHAR_HASH_BUCKETS
    else:
        bg2 = 0
    feats.extend(_one_hot(bg2, CHAR_HASH_BUCKETS))

    arr = np.asarray(feats, dtype=np.float32)
    if arr.shape[0] != feature_dim():
        raise RuntimeError(
            f"feature dim mismatch: got {arr.shape[0]} expected {feature_dim()}"
        )
    return arr


def featurize_text(text: str) -> np.ndarray:
    if not text:
        return np.zeros((0, feature_dim()), dtype=np.float32)
    return np.stack([featurize_char(text, i) for i in range(len(text))], axis=0)


def featurize_corpus(
    texts: Sequence[str], labels: Sequence[Sequence[int]]
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for text, lab in zip(texts, labels):
        if len(text) != len(lab):
            raise ValueError("text/label length mismatch")
        xs.append(featurize_text(text))
        ys.append(np.asarray(lab, dtype=np.int64))
    if not xs:
        return (
            np.zeros((0, feature_dim()), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)
