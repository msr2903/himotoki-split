"""Trainable character-boundary model with numpy inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from himotoki_split.features import feature_dim, featurize_corpus, featurize_text
from himotoki_split.labels import (
    LABEL_B,
    LABEL_I,
    bio_to_segments,
    segments_to_bio,
)

PathLike = Union[str, Path]


@dataclass
class PredictResult:
    segments: List[str]
    labels: List[int]
    # Mean P(predicted label); higher is more confident
    confidence: float


class BoundaryModel:
    """Binary logistic boundary classifier (B vs I) + optional Viterbi."""

    def __init__(
        self,
        weights: np.ndarray,
        bias: float,
        meta: Optional[dict] = None,
        trans_log: Optional[np.ndarray] = None,
    ) -> None:
        if weights.ndim != 1:
            raise ValueError("weights must be 1-D")
        if weights.shape[0] != feature_dim():
            raise ValueError(
                f"weights dim {weights.shape[0]} != feature_dim {feature_dim()}"
            )
        self.weights = weights.astype(np.float32, copy=False)
        self.bias = float(bias)
        self.meta = meta or {}
        # trans_log[prev_label, cur_label]
        if trans_log is None:
            self.trans_log = np.zeros((2, 2), dtype=np.float32)
        else:
            self.trans_log = trans_log.astype(np.float32, copy=False)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float32)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        return out

    def predict_proba_b(self, text: str) -> np.ndarray:
        """P(label=B) per character."""
        if not text:
            return np.zeros((0,), dtype=np.float32)
        x = featurize_text(text)
        logits = x @ self.weights + self.bias
        return self._sigmoid(logits)

    def _viterbi(self, p_b: np.ndarray) -> List[int]:
        """Decode with emission probs + transition log-potentials."""
        n = len(p_b)
        if n == 0:
            return []
        # emissions: label 0=B, 1=I
        emit = np.stack([np.log(np.clip(p_b, 1e-6, 1.0)), np.log(np.clip(1.0 - p_b, 1e-6, 1.0))], axis=1)
        dp = np.full((n, 2), -1e9, dtype=np.float32)
        bp = np.zeros((n, 2), dtype=np.int8)
        dp[0, LABEL_B] = emit[0, LABEL_B]
        dp[0, LABEL_I] = -1e9  # force B at start
        for i in range(1, n):
            for cur in (LABEL_B, LABEL_I):
                scores = dp[i - 1] + self.trans_log[:, cur] + emit[i, cur]
                bp[i, cur] = int(np.argmax(scores))
                dp[i, cur] = float(scores[bp[i, cur]])
        labels = [0] * n
        labels[-1] = int(np.argmax(dp[-1]))
        for i in range(n - 2, -1, -1):
            labels[i] = int(bp[i + 1, labels[i + 1]])
        labels[0] = LABEL_B
        return labels

    def predict(self, text: str) -> PredictResult:
        if not text:
            return PredictResult(segments=[], labels=[], confidence=1.0)
        p_b = self.predict_proba_b(text)
        labels = self._viterbi(p_b)
        chosen = np.where(np.asarray(labels) == LABEL_B, p_b, 1.0 - p_b)
        segments = bio_to_segments(text, labels)
        return PredictResult(
            segments=segments,
            labels=labels,
            confidence=float(chosen.mean()) if len(chosen) else 1.0,
        )

    def save(self, path: PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights,
            bias=np.asarray([self.bias], dtype=np.float32),
            trans_log=self.trans_log,
            meta_json=np.asarray([json.dumps(self.meta, ensure_ascii=False)]),
        )

    @classmethod
    def load(cls, path: PathLike) -> "BoundaryModel":
        data = np.load(path, allow_pickle=False)
        meta_raw = data["meta_json"]
        meta = json.loads(str(meta_raw[0])) if len(meta_raw) else {}
        trans = data["trans_log"] if "trans_log" in data.files else None
        return cls(
            weights=data["weights"],
            bias=float(data["bias"][0]),
            meta=meta,
            trans_log=trans,
        )


def _estimate_transitions(labels_list: Sequence[Sequence[int]]) -> np.ndarray:
    counts = np.ones((2, 2), dtype=np.float64)  # add-one smoothing
    for labels in labels_list:
        for a, b in zip(labels, labels[1:]):
            counts[int(a), int(b)] += 1.0
    probs = counts / counts.sum(axis=1, keepdims=True)
    return np.log(probs).astype(np.float32)


def train_boundary_model(
    texts: Sequence[str],
    segments_list: Sequence[Sequence[str]],
    *,
    C: float = 1.0,
    max_iter: int = 400,
    random_state: int = 0,
) -> BoundaryModel:
    """Train with scikit-learn LogisticRegression (optional train extra)."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError(
            "Training requires scikit-learn. Install with: "
            "pip install 'himotoki-split[train]'"
        ) from exc

    labels = [segments_to_bio(t, segs) for t, segs in zip(texts, segments_list)]
    x, y = featurize_corpus(texts, labels)
    y_bin = (y == LABEL_B).astype(np.int64)
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(x, y_bin)
    weights = clf.coef_.reshape(-1).astype(np.float32)
    bias = float(clf.intercept_[0])
    trans = _estimate_transitions(labels)
    return BoundaryModel(
        weights=weights,
        bias=bias,
        trans_log=trans,
        meta={
            "feature_dim": feature_dim(),
            "n_chars": int(x.shape[0]),
            "n_sentences": len(texts),
            "C": C,
            "decoder": "viterbi",
        },
    )


def load_jsonl_dataset(path: PathLike) -> Tuple[List[str], List[List[str]]]:
    from himotoki_split.labels import iter_jsonl_examples

    texts: List[str] = []
    segs: List[List[str]] = []
    for text, segments in iter_jsonl_examples(str(path)):
        texts.append(text)
        segs.append(segments)
    return texts, segs


def default_model_path() -> Path:
    return Path(__file__).resolve().parent / "models" / "default.npz"


def load_default_model() -> BoundaryModel:
    path = default_model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Default model missing at {path}. Train one with scripts/train.py"
        )
    return BoundaryModel.load(path)


def load_model(path: PathLike) -> BoundaryModel:
    return BoundaryModel.load(path)
