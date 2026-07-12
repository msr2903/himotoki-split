"""Trainable character-boundary model with numpy inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

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
    """Binary logistic boundary classifier (B vs I) with numpy weights."""

    def __init__(
        self,
        weights: np.ndarray,
        bias: float,
        meta: Optional[dict] = None,
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

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # stable sigmoid
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

    def predict(self, text: str) -> PredictResult:
        if not text:
            return PredictResult(segments=[], labels=[], confidence=1.0)
        p_b = self.predict_proba_b(text)
        labels = [
            LABEL_B if (i == 0 or p >= 0.5) else LABEL_I
            for i, p in enumerate(p_b)
        ]
        # Force first char to B
        labels[0] = LABEL_B
        # Confidence: probability of chosen class
        chosen = np.where(
            np.asarray(labels) == LABEL_B, p_b, 1.0 - p_b
        )
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
            meta_json=np.asarray([json.dumps(self.meta, ensure_ascii=False)]),
        )

    @classmethod
    def load(cls, path: PathLike) -> "BoundaryModel":
        data = np.load(path, allow_pickle=False)
        meta_raw = data["meta_json"]
        meta = json.loads(str(meta_raw[0])) if len(meta_raw) else {}
        return cls(
            weights=data["weights"],
            bias=float(data["bias"][0]),
            meta=meta,
        )


def train_boundary_model(
    texts: Sequence[str],
    segments_list: Sequence[Sequence[str]],
    *,
    C: float = 1.0,
    max_iter: int = 200,
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
    # Map: B=1 for positive class convenience in sklearn
    y_bin = (y == LABEL_B).astype(np.int64)
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    clf.fit(x, y_bin)
    weights = clf.coef_.reshape(-1).astype(np.float32)
    bias = float(clf.intercept_[0])
    return BoundaryModel(
        weights=weights,
        bias=bias,
        meta={
            "feature_dim": feature_dim(),
            "n_chars": int(x.shape[0]),
            "n_sentences": len(texts),
            "C": C,
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
    return load_model(path)


def load_model(path: PathLike) -> BoundaryModel:
    return BoundaryModel.load(path)
