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
    batch_sentences: int = 512,
) -> BoundaryModel:
    """Train with scikit-learn SGD (memory-safe for large corpora)."""
    try:
        from sklearn.linear_model import SGDClassifier
    except ImportError as exc:
        raise ImportError(
            "Training requires scikit-learn. Install with: "
            "pip install 'himotoki-split[train]'"
        ) from exc

    labels = [segments_to_bio(t, segs) for t, segs in zip(texts, segments_list)]
    # Estimate class balance from a sample
    sample_y = []
    for lab in labels[: min(len(labels), 5000)]:
        sample_y.extend(lab)
    sample_y_arr = np.asarray(sample_y, dtype=np.int64)
    n_b = int((sample_y_arr == LABEL_B).sum())
    n_i = int((sample_y_arr == LABEL_I).sum())
    # weight inversely proportional to frequency
    w_b = (n_b + n_i) / (2 * max(n_b, 1))
    w_i = (n_b + n_i) / (2 * max(n_i, 1))
    class_weight = {0: w_i, 1: w_b}  # keys are y_bin values: I=0? wait y_bin is B=1
    # y_bin: B -> 1, I -> 0
    class_weight = {1: float(w_b), 0: float(w_i)}

    alpha = 1.0 / max(C, 1e-6) / max(len(texts), 1)
    clf = SGDClassifier(
        loss="log_loss",
        alpha=max(alpha, 1e-6),
        random_state=random_state,
        max_iter=1,
        learning_rate="optimal",
        class_weight=class_weight,
        tol=None,
    )

    rng = np.random.default_rng(random_state)
    order = np.arange(len(texts))
    n_chars = 0
    classes = np.array([0, 1], dtype=np.int64)
    for epoch in range(max(1, max_iter // 50)):
        rng.shuffle(order)
        for start in range(0, len(order), batch_sentences):
            batch_idx = order[start : start + batch_sentences]
            batch_texts = [texts[i] for i in batch_idx]
            batch_labels = [labels[i] for i in batch_idx]
            x, y = featurize_corpus(batch_texts, batch_labels)
            y_bin = (y == LABEL_B).astype(np.int64)
            n_chars += int(x.shape[0])
            if epoch == 0 and start == 0:
                clf.partial_fit(x, y_bin, classes=classes)
            else:
                clf.partial_fit(x, y_bin)
        print(f"  sgd epoch {epoch+1} chars_seen~{n_chars}", flush=True)

    weights = clf.coef_.reshape(-1).astype(np.float32)
    bias = float(clf.intercept_[0])
    # Estimate transitions on a sample for speed
    sample_n = min(len(labels), 20000)
    sample_labels = labels[:sample_n]
    trans = _estimate_transitions(sample_labels)
    return BoundaryModel(
        weights=weights,
        bias=bias,
        trans_log=trans,
        meta={
            "feature_dim": feature_dim(),
            "n_chars_seen": int(n_chars),
            "n_sentences": len(texts),
            "C": C,
            "decoder": "viterbi",
            "trainer": "sgd",
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
