"""Small char BiLSTM boundary tagger with ONNX export/runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from himotoki_split.labels import LABEL_B, LABEL_I, bio_to_segments, segments_to_bio

PathLike = Union[str, Path]

# Fixed Unicode BMP hashing into embedding rows (no external vocab file needed)
VOCAB_SIZE = 8192
UNK_ID = 0
PAD_ID = 1


def char_id(ch: str) -> int:
    return (ord(ch) % (VOCAB_SIZE - 2)) + 2


def encode_text(text: str) -> np.ndarray:
    if not text:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray([char_id(ch) for ch in text], dtype=np.int64)


@dataclass
class NeuralPredictResult:
    segments: List[str]
    labels: List[int]
    confidence: float


class OnnxBoundaryModel:
    """Inference wrapper around an ONNX char-BiLSTM tagger."""

    def __init__(self, session, meta: Optional[dict] = None) -> None:
        self.session = session
        self.meta = meta or {}
        inputs = session.get_inputs()
        self._input_name = inputs[0].name

    @classmethod
    def load(cls, path: PathLike) -> "OnnxBoundaryModel":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX runtime required: pip install 'himotoki-split[onnx]'"
            ) from exc
        sess = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        return cls(sess, meta={"path": str(path)})

    def predict_proba_b(self, text: str) -> np.ndarray:
        ids = encode_text(text)
        if ids.size == 0:
            return np.zeros((0,), dtype=np.float32)
        # model expects [batch, seq]
        logits = self.session.run(
            None, {self._input_name: ids[None, :]}
        )[0]  # [1, seq, 2]
        logits = logits[0]
        # softmax over last dim, take P(B)=class0
        m = logits.max(axis=-1, keepdims=True)
        exp = np.exp(logits - m)
        prob = exp / exp.sum(axis=-1, keepdims=True)
        return prob[:, LABEL_B].astype(np.float32)

    def predict(self, text: str) -> NeuralPredictResult:
        if not text:
            return NeuralPredictResult([], [], 1.0)
        p_b = self.predict_proba_b(text)
        labels = [
            LABEL_B if (i == 0 or p >= 0.5) else LABEL_I for i, p in enumerate(p_b)
        ]
        labels[0] = LABEL_B
        chosen = np.where(np.asarray(labels) == LABEL_B, p_b, 1.0 - p_b)
        return NeuralPredictResult(
            segments=bio_to_segments(text, labels),
            labels=labels,
            confidence=float(chosen.mean()),
        )


def train_bilstm_and_export(
    texts: Sequence[str],
    segments_list: Sequence[Sequence[str]],
    onnx_path: PathLike,
    *,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    emb_dim: int = 64,
    hidden: int = 128,
    device: str = "cpu",
) -> dict:
    """Train a tiny BiLSTM tagger with PyTorch and export ONNX."""
    try:
        import torch
        import torch.nn as nn
        from torch.nn.utils.rnn import pad_sequence
    except ImportError as exc:
        raise ImportError(
            "Training neural student requires PyTorch: pip install 'himotoki-split[neural]'"
        ) from exc

    labels = [segments_to_bio(t, s) for t, s in zip(texts, segments_list)]
    xs = [torch.tensor(encode_text(t), dtype=torch.long) for t in texts]
    ys = [torch.tensor(lab, dtype=torch.long) for lab in labels]

    class CharBiLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(VOCAB_SIZE, emb_dim, padding_idx=PAD_ID)
            self.lstm = nn.LSTM(
                emb_dim,
                hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.out = nn.Linear(hidden * 2, 2)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            # tokens: [batch, seq] — no pack_padded (ONNX-friendly)
            emb = self.emb(tokens)
            out, _ = self.lstm(emb)
            return self.out(out)

    model = CharBiLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    idx = list(range(len(xs)))

    def batches():
        import random

        random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            batch_i = idx[start : start + batch_size]
            seqs = [xs[i] for i in batch_i]
            labs = [ys[i] for i in batch_i]
            padded = pad_sequence(seqs, batch_first=True, padding_value=PAD_ID)
            padded_y = pad_sequence(labs, batch_first=True, padding_value=-100)
            yield padded.to(device), padded_y.to(device)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        n = 0
        for padded, padded_y in batches():
            opt.zero_grad()
            logits = model(padded)
            loss = loss_fn(logits.reshape(-1, 2), padded_y.reshape(-1))
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        print(f"epoch {epoch+1}/{epochs} loss={total / max(n, 1):.4f}")

    # Export ONNX with dynamic sequence length (batch=1)
    model.eval()
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randint(2, VOCAB_SIZE, (1, 16), dtype=torch.long, device="cpu")
    model_cpu = model.to("cpu")
    model_cpu.eval()
    torch.onnx.export(
        model_cpu,
        dummy,
        str(onnx_path),
        input_names=["tokens"],
        output_names=["logits"],
        dynamic_axes={
            "tokens": {1: "seq"},
            "logits": {1: "seq"},
        },
        opset_version=18,
        dynamo=False,
    )
    return {
        "onnx_path": str(onnx_path),
        "n_sentences": len(texts),
        "epochs": epochs,
        "emb_dim": emb_dim,
        "hidden": hidden,
    }
