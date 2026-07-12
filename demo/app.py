"""FastAPI demo for himotoki-split: playground + active query review."""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from himotoki_split.split import reset_model_cache, split as hs_split
from himotoki_split.split import _get_model

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DATA = ROOT / "data"
FEEDBACK_PATH = DATA / "feedback.jsonl"
POOL_PATH = DATA / "pool.jsonl"
# Prefer clean holdout as default unlabeled-looking pool for active query
DEFAULT_POOL_CANDIDATES = [
    POOL_PATH,
    ROOT.parent / "data" / "labels" / "holdout_clean.jsonl",
    ROOT.parent / "data" / "sample_sentences.txt",
]

_lock = threading.Lock()
_reviewed: set[str] = set()
_pool_cache: Optional[List[Dict[str, Any]]] = None


app = FastAPI(
    title="himotoki-split demo",
    description="Try the boundary model and review uncertain splits (active query).",
    version="0.2.1",
)


class SplitRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    fallback: bool = False
    min_confidence: float = 0.96
    model: str = Field(
        default="default",
        description="default | tatoeba | or a .onnx/.npz path",
    )


def _resolve_model_path(name: str) -> Optional[Path]:
    if name in ("", "default", "auto"):
        return None
    models = ROOT.parent / "himotoki_split" / "models"
    if name == "tatoeba":
        p = models / "tatoeba.onnx"
        return p if p.is_file() else None
    p = Path(name)
    if p.is_file():
        return p
    cand = models / name
    return cand if cand.is_file() else None


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    model_segments: List[str]
    model_confidence: float
    action: str = Field(..., pattern="^(accept|correct|skip)$")
    corrected_segments: Optional[List[str]] = None
    notes: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_reviewed() -> None:
    global _reviewed
    _reviewed = set()
    if not FEEDBACK_PATH.is_file():
        return
    with FEEDBACK_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text")
            if text and row.get("action") in ("accept", "correct", "skip"):
                _reviewed.add(text)


def _load_pool() -> List[Dict[str, Any]]:
    global _pool_cache
    if _pool_cache is not None:
        return _pool_cache

    rows: List[Dict[str, Any]] = []
    for path in DEFAULT_POOL_CANDIDATES:
        if not path.is_file():
            continue
        if path.suffix == ".txt":
            with path.open(encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if text:
                        rows.append({"text": text})
        else:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = row.get("text")
                    if text:
                        rows.append(
                            {
                                "text": text,
                                "gold_segments": row.get("segments"),
                            }
                        )
        if rows:
            _pool_cache = rows
            return rows
    _pool_cache = []
    return _pool_cache


def _char_probs(text: str) -> List[float]:
    """P(boundary / label B) per character from the loaded student model."""
    model = _get_model()
    if hasattr(model, "predict_proba_b"):
        probs = model.predict_proba_b(text)
        return [float(p) for p in probs]
    # Fallback: approximate from labels only
    pred = model.predict(text)
    return [1.0 if lab == 0 else 0.0 for lab in pred.labels]


def _uncertainty(text: str) -> Dict[str, Any]:
    """Active-query score: higher = more useful to label."""
    result = hs_split(text, fallback=False)
    probs = _char_probs(text)
    # Boundary entropy (binary) averaged; uncertain near 0.5
    ents = []
    for p in probs:
        p = min(max(p, 1e-6), 1 - 1e-6)
        ents.append(-(p * math.log(p) + (1 - p) * math.log(1 - p)))
    mean_entropy = sum(ents) / len(ents) if ents else 0.0
    # Prefer low confidence + high entropy
    score = (1.0 - float(result.confidence)) + 0.5 * mean_entropy
    return {
        "text": text,
        "segments": result.segments,
        "confidence": float(result.confidence),
        "source": result.source,
        "labels": result.labels,
        "char_probs": probs,
        "mean_entropy": mean_entropy,
        "query_score": score,
    }


def _split_payload(text: str, fallback: bool, min_confidence: float) -> Dict[str, Any]:
    result = hs_split(
        text, fallback=fallback, min_confidence=min_confidence
    )
    probs = _char_probs(text) if result.source == "model" else []
    return {
        "text": result.text,
        "segments": result.segments,
        "confidence": float(result.confidence),
        "source": result.source,
        "labels": result.labels,
        "char_probs": probs,
        "joined_ok": "".join(result.segments) == text,
    }


@app.on_event("startup")
def _startup() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    _load_reviewed()
    _load_pool()
    # Warm model
    reset_model_cache()
    hs_split("猫が食べる", fallback=False)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    model = _get_model()
    return {
        "ok": True,
        "model": type(model).__name__,
        "pool_size": len(_load_pool()),
        "reviewed": len(_reviewed),
        "feedback_path": str(FEEDBACK_PATH),
    }


@app.post("/api/split")
def api_split(body: SplitRequest) -> Dict[str, Any]:
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty text")
    model_path = _resolve_model_path(body.model)
    if body.model not in ("", "default", "auto") and model_path is None:
        raise HTTPException(404, f"Unknown model: {body.model}")
    # Temporarily bypass cache when a specific model is requested
    from himotoki_split.split import split as _split

    if model_path is not None:
        from himotoki_split.neural import OnnxBoundaryModel
        from himotoki_split.model import load_model

        if model_path.suffix == ".onnx":
            model = OnnxBoundaryModel.load(model_path)
        else:
            model = load_model(model_path)
        pred = model.predict(text)
        probs = []
        if hasattr(model, "predict_proba_b"):
            probs = [float(p) for p in model.predict_proba_b(text)]
        return {
            "text": text,
            "segments": pred.segments,
            "confidence": float(pred.confidence),
            "source": "model",
            "labels": pred.labels,
            "char_probs": probs,
            "joined_ok": "".join(pred.segments) == text,
            "model": body.model,
        }
    payload = _split_payload(text, body.fallback, body.min_confidence)
    payload["model"] = "default"
    return payload


@app.get("/api/active/next")
def active_next(limit: int = 1) -> Dict[str, Any]:
    """Return the most uncertain unreviewed pool sentence(s)."""
    limit = max(1, min(limit, 20))
    pool = _load_pool()
    if not pool:
        raise HTTPException(
            404,
            "No pool loaded. Add demo/data/pool.jsonl or holdout_clean.jsonl.",
        )

    candidates = [r for r in pool if r["text"] not in _reviewed]
    if not candidates:
        return {
            "items": [],
            "remaining": 0,
            "message": "Pool exhausted — all items reviewed.",
        }

    # Score a capped slice for responsiveness
    sample = candidates[: min(len(candidates), 300)]
    scored = [_uncertainty(r["text"]) for r in sample]
    scored.sort(key=lambda x: x["query_score"], reverse=True)
    items = scored[:limit]
    # Attach gold if present (for optional reveal / teaching)
    gold_by_text = {
        r["text"]: r.get("gold_segments") for r in sample if r.get("gold_segments")
    }
    for item in items:
        item["has_gold"] = bool(gold_by_text.get(item["text"]))
        # Do not send gold by default — keep active query honest
    return {
        "items": items,
        "remaining": len(candidates),
        "scored": len(sample),
    }


@app.get("/api/active/reveal-gold")
def reveal_gold(text: str) -> Dict[str, Any]:
    """Optional: show teacher/gold segments for a pool sentence."""
    for row in _load_pool():
        if row["text"] == text and row.get("gold_segments"):
            return {"text": text, "gold_segments": row["gold_segments"]}
    raise HTTPException(404, "No gold segments for this text")


@app.post("/api/active/feedback")
def active_feedback(body: FeedbackRequest) -> Dict[str, Any]:
    text = body.text.strip()
    if body.action == "correct":
        segs = body.corrected_segments or []
        if not segs:
            raise HTTPException(400, "corrected_segments required for correct")
        if "".join(segs) != text:
            raise HTTPException(
                400,
                "corrected_segments must concatenate to the original text",
            )
    else:
        segs = body.model_segments

    row = {
        "ts": _utc_now(),
        "text": text,
        "action": body.action,
        "model_segments": body.model_segments,
        "model_confidence": body.model_confidence,
        "segments": segs,
        "notes": body.notes,
    }
    with _lock:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        _reviewed.add(text)

    return {"ok": True, "reviewed": len(_reviewed), "saved": row}


@app.get("/api/active/stats")
def active_stats() -> Dict[str, Any]:
    counts = {"accept": 0, "correct": 0, "skip": 0}
    if FEEDBACK_PATH.is_file():
        with FEEDBACK_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                a = row.get("action")
                if a in counts:
                    counts[a] += 1
    return {
        "reviewed": len(_reviewed),
        "pool_size": len(_load_pool()),
        "remaining": max(0, len(_load_pool()) - len(_reviewed)),
        "actions": counts,
        "feedback_path": str(FEEDBACK_PATH.relative_to(ROOT.parent)),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/walkthrough")
def walkthrough() -> FileResponse:
    return FileResponse(STATIC / "walkthrough.html")


@app.get("/api/walkthrough")
def api_walkthrough() -> Dict[str, Any]:
    """Bundle beginner-facing metrics for the walkthrough canvas."""
    root = ROOT.parent
    out = root / "output"

    def _read(name: str) -> Optional[Dict[str, Any]]:
        path = out / name
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    pre = _read("eval_metrics_pre_active_onnx.json")
    mild = _read("eval_metrics_active_mild_onnx.json")
    heavy = _read("eval_metrics_active_onnx.json")
    aq_sum = _read("active_query_summary.json")
    aq = _read("active_query_metrics.json")
    phase_b = _read("phase_b_summary.json")
    phase_c = _read("phase_c_summary.json")
    train_mild = _read("neural_train_active_mild.json")

    train_log_sample = None
    if train_mild and train_mild.get("epoch_logs"):
        logs = train_mild["epoch_logs"]
        lines = []
        for log in logs[:2] + logs[-1:]:
            lines.append(
                f"epoch {log.get('epoch')}/{train_mild.get('epochs', '?')} "
                f"loss={log.get('loss', 0):.4f} "
                f"val_f1={log.get('val_boundary_f1', 0):.4f} "
                f"val_exact={log.get('val_exact_seg', 0):.4f}"
            )
        # de-dupe if only 1-2 epochs
        train_log_sample = "\n".join(dict.fromkeys(lines))

    selected = (aq_sum or {}).get("selected") or "active_mild"
    return {
        "selected": selected,
        "pre_active": pre,
        "active_mild": mild,
        "active_heavy": heavy,
        "active_query": (aq_sum or {}).get("active_query") or aq,
        "phase_b": phase_b,
        "phase_c": phase_c,
        "train_log_sample": train_log_sample,
    }


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "demo.app:app",
        host="0.0.0.0",
        port=8765,
        reload=False,
    )


if __name__ == "__main__":
    main()
