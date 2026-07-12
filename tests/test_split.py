"""Tests for split API and training roundtrip."""

from pathlib import Path

import pytest

from himotoki_split.labels import boundary_f1, segments_to_bio
from himotoki_split.model import train_boundary_model
from himotoki_split.split import reset_model_cache, split

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_labels.jsonl"
DEFAULT_MODEL = ROOT / "himotoki_split" / "models" / "default.npz"


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    pytest.importorskip("sklearn")
    from himotoki_split.model import load_jsonl_dataset

    texts, segs = load_jsonl_dataset(SAMPLE)
    assert len(texts) >= 3
    model = train_boundary_model(texts, segs, max_iter=200)
    path = tmp_path_factory.mktemp("m") / "m.npz"
    model.save(path)
    return path


def test_train_and_predict(trained_model):
    reset_model_cache()
    result = split(
        "猫が食べる",
        model_path=trained_model,
        fallback=False,
    )
    assert result.source == "model"
    assert "".join(result.segments) == "猫が食べる"
    assert len(result.segments) >= 2


def test_default_model_if_present():
    if not DEFAULT_MODEL.is_file():
        pytest.skip("default model not packaged yet")
    reset_model_cache()
    result = split("今日は天気がいい", fallback=False)
    assert "".join(result.segments) == "今日は天気がいい"
    assert result.source == "model"


def test_train_f1_reasonable(trained_model):
    from himotoki_split.model import load_jsonl_dataset, load_model

    texts, segs = load_jsonl_dataset(SAMPLE)
    model = load_model(trained_model)
    f1s = []
    for text, gold_segs in zip(texts[:50], segs[:50]):
        gold = segments_to_bio(text, gold_segs)
        pred = model.predict(text)
        f1s.append(boundary_f1(gold, pred.labels)[2])
    assert sum(f1s) / len(f1s) > 0.5
