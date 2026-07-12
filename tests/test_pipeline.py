"""Tests for corpus pipeline helpers (no Himotoki DB required)."""

from pathlib import Path

from scripts.build_corpus import is_good_sentence, split_sentences, strip_wikitext


def test_strip_and_split():
    raw = "これは[[テスト|試験]]です。次の文！\n"
    plain = strip_wikitext(raw)
    sents = list(split_sentences(plain))
    assert any("試験" in s or "テスト" in s or "です" in s for s in sents)


def test_good_sentence_filter():
    assert is_good_sentence("今日は良い天気です", 8, 80, 0.5)
    assert not is_good_sentence("abc", 8, 80, 0.5)
    assert not is_good_sentence("短い", 8, 80, 0.5)


def test_merge_labels(tmp_path: Path):
    import json
    import subprocess
    import sys

    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    rows = [
        {"text": "猫が食べる", "segments": ["猫", "が", "食べる"], "teacher": "himotoki"},
        {"text": "犬が走る", "segments": ["犬", "が", "走る"], "teacher": "himotoki"},
        {"text": "鳥が鳴く", "segments": ["鳥", "が", "鳴く"], "teacher": "himotoki"},
    ]
    a.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n" + json.dumps(rows[1], ensure_ascii=False) + "\n", encoding="utf-8")
    b.write_text(json.dumps(rows[2], ensure_ascii=False) + "\n" + json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    train = tmp_path / "train.jsonl"
    hold = tmp_path / "hold.jsonl"
    root = Path(__file__).resolve().parents[1]
    subprocess.check_call(
        [
            sys.executable,
            str(root / "scripts" / "merge_labels.py"),
            "-i",
            str(a),
            str(b),
            "--train-out",
            str(train),
            "--holdout-out",
            str(hold),
            "--holdout-size",
            "1",
            "--seed",
            "0",
        ],
        cwd=str(root),
    )
    assert train.is_file() and hold.is_file()
    hold_n = len(hold.read_text(encoding="utf-8").strip().splitlines())
    train_n = len(train.read_text(encoding="utf-8").strip().splitlines())
    assert hold_n == 1
    assert train_n == 2
