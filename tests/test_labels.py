"""Tests for BIO labels."""

from himotoki_split.labels import bio_to_segments, boundary_f1, segments_to_bio


def test_roundtrip():
    text = "猫が食べる"
    segs = ["猫", "が", "食べる"]
    labels = segments_to_bio(text, segs)
    assert bio_to_segments(text, labels) == segs


def test_boundary_f1_perfect():
    labels = [0, 1, 0, 1, 0, 1, 1]
    p, r, f = boundary_f1(labels, labels)
    assert p == r == f == 1.0
