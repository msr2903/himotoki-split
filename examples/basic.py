"""Minimal usage example for himotoki-split."""

from himotoki_split import split

text = "学校で勉強しています"
result = split(text, fallback=False)
print("text:", text)
print("segments:", result.segments)
print("confidence:", round(result.confidence, 3))
print("source:", result.source)
