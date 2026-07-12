# himotoki-split

Lightweight **split-only** Japanese tokenizer distilled from [Himotoki](https://github.com/msr2903/himotoki).

Himotoki is a full morphological analyzer (dictionary + conjugations + meanings, ~1.8 GB DB).  
**himotoki-split** is a separate product: a tiny boundary model that approximates Himotoki’s word splits for users who only need segmentation.

| | Himotoki | himotoki-split |
|--|----------|----------------|
| Output | words + readings + POS + meanings | surface splits only |
| Footprint | ~1.8 GB SQLite | ~MB model (numpy) |
| Accuracy | gold / dictionary | approximates teacher |
| Fallback | — | optional Himotoki on low confidence |

## Install

```bash
pip install -e .
# training
pip install -e ".[train]"
# teacher / fallback (needs Himotoki + its DB)
pip install -e ".[teacher]"
```

## Usage

```python
from himotoki_split import split

result = split("学校で勉強しています")
print(result.segments)     # ['学校', 'で', ...]
print(result.confidence)   # 0.0–1.0
print(result.source)       # "model" or "himotoki"
```

CLI:

```bash
himotoki-split "猫が食べる"
himotoki-split --json "猫が食べる"
```

## Distill from Himotoki

1. Prepare sentences (one per line).
2. Dump silver labels (requires Himotoki DB):

```bash
python scripts/dump_labels.py -i data/sample_sentences.txt -o data/labels.jsonl
```

3. Train and overwrite the packaged model:

```bash
python scripts/train.py -i data/labels.jsonl -o himotoki_split/models/default.npz
```

Shipped `default.npz` was trained on **559** Himotoki silver labels
(`data/labels_himotoki510.jsonl` + samples): ~**0.86** held-out boundary F1
(char B/I), logistic regression + Viterbi, numpy runtime (~8 KB).

## Design

- **Teacher:** Himotoki `segment_text` top path → character B/I labels  
- **Student:** logistic regression over character-script window features  
- **Runtime:** numpy only (`default.npz`)  
- **Fallback:** if model confidence &lt; threshold and Himotoki is installed, use Himotoki

This is an MVP. Stronger students (BiLSTM/CRF → ONNX) can replace the classifier without changing the dump format or public `split()` API.

## Relationship to Himotoki

- **Separate repository / package** on purpose — do not merge into core `himotoki`.
- Himotoki remains the accurate analyzer and training teacher.
- himotoki-split depends on Himotoki only as an optional extra.

## License

MIT — see [LICENSE](LICENSE).
