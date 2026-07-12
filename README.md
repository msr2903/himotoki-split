# himotoki-split

Lightweight **split-only** Japanese tokenizer distilled from [Himotoki](https://github.com/msr2903/himotoki).

Himotoki is a full morphological analyzer (dictionary + conjugations + meanings, ~1.8 GB DB).  
**himotoki-split** is a separate product: a tiny boundary model that approximates Himotoki’s word splits for users who only need segmentation.

| | Himotoki | himotoki-split |
|--|----------|----------------|
| Output | words + readings + POS + meanings | surface splits only |
| Footprint | ~1.8 GB SQLite | ~KB–MB model |
| Accuracy | gold / dictionary | approximates teacher |
| Fallback | — | optional Himotoki on low confidence |

## Install

```bash
pip install -e .
pip install -e ".[train]"      # linear student training
pip install -e ".[teacher]"    # Himotoki dump / fallback (needs DB)
pip install -e ".[neural]"     # BiLSTM train + ONNX export
pip install -e ".[onnx]"       # ONNX runtime only
```

## Usage

```python
from himotoki_split import split

result = split("学校で勉強しています")
print(result.segments)
print(result.confidence, result.source)
```

```bash
himotoki-split "猫が食べる"
himotoki-split --json "猫が食べる"
```

## Scale silver labels (Wikipedia → Himotoki → train)

Large dumps stay **local** (gitignored). Wikipedia text is **CC BY-SA**.

### Phase A — 100k sentences + linear student

```bash
# 1) Build cleaned sentence list (streams dump until --target)
python scripts/build_corpus.py --download --target 100000 -o data/corpus/sentences.txt
# or: python scripts/build_corpus.py --dump /path/to/jawiki-latest-pages-articles.xml.bz2 --target 100000

# 2) Shard for parallel labeling
python scripts/shard_corpus.py -i data/corpus/sentences.txt -o data/corpus/shards --num-shards 8

# 3) Dump Himotoki silver labels (requires Himotoki + DB; hours on one CPU)
for i in $(seq 0 7); do
  python scripts/dump_labels.py \
    -i data/corpus/sentences.txt \
    -o data/labels/shards/part-$(printf '%03d' $i).jsonl \
    --shard-id $i --num-shards 8 --resume &
done
wait

# 4) Merge + freeze 5k holdout
python scripts/merge_labels.py -i data/labels/shards/*.jsonl \
  --train-out data/labels/train.jsonl --holdout-out data/labels/holdout.jsonl

# 5) Train linear model + eval
python scripts/train.py -i data/labels/train.jsonl -o himotoki_split/models/default.npz
python scripts/eval_model.py -m himotoki_split/models/default.npz -i data/labels/holdout.jsonl
```

Rough cost: wiki extract is large (multi-GB download if using full articles dump); labeling ~20–100 ms/sentence ⇒ **100k ≈ a few hours** single-threaded (faster with shards).

### Phase B — 500k + neural ONNX student

```bash
python scripts/build_corpus.py --download --target 500000 -o data/corpus/sentences.txt
# ... dump + merge as above ...
python scripts/train_neural.py -i data/labels/train.jsonl -o himotoki_split/models/default.onnx --epochs 5
python scripts/eval_model.py -m himotoki_split/models/default.onnx -i data/labels/holdout.jsonl --backend onnx
```

`split()` prefers `default.onnx` when present (and onnxruntime is installed), otherwise `default.npz`.

## Small-scale distill (existing)

```bash
python scripts/dump_labels.py -i data/sample_sentences.txt -o data/labels.jsonl
python scripts/train.py -i data/labels.jsonl -o himotoki_split/models/default.npz
```

Shipped `default.npz` after Phase A: trained on a **30k** subset of ~**95k**
Himotoki-labeled Wikipedia sentences (100k corpus → 99,997 labels → 5k holdout).
Holdout boundary F1 ≈ **0.92**, exact-seg ≈ **10%** on noisy wiki text.
Optional `default.onnx` (BiLSTM) is included as a Phase B starter (~0.89 F1 on same holdout after a 5k smoke train).

Large artifacts (`data/corpus/`, `data/labels/train.jsonl`, wiki dumps) are **gitignored**.

## Design

- **Teacher:** Himotoki `segment_text` top path → `{text, segments}` JSONL  
- **Student (default):** logistic regression + Viterbi (`default.npz`, numpy)  
- **Student (optional):** char BiLSTM → ONNX  
- **Fallback:** low-confidence → Himotoki if installed  

## Relationship to Himotoki

Separate repository on purpose. Himotoki remains the accurate analyzer and training teacher.

## License

MIT — see [LICENSE](LICENSE).
