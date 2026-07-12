# Tatoeba → himotoki-split

[Tatoeba](https://tatoeba.org) provides clean learner/example Japanese sentences.
Many are linked to **JMdict** via Tanaka-style word indices — a better domain
for split training than noisy Wikipedia.

## Two paths

### A. JMdict indices (recommended first)

Uses curated word indices (~150k) already aligned to JMdict senses/readings.
Gaps (proper nouns, punctuation) are filled by greedy alignment.

```bash
python scripts/tatoeba_indices_to_labels.py --download \
  -o data/labels/tatoeba_indices.jsonl
# → ~148k {text, segments} rows (teacher: tatoeba_jpn_indices)
```

### B. Sentences + Himotoki teacher

Same silver-label path as Wikipedia, but on cleaner Tatoeba text:

```bash
python scripts/build_tatoeba_corpus.py --download --target 200000 \
  -o data/corpus/tatoeba_sentences.txt

export HIMOTOKI_DB_PATH=~/.himotoki/himotoki.db
python scripts/dump_labels.py \
  -i data/corpus/tatoeba_sentences.txt \
  -o data/labels/tatoeba_himotoki.jsonl
```

Himotoki itself is built on JMdict glosses/conjugations, so path B is
“Tatoeba text × JMdict-backed teacher.” Path A is denser JMdict linking
when indices exist.

## Train / eval

```bash
# Freeze a Tatoeba holdout, rest for train
python scripts/merge_labels.py \
  -i data/labels/tatoeba_indices.jsonl \
  --train-out data/labels/tatoeba_train.jsonl \
  --holdout-out data/labels/tatoeba_holdout.jsonl \
  --holdout-size 5000

python scripts/train_neural.py -i data/labels/tatoeba_train.jsonl \
  -o himotoki_split/models/tatoeba.onnx --epochs 6

python scripts/eval_model.py -m himotoki_split/models/tatoeba.onnx \
  -i data/labels/tatoeba_holdout.jsonl --backend onnx
# Also compare on wiki clean:
python scripts/eval_model.py -m himotoki_split/models/tatoeba.onnx \
  -i data/labels/holdout_clean.jsonl --backend onnx
```

Optional mix with wiki silver (append-only, keep wiki holdout frozen):

```bash
python scripts/merge_labels.py \
  -i data/labels/tatoeba_indices.jsonl data/labels/shards/*.jsonl \
  --freeze-holdout data/labels/holdout.jsonl \
  --train-out data/labels/train_mixed.jsonl \
  --holdout-out data/labels/holdout.jsonl
```

## License

- Tatoeba sentences: **CC BY 2.0 FR** — attribute [Tatoeba](https://tatoeba.org) and sentence authors.
- Japanese indices: Tanaka Corpus / Tatoeba indexing contributors — attribute as required by Tatoeba downloads page.
- Large downloads stay local (gitignored).
