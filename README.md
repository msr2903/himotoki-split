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

`split()` prefers packaged `default.onnx` when `onnxruntime` is installed, otherwise `default.npz`.  
Default hybrid fallback threshold is **`min_confidence=0.96`** (Phase B calibration).

## Demo UI (FastAPI)

Interactive playground + **active query** (surfaces low-confidence pool sentences for Accept / Correct feedback):

```bash
pip install -e ".[demo]"
python -m demo.app
# → http://127.0.0.1:8765
```

Feedback is appended to `demo/data/feedback.jsonl` (gitignored). The active-query pool defaults to `data/labels/holdout_clean.jsonl` when present, or `demo/data/pool.jsonl`.

### Active-query retrain (batch)

Agent/oracle loop over uncertain *train* examples (holdout stays frozen):

```bash
export HIMOTOKI_DB_PATH=~/.himotoki/himotoki.db
python scripts/active_query_train.py --sample-n 8000 --query-k 500 --upsample 5 --epochs 6 --train
python scripts/eval_model.py -m himotoki_split/models/default.onnx \
  -i data/labels/holdout.jsonl --clean data/labels/holdout_clean.jsonl
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

### Phase B — full-data neural ONNX student

Trained BiLSTM on the **full ~95k** silver train set (6 epochs, CPU) and shipped `default.onnx`.

```bash
# Train ONNX student on full train.jsonl
python scripts/train_neural.py -i data/labels/train.jsonl \
  -o himotoki_split/models/default.onnx --epochs 6 --batch-size 64 --device cpu

# Clean eval slice + dual metrics
python scripts/make_clean_eval.py
python scripts/eval_phase_b.py

# Calibrate hybrid fallback (needs Himotoki DB)
python scripts/calibrate_fallback.py -m himotoki_split/models/default.onnx
```

**Holdout results (frozen 5k wiki silver):**

| Backend | Boundary F1 | Exact-seg |
|---------|-------------|-----------|
| Linear `default.npz` (30k subset) | 0.917 | 10.6% |
| ONNX `default.onnx` (full 95k) | **0.970** | **41.9%** |

**Clean slice (800 sentences):** ONNX F1 **0.972**, exact-seg **45.3%**.

Hybrid fallback at `min_confidence=0.96`: exact-seg **48.2%** with ~**14%** Himotoki calls.

### Phase C — soft-launch (v0.2.0)

Packaging polish for a public soft-launch:

```bash
# Retrain zero-deps linear student on full train (optional; already shipped)
python scripts/train.py -i data/labels/train.jsonl -o himotoki_split/models/default.npz

# Latency microbench (clean-100 sample)
python scripts/bench_latency.py

# Build + smoke
pip install build twine
python -m build && twine check dist/*
```

**Latency** (100 clean sentences, CPU): ONNX ≈ **0.34 ms/sent** (~3k sent/s); linear ≈ **2.4 ms/sent**; `split()` API ≈ **0.61 ms/sent**.

Linear full-95k retrain landed at similar holdout F1 to the Phase A 30k model (~0.917) — capacity-limited; prefer ONNX for quality.

See [CHANGELOG.md](CHANGELOG.md) and [PUBLISH.md](PUBLISH.md) for release steps (TestPyPI/PyPI).

Optional later growth (append-only train, freeze holdout):

```bash
python scripts/build_corpus.py --download --target 500000 -o data/corpus/sentences.txt
# ... dump shards ...
python scripts/merge_labels.py -i data/labels/shards/*.jsonl \
  --freeze-holdout data/labels/holdout.jsonl \
  --train-out data/labels/train.jsonl --holdout-out data/labels/holdout.jsonl
python scripts/train_neural.py -i data/labels/train.jsonl -o himotoki_split/models/default.onnx
```

## Small-scale distill (existing)

```bash
python scripts/dump_labels.py -i data/sample_sentences.txt -o data/labels.jsonl
python scripts/train.py -i data/labels.jsonl -o himotoki_split/models/default.npz
```

Large artifacts (`data/corpus/`, `data/labels/train.jsonl`, wiki dumps) are **gitignored**.  
`data/labels/holdout_clean.jsonl` is kept in-repo for reproducible clean eval.

## Design

- **Teacher:** Himotoki analyze top path → `{text, segments}` JSONL  
- **Student (default runtime):** char BiLSTM → ONNX (`default.onnx`) when onnxruntime is available  
- **Student (zero-deps fallback):** logistic + Viterbi (`default.npz`, numpy)  
- **Fallback:** low-confidence → Himotoki if installed (`min_confidence=0.96`)  

## Relationship to Himotoki

Separate repository on purpose. Himotoki remains the accurate analyzer and training teacher.

## License

MIT — see [LICENSE](LICENSE).
