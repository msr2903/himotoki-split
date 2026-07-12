# Beginner walkthrough

Interactive canvas (same content, nicer UI):

```bash
python -m demo.app
# open http://127.0.0.1:8765/walkthrough
```

## 60-second story

Japanese has no spaces. **himotoki-split** predicts word boundaries.

1. Himotoki (teacher) labels Wikipedia sentences → `data/labels/train.jsonl`
2. We train a small BiLSTM → `himotoki_split/models/default.onnx`
3. We score on frozen holdout → `output/eval_*.json`

## Eval (the only numbers that matter)

| Metric | Meaning | Good |
|--------|---------|------|
| **Boundary F1** | Are cuts in the right places? (0–1) | ≥ 0.97 on clean |
| **Exact-seg** | Is the *whole sentence* perfect? | Harder; ~45% clean is OK for now |

**Holdout 5k** = frozen wiki (noisier). **Clean ~800** = shorter / less junk — quote this for “how good does it feel?”

Exact can look low while F1 looks great: one wrong cut fails the whole sentence.

## Training logs

```
epoch 6/6 loss=0.0594 val_f1=0.9824 val_exact=0.5570
```

- `loss` should fall
- `val_*` is a progress check on train data — **optimistic**
- Always confirm with `scripts/eval_model.py` on holdout/clean

## Files to open

- `output/phase_b_summary.json` — Phase B story
- `output/active_query_summary.json` — active-learning comparison
- `output/eval_metrics_active_mild_onnx.json` — current shipped eval

## Try it

```bash
himotoki-split "猫が食べる"
python scripts/eval_model.py -m himotoki_split/models/default.onnx \
  -i data/labels/holdout.jsonl --clean data/labels/holdout_clean.jsonl
```
