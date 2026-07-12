# Changelog

## 0.2.2 — Active-query retrain

- Uncertainty sampling on train (8k scored → top 500); Himotoki oracle labels
- Mild upsample retrain: clean exact-seg **45.3% → 46.0%**, F1 **0.9715 → 0.9718**
- Script: `scripts/active_query_train.py`; feedback in `demo/data/feedback_active.jsonl`

## 0.2.1 — Demo UI

- Add FastAPI playground (`demo/`) with split + active-query review
- Feedback saved to `demo/data/feedback.jsonl` for later distillation

## 0.2.0 — Phase C soft-launch

- Ship full-data students: ONNX BiLSTM on ~95k silver labels; linear SGD on full train
- Default runtime prefers `default.onnx` when `onnxruntime` is installed
- Hybrid fallback default `min_confidence=0.96` (Phase B calibration)
- Add clean holdout slice, Phase B eval, fallback calibration, latency bench scripts
- Holdout (5k): ONNX F1 ≈0.97 / exact-seg ≈42%; linear improved vs Phase A 30k subset

## 0.1.0 — Phase A

- Initial package: linear boundary model, optional ONNX, Himotoki teacher dump pipeline
- 100k Wikipedia silver-label pipeline + frozen 5k holdout
