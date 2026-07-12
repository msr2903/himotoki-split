# Changelog

## 0.2.0 — Phase C soft-launch

- Ship full-data students: ONNX BiLSTM on ~95k silver labels; linear SGD on full train
- Default runtime prefers `default.onnx` when `onnxruntime` is installed
- Hybrid fallback default `min_confidence=0.96` (Phase B calibration)
- Add clean holdout slice, Phase B eval, fallback calibration, latency bench scripts
- Holdout (5k): ONNX F1 ≈0.97 / exact-seg ≈42%; linear improved vs Phase A 30k subset

## 0.1.0 — Phase A

- Initial package: linear boundary model, optional ONNX, Himotoki teacher dump pipeline
- 100k Wikipedia silver-label pipeline + frozen 5k holdout
