#!/usr/bin/env python3
"""Compare linear vs ONNX on full holdout and clean slice → phase_b metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_eval(model: Path, holdout: Path, clean: Path | None, out: Path, backend: str) -> dict:
    cmd = [
        sys.executable,
        "scripts/eval_model.py",
        "-m",
        str(model),
        "-i",
        str(holdout),
        "-o",
        str(out),
        "--backend",
        backend,
    ]
    if clean is not None:
        cmd.extend(["--clean", str(clean)])
    subprocess.check_call(cmd)
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--linear",
        type=Path,
        default=Path("himotoki_split/models/default.npz"),
    )
    ap.add_argument(
        "--onnx",
        type=Path,
        default=Path("himotoki_split/models/default.onnx"),
    )
    ap.add_argument(
        "--holdout",
        type=Path,
        default=Path("data/labels/holdout.jsonl"),
    )
    ap.add_argument(
        "--clean",
        type=Path,
        default=Path("data/labels/holdout_clean.jsonl"),
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output/eval_metrics_phase_b.json"),
    )
    args = ap.parse_args()

    clean = args.clean if args.clean.is_file() else None
    linear = _run_eval(
        args.linear,
        args.holdout,
        clean,
        Path("output/eval_metrics_phase_b_linear.json"),
        "linear",
    )
    onnx = _run_eval(
        args.onnx,
        args.holdout,
        clean,
        Path("output/eval_metrics_phase_b_onnx.json"),
        "onnx",
    )

    prefer = "onnx"
    # Prefer higher clean exact-seg, then clean F1, then full holdout F1
    def score(m: dict) -> tuple:
        c = m.get("clean") or {}
        return (
            c.get("exact_seg", 0.0),
            c.get("boundary_f1", 0.0),
            m.get("exact_seg", 0.0),
            m.get("boundary_f1", 0.0),
        )

    if score(linear) > score(onnx):
        prefer = "linear"

    summary = {
        "linear": linear,
        "onnx": onnx,
        "prefer_runtime": prefer,
        "success_bar": {
            "exact_seg_ge_0_20": onnx.get("exact_seg", 0) >= 0.20,
            "boundary_f1_ge_0_94": onnx.get("boundary_f1", 0) >= 0.94,
            "met": onnx.get("exact_seg", 0) >= 0.20
            or onnx.get("boundary_f1", 0) >= 0.94,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
