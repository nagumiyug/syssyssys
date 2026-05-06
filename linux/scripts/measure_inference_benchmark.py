"""P5 Part B (retry): user-space inference benchmark on this hardware.

Use the per-window CSVs already generated in P4 (section_7_1/per_window_dump/).
Drop non-feature columns, train a fresh RF with the same hyperparams used in
6.2/run_7_1_evaluation, then time predict_proba on single windows + batches.
This isolates the deployed-side inference cost without re-running the full
vocab/feature pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier

NON_FEATURE = {"session_id", "base_session", "chunk_idx", "y_true", "y_pred",
               "y_proba", "label", "scenario", "software"}


def benchmark_software(per_window_csv: Path) -> dict:
    df = pd.read_csv(per_window_csv)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    X = df[feat_cols].values
    y = df["y_true"].values

    proc = psutil.Process(os.getpid())

    def sample(stage):
        return {
            "stage": stage,
            "rss_mb": round(proc.memory_info().rss / 1024 / 1024, 1),
            "cpu_pct": proc.cpu_percent(interval=0.5),
        }

    samples = [sample("baseline")]

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    split = int(0.7 * len(X))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test = X[test_idx]

    # train (matching 6.2 hyperparams)
    t0 = time.perf_counter()
    clf = RandomForestClassifier(n_estimators=300, max_depth=9,
                                 class_weight="balanced", random_state=42,
                                 n_jobs=1)
    clf.fit(X_train, y_train)
    t_train = time.perf_counter() - t0
    samples.append(sample("after_train"))

    # batch predict
    t0 = time.perf_counter()
    _ = clf.predict_proba(X_test)
    t_batch = time.perf_counter() - t0
    samples.append(sample("after_batch_predict"))

    # single-window timing — repeat 200 times
    timings = []
    for i in range(200):
        x_one = X_test[i % len(X_test):i % len(X_test) + 1]
        t0 = time.perf_counter()
        clf.predict_proba(x_one)
        timings.append((time.perf_counter() - t0) * 1000)  # ms
    samples.append(sample("after_single_predict"))

    # n-jobs=4 batch comparison (deployed multi-core)
    clf4 = RandomForestClassifier(n_estimators=300, max_depth=9,
                                  class_weight="balanced", random_state=42,
                                  n_jobs=4)
    clf4.fit(X_train, y_train)
    t0 = time.perf_counter()
    _ = clf4.predict_proba(X_test)
    t_batch4 = time.perf_counter() - t0

    return {
        "n_features": len(feat_cols),
        "n_total_windows": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_s": round(t_train, 3),
        "batch_predict_s": round(t_batch, 4),
        "batch_predict_n_jobs1_us_per_window": round(t_batch / max(len(X_test), 1) * 1e6, 1),
        "batch_predict_n_jobs4_us_per_window": round(t_batch4 / max(len(X_test), 1) * 1e6, 1),
        "single_window_predict_ms_p50": round(median(timings), 4),
        "single_window_predict_ms_p95": round(sorted(timings)[int(0.95 * len(timings))], 4),
        "single_window_predict_ms_mean": round(mean(timings), 4),
        "samples": samples,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freecad-csv",
                   default="/root/s428/models/section_7_1/per_window_dump/freecad_per_window.csv")
    p.add_argument("--kicad-csv",
                   default="/root/s428/models/section_7_1/per_window_dump/kicad_per_window.csv")
    p.add_argument("--output",
                   default="/root/s428/models/deployment/inference_benchmark_results.json")
    args = p.parse_args()

    out = {}
    for sw, csv_p in [("FreeCAD", Path(args.freecad_csv)),
                      ("KiCad",   Path(args.kicad_csv))]:
        if not csv_p.exists():
            print(f"  skip {sw}: {csv_p} not found")
            continue
        print(f"\n=== inference benchmark on {sw} ===")
        out[sw] = benchmark_software(csv_p)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                      for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
