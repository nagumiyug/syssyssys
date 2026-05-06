"""P5 step 2: empirically measure capture & inference performance.

Two parts:

PART A — derive REAL kernel→user-space throughput from existing capture CSVs.
  Each CSV has per-event timestamp_ns from the eBPF tracepoint, so:
    events·sec = N / (max_ts_ns - min_ts_ns) * 1e9
    bytes·sec  = file_size / wall_duration
    mean syscall handling cost = mean(duration_us)  (kernel-side per-syscall time)
  Aggregated across scenarios per software.

PART B — measure user-space INFERENCE cost on this hardware:
  Load one CSV, run F5 feature extraction + RF predict_proba on the deployed
  pipeline, sample CPU% and RSS during the run. Records per-window inference
  latency (the cost incurred at deployment when each new window is scored).

All numbers are reproducible by re-running this script. Output JSON +
per-scenario CSV to /root/s428/models/deployment/.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median

import pandas as pd
import psutil

ROOT = Path("/root/s428")
sys.path.insert(0, str(ROOT))


def measure_csv_throughput(csv_path: Path) -> dict:
    """PART A: per-CSV throughput from timestamps."""
    file_size = csv_path.stat().st_size
    df = pd.read_csv(csv_path, usecols=["timestamp_ns", "duration_us", "syscall_name"])
    n = len(df)
    if n < 2:
        return {"path": str(csv_path), "events": n, "events_per_sec": 0,
                "bytes_per_sec": 0, "duration_s": 0, "mean_duration_us": 0,
                "p95_duration_us": 0}
    ts_min = int(df["timestamp_ns"].min())
    ts_max = int(df["timestamp_ns"].max())
    duration_s = (ts_max - ts_min) / 1e9
    events_per_sec = n / duration_s if duration_s > 0 else 0
    bytes_per_sec = file_size / duration_s if duration_s > 0 else 0
    durs = df["duration_us"].dropna().astype(float)
    return {
        "path": str(csv_path.name),
        "events": int(n),
        "duration_s": round(duration_s, 4),
        "events_per_sec": round(events_per_sec, 1),
        "file_size_kb": round(file_size / 1024, 1),
        "bytes_per_sec": round(bytes_per_sec, 1),
        "mean_duration_us": round(float(durs.mean()), 3) if len(durs) else 0,
        "p95_duration_us": round(float(durs.quantile(0.95)), 3) if len(durs) else 0,
    }


def measure_inference_cost(csv_path: Path) -> dict:
    """PART B: user-space inference cost on this hardware."""
    from src.syscall_anomaly.models import make_random_forest_model, _align_features
    from linux.scripts.eval_per_software import (
        build_vocab_streaming, build_features_streaming, stratified_split,
    )

    class Args:
        syscall_limit = 32
        transition_limit = 64
        bigram_limit = 64
        trigram_limit = 64
        context_value_limit = 16

    files = sorted(csv_path.parent.glob("*.csv"))
    proc = psutil.Process(os.getpid())

    def sample(stage):
        return {
            "stage": stage,
            "rss_mb": round(proc.memory_info().rss / 1024 / 1024, 1),
            "cpu_pct": proc.cpu_percent(interval=0.5),
        }

    samples = [sample("baseline")]

    t0 = time.perf_counter()
    vocab, raw_n, filt_n = build_vocab_streaming(files, Args)
    t_vocab = time.perf_counter() - t0
    samples.append(sample("after_vocab"))

    t0 = time.perf_counter()
    _, ctx_df = build_features_streaming(files, vocab)
    t_features = time.perf_counter() - t0
    samples.append(sample("after_features"))

    train_df, test_df = stratified_split(ctx_df, 0.3, 42)
    n_train = len(train_df)
    n_test = len(test_df)
    del ctx_df
    gc.collect()

    t0 = time.perf_counter()
    model = make_random_forest_model(train_df, n_estimators=300, max_depth=9,
                                     random_state=42)
    t_train = time.perf_counter() - t0
    samples.append(sample("after_train"))

    x_test = _align_features(model, test_df)
    clf = model["sklearn_model"]

    # per-window predict latency (loop over rows to amortize)
    n_iter = max(1, len(x_test))
    t0 = time.perf_counter()
    _ = clf.predict_proba(x_test)
    t_batch = time.perf_counter() - t0
    samples.append(sample("after_predict"))

    # single-window timing — repeat 50 times to get stable numbers
    single_x = x_test.iloc[[0]] if hasattr(x_test, "iloc") else x_test[:1]
    timings = []
    for _ in range(50):
        t0 = time.perf_counter()
        clf.predict_proba(single_x)
        timings.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "n_train_windows": int(n_train),
        "n_test_windows": int(n_test),
        "events_raw": int(raw_n),
        "events_filtered": int(filt_n),
        "vocab_build_s": round(t_vocab, 3),
        "feature_build_s": round(t_features, 3),
        "train_s": round(t_train, 3),
        "batch_predict_s": round(t_batch, 4),
        "batch_predict_us_per_window": round(t_batch / max(n_iter, 1) * 1e6, 1),
        "single_window_predict_ms_p50": round(median(timings), 4),
        "single_window_predict_ms_p95": round(sorted(timings)[int(0.95 * len(timings))], 4),
        "single_window_predict_ms_mean": round(mean(timings), 4),
        "samples": samples,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freecad-dir", default="/root/s428/data/raw/freecad_only")
    p.add_argument("--kicad-dir", default="/root/s428/data/raw/kicad_only")
    p.add_argument("--output-dir", default="/root/s428/models/deployment")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # PART A: per-CSV throughput
    a_rows = []
    for sw_name, sw_dir in [("FreeCAD", Path(args.freecad_dir)),
                            ("KiCad", Path(args.kicad_dir))]:
        csvs = sorted(sw_dir.glob("*.csv"))
        for c in csvs:
            try:
                m = measure_csv_throughput(c)
                m["software"] = sw_name
                a_rows.append(m)
            except Exception as e:
                print(f"  skip {c.name}: {e}")
    df_a = pd.DataFrame(a_rows)
    df_a.to_csv(out / "throughput_per_session.csv", index=False)
    summary_a = {}
    for sw, g in df_a.groupby("software"):
        summary_a[sw] = {
            "n_sessions": int(len(g)),
            "events_per_sec": {
                "min": round(float(g.events_per_sec.min()), 1),
                "p50": round(float(g.events_per_sec.median()), 1),
                "p95": round(float(g.events_per_sec.quantile(0.95)), 1),
                "max": round(float(g.events_per_sec.max()), 1),
            },
            "bytes_per_sec_kb": {
                "p50": round(float(g.bytes_per_sec.median()) / 1024, 1),
                "p95": round(float(g.bytes_per_sec.quantile(0.95)) / 1024, 1),
            },
            "mean_syscall_duration_us": {
                "p50": round(float(g.mean_duration_us.median()), 3),
                "p95": round(float(g.p95_duration_us.median()), 3),
            },
            "session_length_s": {
                "p50": round(float(g.duration_s.median()), 2),
                "max": round(float(g.duration_s.max()), 2),
            },
        }

    # PART B: inference cost on FreeCAD + KiCad
    summary_b = {}
    for sw_name, sw_dir in [("FreeCAD", Path(args.freecad_dir)),
                            ("KiCad", Path(args.kicad_dir))]:
        print(f"\n=== inference benchmark on {sw_name} ===")
        try:
            summary_b[sw_name] = measure_inference_cost(sw_dir)
        except Exception as e:
            summary_b[sw_name] = {"error": str(e)}
            print(f"  failed: {e}")

    out_json = out / "capture_overhead_results.json"
    out_json.write_text(json.dumps({
        "throughput_summary": summary_a,
        "inference_benchmark": summary_b,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}")

    print("\n=== throughput summary ===")
    print(json.dumps(summary_a, indent=2))
    print("\n=== inference benchmark summary ===")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                      for k, v in summary_b.items()}, indent=2))


if __name__ == "__main__":
    main()
