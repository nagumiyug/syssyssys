"""P4 helper: replay 7.1 evaluation pipeline (random_state=42) and dump
per-window predictions + 295-d feature vectors so we can inspect FP / FN
windows. The deterministic split + identical model config means the
chunk_metrics produced here MUST match models/section_7_1/7_1_evaluation_results.json
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.syscall_anomaly.models import make_random_forest_model, _align_features
from linux.scripts.eval_per_software import (
    build_vocab_streaming,
    build_features_streaming,
    stratified_split,
)


def calc(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def run_one(raw_dir: Path, name: str, args, out_dir: Path):
    print(f"\n===== {name} =====")
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"no csv in {raw_dir}")
    vocab, raw_n, filt_n = build_vocab_streaming(files, args)
    _, ctx_df = build_features_streaming(files, vocab)
    train_df, test_df = stratified_split(ctx_df, args.test_size, args.random_state)
    del ctx_df
    gc.collect()

    model = make_random_forest_model(
        train_df,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    x_test = _align_features(model, test_df)
    clf = model["sklearn_model"]
    chunk_pred = clf.predict(x_test).astype(np.int32)
    chunk_proba = clf.predict_proba(x_test)[:, 1]
    chunk_true = (test_df["label"] == "abnormal").astype(int).to_numpy()

    tp, tn, fp, fn = calc(chunk_true, chunk_pred)
    print(f"window-level: tp={tp} tn={tn} fp={fp} fn={fn}")

    feature_cols = [c for c in test_df.columns
                    if c not in {"session_id", "label", "software"}]
    out = pd.DataFrame({
        "session_id": test_df["session_id"].values,
        "base_session": test_df["session_id"].apply(
            lambda x: str(x).split("_chunk_")[0]).values,
        "chunk_idx": test_df["session_id"].apply(
            lambda x: int(str(x).split("_chunk_")[1])
            if "_chunk_" in str(x) else -1).values,
        "y_true": chunk_true,
        "y_pred": chunk_pred,
        "y_proba": chunk_proba,
        "label": test_df["label"].values,
    })
    feat_block = test_df[feature_cols].reset_index(drop=True)
    out = pd.concat([out.reset_index(drop=True), feat_block], axis=1)

    out_path = out_dir / f"{name.lower()}_per_window.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({out.shape[0]} rows, {out.shape[1]} cols)")

    # session-level vote
    sess = out.groupby("base_session").agg(
        y_true=("y_true", "max"),
        n_chunks=("y_pred", "size"),
        n_abnormal=("y_pred", "sum"),
        mean_proba=("y_proba", "mean"),
    ).reset_index()
    sess["y_pred"] = (sess["n_abnormal"] / sess["n_chunks"] > 0.5).astype(int)
    s_path = out_dir / f"{name.lower()}_per_session.csv"
    sess.to_csv(s_path, index=False)
    print(f"wrote {s_path} ({sess.shape[0]} rows)")
    return {
        "window": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "session": calc(sess["y_true"].to_numpy(), sess["y_pred"].to_numpy()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freecad-dir", default="data/raw/freecad_only")
    p.add_argument("--kicad-dir", default="data/raw/kicad_only")
    p.add_argument("--output-dir", default="models/section_7_1/per_window_dump")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=9)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--syscall-limit", type=int, default=32)
    p.add_argument("--transition-limit", type=int, default=64)
    p.add_argument("--bigram-limit", type=int, default=64)
    p.add_argument("--trigram-limit", type=int, default=64)
    p.add_argument("--context-value-limit", type=int, default=16)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "FreeCAD": run_one(Path(args.freecad_dir), "FreeCAD", args, out),
        "KiCad":   run_one(Path(args.kicad_dir),   "KiCad",   args, out),
    }
    (out / "p4_dump_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
