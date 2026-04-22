"""
Per-software evaluation with streaming feature extraction.

Usage:
  python linux/scripts/eval_per_software.py \
      --raw-dir data/raw/freecad_only \
      --output-dir models/freecad_final

Designed to handle large datasets (e.g. KiCad 18M events) within a 2GB container
limit by streaming file-by-file feature extraction.

Pipeline:
  1. Scan raw CSV files in --raw-dir
  2. Build feature vocabulary by streaming
  3. Build session features chunk-by-chunk
  4. Stratified train/test split at session level (test_size=0.3)
  5. Train RandomForest (base = sequence-only, context = sequence + context)
  6. Evaluate at session level using majority vote across chunks
  7. Save models, metrics, summary to --output-dir
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.syscall_anomaly.features import (
    NOISE_SYSCALLS,
    build_session_features,
    _ngrams,
    _top_keys,
)
from src.syscall_anomaly.models import (
    _align_features,
    make_random_forest_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-software syscall anomaly evaluation.")
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw per-session CSV files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for models and metrics.")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--syscall-limit", type=int, default=32)
    parser.add_argument("--transition-limit", type=int, default=64)
    parser.add_argument("--bigram-limit", type=int, default=64)
    parser.add_argument("--trigram-limit", type=int, default=64)
    parser.add_argument("--context-value-limit", type=int, default=16)
    return parser.parse_args()


def build_vocab_streaming(files: list[Path], args: argparse.Namespace) -> tuple[dict, int, int]:
    """Stream files to build global feature vocabulary without loading all data at once.

    Returns (vocab, raw_events_total, filtered_events_total).
    """
    sc = Counter(); tr = Counter(); bg = Counter(); tg = Counter()
    op = Counter(); pt = Counter(); ex = Counter(); rc = Counter()
    raw_total = 0
    filtered_total = 0
    for f in files:
        df = pd.read_csv(f)
        raw_total += len(df)
        df = df[~df["syscall_name"].isin(NOISE_SYSCALLS)]
        filtered_total += len(df)
        syscalls = df["syscall_name"].astype(str).tolist()
        sc.update(syscalls)
        tr.update(f"{a}->{b}" for a, b in zip(syscalls[:-1], syscalls[1:]))
        bg.update("__".join(t) for t in _ngrams(syscalls, 2))
        tg.update("__".join(t) for t in _ngrams(syscalls, 3))
        op.update(df["op_category"].fillna("unknown").astype(str))
        pt.update(df["path_type"].fillna("unknown").astype(str))
        ex.update(df["file_ext"].fillna("unknown").astype(str))
        rc.update(df["return_code"].fillna("unknown").astype(str))
        del df, syscalls
        gc.collect()
    vocab = {
        "syscalls": _top_keys(sc, args.syscall_limit),
        "transitions": _top_keys(tr, args.transition_limit),
        "bigrams": _top_keys(bg, args.bigram_limit),
        "trigrams": _top_keys(tg, args.trigram_limit),
        "ctx_ops": _top_keys(op, args.context_value_limit),
        "ctx_path_types": _top_keys(pt, args.context_value_limit),
        "ctx_exts": _top_keys(ex, args.context_value_limit),
        "ctx_rcs": _top_keys(rc, args.context_value_limit),
    }
    return vocab, raw_total, filtered_total


def build_features_streaming(files: list[Path], vocab: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream files to build session features without loading all data at once."""
    base_rows, ctx_rows = [], []
    for f in files:
        df = pd.read_csv(f)
        df = df[~df["syscall_name"].isin(NOISE_SYSCALLS)]
        base_rows.append(build_session_features(df, include_context=False, vocab=vocab))
        ctx_rows.append(build_session_features(df, include_context=True, vocab=vocab))
        del df
        gc.collect()
    base_df = pd.concat(base_rows, ignore_index=True).fillna(0.0)
    ctx_df = pd.concat(ctx_rows, ignore_index=True).fillna(0.0)
    del base_rows, ctx_rows
    gc.collect()
    return base_df, ctx_df


def stratified_split(feature_df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = feature_df.copy()
    feature_df["base_session"] = feature_df["session_id"].apply(lambda x: str(x).split("_chunk_")[0])
    session_meta = feature_df[["base_session", "software", "label"]].drop_duplicates()
    train_sessions, test_sessions = [], []
    for _, g in session_meta.groupby(["software", "label"], sort=True):
        s = g.sample(frac=1.0, random_state=random_state)
        tc = max(1, int(round(len(s) * test_size)))
        test_sessions.extend(s["base_session"].iloc[:tc].tolist())
        train_sessions.extend(s["base_session"].iloc[tc:].tolist())
    test_df = feature_df[feature_df["base_session"].isin(test_sessions)].drop(columns=["base_session"])
    train_df = feature_df[feature_df["base_session"].isin(train_sessions)].drop(columns=["base_session"])
    if train_df.empty or test_df.empty:
        raise SystemExit("Train/test split produced an empty partition.")
    return train_df, test_df


def evaluate_session_level(model: dict, test_df: pd.DataFrame) -> dict:
    """Session-level evaluation: majority vote across chunks."""
    x_test = _align_features(model, test_df)
    clf = model["sklearn_model"]
    chunk_pred = clf.predict(x_test).astype(np.int32)
    chunk_score = clf.predict_proba(x_test)[:, 1]

    df_out = pd.DataFrame({
        "base_session": test_df["session_id"].apply(lambda x: str(x).split("_chunk_")[0]).values,
        "label": (test_df["label"] == "abnormal").astype(int).values,
        "pred": chunk_pred,
        "score": chunk_score,
    })
    sess = df_out.groupby("base_session").agg(
        true_label=("label", "max"),
        n_chunks=("pred", "size"),
        n_abnormal=("pred", "sum"),
        mean_score=("score", "mean"),
    ).reset_index()
    sess["pred"] = (sess["n_abnormal"] / sess["n_chunks"] > 0.5).astype(int)

    y_true = sess["true_label"].to_numpy()
    y_pred = sess["pred"].to_numpy()
    y_score = sess["mean_score"].to_numpy()

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    n = max(len(y_true), 1)
    accuracy = float((tp + tn) / n)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
    fpr = float(fp / max(fp + tn, 1))

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        roc_auc = 0.5
    else:
        sorted_neg = np.sort(neg_scores)
        wins = 0.0
        for pos in pos_scores:
            left = np.searchsorted(sorted_neg, pos, side="left")
            right = np.searchsorted(sorted_neg, pos, side="right")
            wins += float(left) + 0.5 * float(right - left)
        roc_auc = float(wins / (len(pos_scores) * len(neg_scores)))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "total_sessions": float(n),
    }


def write_summary(output_dir: Path, software: str, metrics: dict, n_sessions: int, n_events: int) -> None:
    bm = metrics["base_model"]
    cm = metrics["context_model"]
    summary = "\n".join([
        f"# {software} Evaluation Summary",
        "",
        f"- Sessions: {n_sessions}",
        f"- Events (after noise filter): {n_events:,}",
        "",
        "## Sequence-only model (session-level)",
        f"- Accuracy:  {bm['accuracy']:.4f}",
        f"- Precision: {bm['precision']:.4f}",
        f"- Recall:    {bm['recall']:.4f}",
        f"- F1:        {bm['f1']:.4f}",
        f"- FPR:       {bm['fpr']:.4f}",
        f"- ROC-AUC:   {bm['roc_auc']:.4f}",
        f"- Confusion: TP={int(bm['tp'])} TN={int(bm['tn'])} FP={int(bm['fp'])} FN={int(bm['fn'])}",
        "",
        "## Sequence + context model (session-level)",
        f"- Accuracy:  {cm['accuracy']:.4f}",
        f"- Precision: {cm['precision']:.4f}",
        f"- Recall:    {cm['recall']:.4f}",
        f"- F1:        {cm['f1']:.4f}",
        f"- FPR:       {cm['fpr']:.4f}",
        f"- ROC-AUC:   {cm['roc_auc']:.4f}",
        f"- Confusion: TP={int(cm['tp'])} TN={int(cm['tn'])} FP={int(cm['fp'])} FN={int(cm['fn'])}",
        "",
    ])
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {raw_dir}")
    print(f"[1/5] Found {len(files)} raw CSV files in {raw_dir}")

    print("[2/5] Building vocabulary (streaming)...")
    vocab, n_events_raw, n_events_filtered = build_vocab_streaming(files, args)
    print(f"      Events raw={n_events_raw:,}, filtered={n_events_filtered:,} (dropped {n_events_raw - n_events_filtered:,} noise)")

    print("[3/5] Building features (streaming)...")
    base_df, ctx_df = build_features_streaming(files, vocab)

    meta_cols = ["session_id", "software", "scenario", "label"]
    n_base_feat = len([c for c in base_df.columns if c not in meta_cols])
    n_ctx_feat = len([c for c in ctx_df.columns if c not in meta_cols])
    n_chunks = len(base_df)
    software = str(base_df["software"].iloc[0])
    n_sessions = base_df["session_id"].apply(lambda x: str(x).split("_chunk_")[0]).nunique()
    print(f"      Software={software}, Sessions={n_sessions}, Chunks={n_chunks}, BaseFeat={n_base_feat}, CtxFeat={n_ctx_feat}")

    print("[4/5] Training RandomForest...")
    base_train, base_test = stratified_split(base_df, args.test_size, args.random_state)
    ctx_train, ctx_test = stratified_split(ctx_df, args.test_size, args.random_state)
    del base_df, ctx_df
    gc.collect()

    base_model = make_random_forest_model(
        base_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    ctx_model = make_random_forest_model(
        ctx_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )

    print("[5/5] Evaluating (session-level majority vote)...")
    base_metrics = evaluate_session_level(base_model, base_test)
    ctx_metrics = evaluate_session_level(ctx_model, ctx_test)

    metrics = {
        "software": software,
        "n_sessions_total": n_sessions,
        "n_chunks_total": n_chunks,
        "n_base_features": n_base_feat,
        "n_context_features": n_ctx_feat,
        "test_size": args.test_size,
        "model_config": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "class_weight": "balanced",
            "random_state": args.random_state,
        },
        "evaluation_method": "session_level_majority_vote",
        "n_events_raw": n_events_raw,
        "n_events_after_filter": n_events_filtered,
        "base_model": base_metrics,
        "context_model": ctx_metrics,
        "accuracy_gain": ctx_metrics["accuracy"] - base_metrics["accuracy"],
        "fpr_drop": base_metrics["fpr"] - ctx_metrics["fpr"],
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    joblib.dump(base_model["sklearn_model"], output_dir / "base_model.joblib")
    joblib.dump(ctx_model["sklearn_model"], output_dir / "context_model.joblib")
    (output_dir / "base_model_features.json").write_text(
        json.dumps(list(base_model["features"]), ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "context_model_features.json").write_text(
        json.dumps(list(ctx_model["features"]), ensure_ascii=False), encoding="utf-8"
    )
    write_summary(output_dir, software, metrics, n_sessions, n_events_filtered)

    print(f"\n=== {software} Results (session-level) ===")
    print(f"  Base model:    accuracy={base_metrics['accuracy']:.4f}  FPR={base_metrics['fpr']:.4f}  F1={base_metrics['f1']:.4f}")
    print(f"  Context model: accuracy={ctx_metrics['accuracy']:.4f}  FPR={ctx_metrics['fpr']:.4f}  F1={ctx_metrics['f1']:.4f}")
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    main()
