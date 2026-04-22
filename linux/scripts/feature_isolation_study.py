#!/usr/bin/env python3
"""
Feature isolation study for Linux syscall anomaly detection.

Tests each feature group independently to quantify individual contributions
and computes per-anomaly-type recall.

Feature groups:
  scalars  : syscall_entropy, transition_entropy, self_loop_ratio
  freq     : freq_count::*
  seq      : seq2_norm::* + seq3_norm::*
  trans    : trans_norm::*
  ctx      : ctx_path_type::*
  fusion   : all of the above combined (with noise filter)

Usage:
  python feature_isolation_study.py \
      --raw-dir data/raw/freecad_only \
      --output-dir models/isolation/freecad

  python feature_isolation_study.py \
      --raw-dir data/raw/kicad_only \
      --output-dir models/isolation/kicad
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split

# ── constants ────────────────────────────────────────────────────────────────

NOISE_SYSCALLS = {"mmap", "munmap", "mprotect"}
CHUNK_SIZE = 1000
NORMAL_LABEL = "normal"

ISOLATION_MODES = ["scalars", "freq", "seq", "trans", "ctx", "fusion"]

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=9,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# ── data loading ─────────────────────────────────────────────────────────────

def list_csvs(raw_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        sys.exit(f"No CSV files found in {raw_dir}")
    return paths


def load_csv_filtered(path: str) -> pd.DataFrame:
    """Load one session CSV and drop noise syscalls."""
    df = pd.read_csv(
        path,
        usecols=["syscall_name", "path_type", "label", "session_id", "scenario"],
        dtype={"syscall_name": str, "path_type": str, "label": str,
               "session_id": str, "scenario": str},
    )
    df = df[~df["syscall_name"].isin(NOISE_SYSCALLS)].reset_index(drop=True)
    return df


def _scenario_to_subtype(scenario: str) -> str:
    """'bulk_export_linux_r1' → 'bulk_export'"""
    return re.sub(r"_linux_r\d+$", "", scenario)


# ── vocabulary building ───────────────────────────────────────────────────────

def build_vocab(csv_paths: list[str]) -> dict:
    """
    First pass: collect unique syscalls, bigrams, trigrams, path_types
    across all sessions. Processes one CSV at a time to keep memory low.
    """
    syscalls: set = set()
    bigrams: set = set()
    trigrams: set = set()
    path_types: set = set()

    for p in csv_paths:
        df = load_csv_filtered(p)
        syscalls.update(df["syscall_name"].unique())
        path_types.update(df["path_type"].dropna().unique())

        seq = df["syscall_name"].tolist()
        for i in range(len(seq) - 1):
            bigrams.add((seq[i], seq[i + 1]))
        for i in range(len(seq) - 2):
            trigrams.add((seq[i], seq[i + 1], seq[i + 2]))

        # help GC reclaim memory early
        del df

    return {
        "syscalls": sorted(syscalls),
        "bigrams": sorted(bigrams),
        "trigrams": sorted(trigrams),
        "path_types": sorted(path_types),
    }


# ── feature extraction ────────────────────────────────────────────────────────

def _entropy(counts_dict: Counter) -> float:
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts_dict.values()), dtype=float) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def extract_chunk_features(chunk: pd.DataFrame, vocab: dict) -> dict:
    """Return feature dict for one chunk (all groups)."""
    seq = chunk["syscall_name"].tolist()
    n = len(seq)

    sc_counts = Counter(seq)
    bg_counts = Counter()
    for i in range(n - 1):
        bg_counts[(seq[i], seq[i + 1])] += 1
    tg_counts = Counter()
    for i in range(n - 2):
        tg_counts[(seq[i], seq[i + 1], seq[i + 2])] += 1

    bg_total = sum(bg_counts.values())
    tg_total = sum(tg_counts.values())

    # scalars
    feat: dict = {
        "syscall_entropy": _entropy(sc_counts),
        "transition_entropy": _entropy(bg_counts),
        "self_loop_ratio": (
            sum(1 for i in range(n - 1) if seq[i] == seq[i + 1]) / (n - 1)
            if n > 1 else 0.0
        ),
    }

    # freq
    for s in vocab["syscalls"]:
        feat[f"freq_count::{s}"] = sc_counts.get(s, 0) / n

    # seq (bigram + trigram, normalised by respective totals)
    for b in vocab["bigrams"]:
        feat[f"seq2_norm::{b[0]}->{b[1]}"] = bg_counts.get(b, 0) / max(bg_total, 1)
    for t in vocab["trigrams"]:
        feat[f"seq3_norm::{t[0]}->{t[1]}->{t[2]}"] = tg_counts.get(t, 0) / max(tg_total, 1)

    # trans (P(B|A) = count(A→B) / count(A))
    for b in vocab["bigrams"]:
        a_cnt = sc_counts.get(b[0], 0)
        feat[f"trans_norm::{b[0]}->{b[1]}"] = bg_counts.get(b, 0) / max(a_cnt, 1)

    # ctx (path_type fractions)
    pt_counts = Counter(chunk["path_type"].dropna().tolist())
    for pt in vocab["path_types"]:
        feat[f"ctx_path_type::{pt}"] = pt_counts.get(pt, 0) / n

    return feat


def build_feature_matrix(csv_paths: list[str], vocab: dict) -> pd.DataFrame:
    """
    Second pass: build full feature matrix (one row per chunk).
    Adds meta-columns: session_id, label, anomaly_subtype.
    """
    rows = []

    for p in csv_paths:
        df = load_csv_filtered(p)
        if df.empty:
            continue

        session_id = str(df["session_id"].iloc[0])
        label = str(df["label"].iloc[0])           # 'normal' or 'abnormal'
        scenario = str(df["scenario"].iloc[0])
        is_abnormal = int(label != NORMAL_LABEL)
        # Use label as subtype for normal sessions to keep the filter consistent
        subtype = _scenario_to_subtype(scenario) if is_abnormal else NORMAL_LABEL

        # chunk the session; range guarantees each slice is exactly CHUNK_SIZE
        for start in range(0, len(df) - CHUNK_SIZE + 1, CHUNK_SIZE):
            chunk = df.iloc[start: start + CHUNK_SIZE]
            feat = extract_chunk_features(chunk, vocab)
            feat["session_id"] = session_id
            feat["label"] = is_abnormal
            feat["anomaly_subtype"] = subtype      # per-anomaly-type recall key
            rows.append(feat)

        del df

    return pd.DataFrame(rows)


# ── column selection by isolation mode ───────────────────────────────────────

def select_columns(df: pd.DataFrame, mode: str) -> list[str]:
    all_cols = df.columns.tolist()
    meta = {"session_id", "label", "anomaly_subtype"}

    if mode == "scalars":
        return [c for c in ["syscall_entropy", "transition_entropy", "self_loop_ratio"]
                if c in all_cols]
    if mode == "freq":
        return [c for c in all_cols if c.startswith("freq_count::")]
    if mode == "seq":
        return [c for c in all_cols if c.startswith("seq2_norm::") or c.startswith("seq3_norm::")]
    if mode == "trans":
        return [c for c in all_cols if c.startswith("trans_norm::")]
    if mode == "ctx":
        return [c for c in all_cols if c.startswith("ctx_path_type::")]
    if mode == "fusion":
        return [c for c in all_cols if c not in meta]
    raise ValueError(f"Unknown mode: {mode}")


# ── evaluation ────────────────────────────────────────────────────────────────

def session_level_eval(y_true_chunk, y_pred_chunk, session_ids, threshold=0.5):
    """Aggregate chunk predictions to session level via majority vote."""
    session_true = defaultdict(list)
    session_pred = defaultdict(list)
    for sid, yt, yp in zip(session_ids, y_true_chunk, y_pred_chunk):
        session_true[sid].append(yt)
        session_pred[sid].append(yp)

    y_true_s, y_pred_s = [], []
    for sid in session_true:
        y_true_s.append(int(any(v == 1 for v in session_true[sid])))
        vote = sum(session_pred[sid]) / len(session_pred[sid])
        y_pred_s.append(int(vote >= threshold))
    return np.array(y_true_s), np.array(y_pred_s)


def compute_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def per_anomaly_recall(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    subtypes: list[str],
) -> dict[str, float]:
    """
    For each anomaly subtype, compute recall on the test chunks belonging to that subtype.
    Normal chunks are excluded (recall is only defined for positive classes).
    """
    result = {}
    for st in subtypes:
        mask = test_df["anomaly_subtype"] == st
        if mask.sum() == 0:
            continue
        y_true_sub = test_df.loc[mask, "label"].values
        y_pred_sub = y_pred[mask.values]
        result[st] = float(recall_score(y_true_sub, y_pred_sub, zero_division=0))
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feature isolation study")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = list_csvs(args.raw_dir)
    print(f"Found {len(csv_paths)} CSV files in {args.raw_dir}")

    # ── vocab ──
    print("Pass 1: building vocabulary …")
    vocab = build_vocab(csv_paths)
    print(f"  syscalls={len(vocab['syscalls'])}  bigrams={len(vocab['bigrams'])}  "
          f"trigrams={len(vocab['trigrams'])}  path_types={len(vocab['path_types'])}")

    # ── feature matrix ──
    print("Pass 2: building feature matrix …")
    df = build_feature_matrix(csv_paths, vocab)
    print(f"  shape={df.shape}  sessions={df['session_id'].nunique()}")

    # ── train/test split at session level ──
    sessions = df[["session_id", "label"]].drop_duplicates()
    train_sids, test_sids = train_test_split(
        sessions["session_id"],
        test_size=args.test_size,
        stratify=sessions["label"],
        random_state=args.random_state,
    )
    train_df = df[df["session_id"].isin(train_sids)].reset_index(drop=True)
    test_df  = df[df["session_id"].isin(test_sids)].reset_index(drop=True)
    print(f"  train chunks={len(train_df)}  test chunks={len(test_df)}")

    anomaly_subtypes = [st for st in test_df["anomaly_subtype"].unique() if st != NORMAL_LABEL]
    print(f"  anomaly subtypes in test: {anomaly_subtypes}")

    # ── run isolation experiments ──
    results = []
    feature_importances = {}  # only saved for fusion mode

    for mode in ISOLATION_MODES:
        feat_cols = select_columns(train_df, mode)
        if not feat_cols:
            print(f"  [{mode}] no columns — skipping")
            continue
        print(f"  [{mode}] {len(feat_cols)} features …", end="", flush=True)

        X_train = train_df[feat_cols].values
        y_train = train_df["label"].values
        X_test  = test_df[feat_cols].values
        y_test  = test_df["label"].values

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # chunk-level metrics
        chunk_metrics = compute_metrics(y_test, y_pred)

        # session-level metrics
        y_true_s, y_pred_s = session_level_eval(
            y_test, y_pred, test_df["session_id"].values
        )
        session_metrics = compute_metrics(y_true_s, y_pred_s)

        # per-anomaly recall (chunk level)
        per_type = per_anomaly_recall(test_df, y_pred, anomaly_subtypes)

        row = {
            "mode": mode,
            "n_features": len(feat_cols),
            **{f"chunk_{k}": v for k, v in chunk_metrics.items()},
            **{f"session_{k}": v for k, v in session_metrics.items()},
            **{f"recall_{st}": per_type.get(st, float("nan")) for st in anomaly_subtypes},
        }
        results.append(row)

        # save feature importance for fusion
        if mode == "fusion":
            imp = pd.DataFrame({
                "feature": feat_cols,
                "importance": clf.feature_importances_,
            }).sort_values("importance", ascending=False)
            feature_importances = {
                "top20": imp.head(20)[["feature", "importance"]].to_dict(orient="records"),
                "by_group": _importance_by_group(imp),
            }

        acc_chunk = chunk_metrics["accuracy"]
        acc_sess  = session_metrics["accuracy"]
        print(f" chunk_acc={acc_chunk:.3f}  session_acc={acc_sess:.3f}")

    # ── save results ──
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(args.output_dir, "isolation_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved → {results_csv}")

    summary = {
        "modes": ISOLATION_MODES,
        "anomaly_subtypes": anomaly_subtypes,
        "n_train_chunks": len(train_df),
        "n_test_chunks": len(test_df),
        "n_train_sessions": len(train_sids),
        "n_test_sessions": len(test_sids),
        "results": results_df.to_dict(orient="records"),
        "feature_importances": feature_importances,
    }
    summary_json = os.path.join(args.output_dir, "isolation_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {summary_json}")

    # ── console table ──
    print("\n" + "=" * 70)
    print(f"{'Mode':<10} {'#Feat':>6} {'Chunk Acc':>10} {'Sess Acc':>10}", end="")
    for st in anomaly_subtypes:
        print(f"  {'R:' + st[:10]:>14}", end="")
    print()
    print("-" * 70)
    for r in results:
        line = (f"{r['mode']:<10} {r['n_features']:>6} "
                f"{r['chunk_accuracy']:>10.3f} {r['session_accuracy']:>10.3f}")
        for st in anomaly_subtypes:
            v = r.get(f"recall_{st}", float("nan"))
            line += f"  {v:>14.3f}"
        print(line)
    print("=" * 70)


def _importance_by_group(imp_df: pd.DataFrame) -> dict:
    """Sum feature importances by group prefix."""
    groups = {
        "scalars": ["syscall_entropy", "transition_entropy", "self_loop_ratio"],
        "freq": "freq_count::",
        "seq": ("seq2_norm::", "seq3_norm::"),
        "trans": "trans_norm::",
        "ctx": "ctx_path_type::",
    }
    result = {}
    for grp, prefix in groups.items():
        if isinstance(prefix, list):
            mask = imp_df["feature"].isin(prefix)
        elif isinstance(prefix, tuple):
            mask = imp_df["feature"].apply(lambda c: any(c.startswith(p) for p in prefix))
        else:
            mask = imp_df["feature"].str.startswith(prefix)
        result[grp] = float(imp_df.loc[mask, "importance"].sum())
    return result


if __name__ == "__main__":
    main()
