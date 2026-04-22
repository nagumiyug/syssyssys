#!/usr/bin/env python3
"""
Parameter sensitivity sweep for Linux syscall anomaly detection.

Sweeps chunk_size and voting_threshold independently to justify the
production choices of chunk_size=1000 and threshold=0.5.

Grid:
  chunk_size  ∈ {500, 750, 1000, 1500, 2000}
  threshold   ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
  = 25 configurations × 2 software = 50 experiments

Output per software:
  param_sweep_results.csv  – one row per (chunk_size, threshold)
  param_sweep_summary.json – structured results for visualisation

Usage:
  python param_sweep.py --raw-dir data/raw/freecad_only --output-dir models/param_sweep/freecad
  python param_sweep.py --raw-dir data/raw/kicad_only   --output-dir models/param_sweep/kicad
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ── constants ────────────────────────────────────────────────────────────────

NOISE_SYSCALLS = {"mmap", "munmap", "mprotect"}
NORMAL_LABEL = "normal"

CHUNK_SIZES  = [500, 750, 1000, 1500, 2000]
THRESHOLDS   = [0.3, 0.4, 0.5, 0.6, 0.7]

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
        usecols=["syscall_name", "path_type", "label", "session_id"],
        dtype={"syscall_name": str, "path_type": str, "label": str, "session_id": str},
    )
    return df[~df["syscall_name"].isin(NOISE_SYSCALLS)].reset_index(drop=True)


# ── session metadata ──────────────────────────────────────────────────────────

def collect_session_meta(csv_paths: list[str]) -> pd.DataFrame:
    """Return DataFrame with one row per session: session_id, label, path."""
    rows = []
    for p in csv_paths:
        df = pd.read_csv(p, usecols=["label", "session_id"],
                         dtype=str, nrows=1)
        rows.append({
            "session_id": str(df["session_id"].iloc[0]),
            "label": str(df["label"].iloc[0]),
            "path": p,
        })
    return pd.DataFrame(rows)


# ── vocabulary building ───────────────────────────────────────────────────────

def build_vocab(csv_paths: list[str]) -> dict:
    """
    Collect unique syscalls, bigrams, trigrams, path_types across all sessions.
    Vocab is built at session-sequence level; all resulting feature columns
    will be present regardless of chunk_size.
    """
    syscalls: set = set()
    bigrams: set  = set()
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
        del df

    return {
        "syscalls":    sorted(syscalls),
        "bigrams":     sorted(bigrams),
        "trigrams":    sorted(trigrams),
        "path_types":  sorted(path_types),
    }


# ── feature extraction ────────────────────────────────────────────────────────

def _entropy(counts_dict: Counter) -> float:
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts_dict.values()), dtype=float) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _get_feature_names(vocab: dict) -> list[str]:
    """Return canonical ordered list of feature column names."""
    names = ["syscall_entropy", "transition_entropy", "self_loop_ratio"]
    names += [f"freq_count::{s}" for s in vocab["syscalls"]]
    names += [f"seq2_norm::{b[0]}->{b[1]}" for b in vocab["bigrams"]]
    names += [f"seq3_norm::{t[0]}->{t[1]}->{t[2]}" for t in vocab["trigrams"]]
    names += [f"trans_norm::{b[0]}->{b[1]}" for b in vocab["bigrams"]]
    names += [f"ctx_path_type::{pt}" for pt in vocab["path_types"]]
    return names


def _extract_np(chunk: pd.DataFrame, vocab: dict,
                feat_idx: dict, n_feats: int) -> np.ndarray:
    """Return 1-D float64 array aligned to feat_idx (memory-efficient)."""
    arr = np.zeros(n_feats, dtype=np.float64)
    seq = chunk["syscall_name"].tolist()
    n = len(seq)

    sc_counts = Counter(seq)
    bg_counts: Counter = Counter()
    for i in range(n - 1):
        bg_counts[(seq[i], seq[i + 1])] += 1
    tg_counts: Counter = Counter()
    for i in range(n - 2):
        tg_counts[(seq[i], seq[i + 1], seq[i + 2])] += 1

    bg_total = sum(bg_counts.values())
    tg_total = sum(tg_counts.values())

    arr[feat_idx["syscall_entropy"]]    = _entropy(sc_counts)
    arr[feat_idx["transition_entropy"]] = _entropy(bg_counts)
    arr[feat_idx["self_loop_ratio"]]    = (
        sum(1 for i in range(n - 1) if seq[i] == seq[i + 1]) / (n - 1)
        if n > 1 else 0.0
    )
    for s in vocab["syscalls"]:
        arr[feat_idx[f"freq_count::{s}"]] = sc_counts.get(s, 0) / n
    for b in vocab["bigrams"]:
        arr[feat_idx[f"seq2_norm::{b[0]}->{b[1]}"]] = bg_counts.get(b, 0) / max(bg_total, 1)
    for t in vocab["trigrams"]:
        arr[feat_idx[f"seq3_norm::{t[0]}->{t[1]}->{t[2]}"]] = (
            tg_counts.get(t, 0) / max(tg_total, 1)
        )
    for b in vocab["bigrams"]:
        a_cnt = sc_counts.get(b[0], 0)
        arr[feat_idx[f"trans_norm::{b[0]}->{b[1]}"]] = (
            bg_counts.get(b, 0) / max(a_cnt, 1)
        )
    pt_counts = Counter(chunk["path_type"].dropna().tolist())
    for pt in vocab["path_types"]:
        arr[feat_idx[f"ctx_path_type::{pt}"]] = pt_counts.get(pt, 0) / n

    return arr


def build_feature_matrix(csv_paths: list[str], vocab: dict, chunk_size: int) -> pd.DataFrame:
    """
    Build feature matrix using numpy row accumulation.
    ~7× lower peak memory vs list-of-dicts (critical for KiCad in 2GB cgroup).
    """
    feat_names = _get_feature_names(vocab)
    feat_idx   = {name: i for i, name in enumerate(feat_names)}
    n_feats    = len(feat_names)

    arrays:      list[np.ndarray] = []
    session_ids: list[str]        = []
    labels:      list[int]        = []

    for p in csv_paths:
        df = load_csv_filtered(p)
        if df.empty:
            continue

        session_id  = str(df["session_id"].iloc[0])
        is_abnormal = int(df["label"].iloc[0] != NORMAL_LABEL)

        for start in range(0, len(df) - chunk_size + 1, chunk_size):
            chunk = df.iloc[start: start + chunk_size]
            arrays.append(_extract_np(chunk, vocab, feat_idx, n_feats))
            session_ids.append(session_id)
            labels.append(is_abnormal)

        del df

    X      = np.vstack(arrays)   # single contiguous 2-D array
    result = pd.DataFrame(X, columns=feat_names)
    result["session_id"] = session_ids
    result["label"]      = labels
    return result


# ── evaluation ────────────────────────────────────────────────────────────────

def session_level_eval(y_true_chunk, y_pred_chunk, session_ids, threshold: float):
    """Majority-vote aggregation at session level."""
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


def compute_session_metrics(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    return {
        "session_accuracy": float(acc),
        "session_fpr": float(fpr),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parameter sensitivity sweep")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = list_csvs(args.raw_dir)
    print(f"Found {len(csv_paths)} CSV files in {args.raw_dir}")

    # ── fixed train/test session split ──
    # Split sessions once up front; chunk-level content differs per chunk_size
    # but the session assignment stays fixed across all experiments.
    meta = collect_session_meta(csv_paths)
    train_sids, test_sids = train_test_split(
        meta["session_id"],
        test_size=args.test_size,
        stratify=meta["label"],
        random_state=args.random_state,
    )
    train_paths = meta[meta["session_id"].isin(train_sids)]["path"].tolist()
    test_paths  = meta[meta["session_id"].isin(test_sids)]["path"].tolist()
    print(f"  train sessions={len(train_sids)}  test sessions={len(test_sids)}")

    # ── build vocab once (common to all chunk sizes) ──
    print("Building vocabulary …")
    vocab = build_vocab(csv_paths)
    print(f"  syscalls={len(vocab['syscalls'])}  bigrams={len(vocab['bigrams'])}  "
          f"trigrams={len(vocab['trigrams'])}")

    total = len(CHUNK_SIZES) * len(THRESHOLDS)
    done = 0
    results = []

    for chunk_size in CHUNK_SIZES:
        print(f"\nchunk_size={chunk_size}: building feature matrices …")
        train_df = build_feature_matrix(train_paths, vocab, chunk_size)
        test_df  = build_feature_matrix(test_paths,  vocab, chunk_size)

        feat_cols = [c for c in train_df.columns if c not in {"session_id", "label"}]
        X_train = train_df[feat_cols].values
        y_train = train_df["label"].values
        X_test  = test_df[feat_cols].values
        y_test  = test_df["label"].values
        sids    = test_df["session_id"].values

        print(f"  train_chunks={len(train_df)}  test_chunks={len(test_df)}  "
              f"features={len(feat_cols)}")

        # train once per chunk_size; only threshold varies
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_train, y_train)
        chunk_preds = clf.predict(X_test)

        for threshold in THRESHOLDS:
            y_true_s, y_pred_s = session_level_eval(y_test, chunk_preds, sids, threshold)
            metrics = compute_session_metrics(y_true_s, y_pred_s)

            row = {
                "chunk_size": chunk_size,
                "threshold": threshold,
                "n_train_chunks": len(train_df),
                "n_test_chunks": len(test_df),
                **metrics,
            }
            results.append(row)

            done += 1
            print(f"  threshold={threshold:.1f}  acc={metrics['session_accuracy']:.3f}  "
                  f"fpr={metrics['session_fpr']:.3f}   [{done}/{total}]")

        # release memory before next chunk_size
        del train_df, test_df, X_train, X_test

    # ── save ──
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "param_sweep_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    summary = {
        "chunk_sizes": CHUNK_SIZES,
        "thresholds": THRESHOLDS,
        "n_train_sessions": len(train_sids),
        "n_test_sessions": len(test_sids),
        "results": results_df.to_dict(orient="records"),
    }
    json_path = os.path.join(args.output_dir, "param_sweep_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {json_path}")

    # ── console table ──
    print("\n" + "=" * 65)
    print(f"{'chunk_size':>12}", end="")
    for th in THRESHOLDS:
        print(f"  thr={th:.1f}", end="")
    print()
    print("-" * 65)

    pivot = results_df.pivot(index="chunk_size", columns="threshold", values="session_accuracy")
    for cs in CHUNK_SIZES:
        row_str = f"{cs:>12}"
        for th in THRESHOLDS:
            val = pivot.loc[cs, th]
            marker = " *" if (cs == 1000 and th == 0.5) else "  "
            row_str += f"{marker}{val:.3f}"
        print(row_str)
    print("  (* = production default: chunk_size=1000, threshold=0.5)")
    print("=" * 65)


if __name__ == "__main__":
    main()
