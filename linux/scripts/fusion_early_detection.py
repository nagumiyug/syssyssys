#!/usr/bin/env python3
"""
Early detection study: how quickly can each feature mode detect an anomaly?

For each isolation mode and chunk_size, evaluates session-level accuracy using
only the first K chunks of each session (K = 1, 2, 3, 5, 8, 10, 15, 20).

Key hypothesis: fusion reaches high accuracy with fewer chunks because it
combines diverse signals that compensate for each other in short windows.

Grid:
  chunk_size  ∈ {200, 500, 1000}
  K           ∈ {1, 2, 3, 5, 8, 10, 15, 20}
  mode        ∈ {scalars, freq, seq, trans, ctx, fusion}

Output per software:
  early_detection_results.csv  — one row per (chunk_size, mode, K)
  early_detection_summary.json — structured summary

Usage:
  python fusion_early_detection.py --raw-dir data/raw/freecad_only --output-dir models/fusion_advantage/early_detection/freecad
  python fusion_early_detection.py --raw-dir data/raw/kicad_only   --output-dir models/fusion_advantage/early_detection/kicad
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
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ── constants ─────────────────────────────────────────────────────────────────

NOISE_SYSCALLS = {"mmap", "munmap", "mprotect"}
NORMAL_LABEL   = "normal"

CHUNK_SIZES    = [200, 500, 1000]
K_VALUES       = [1, 2, 3, 5, 8, 10, 15, 20]
ISOLATION_MODES = ["scalars", "freq", "seq", "trans", "ctx", "fusion"]

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=9,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# ── data loading ──────────────────────────────────────────────────────────────

def list_csvs(raw_dir: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        sys.exit(f"No CSV files found in {raw_dir}")
    return paths


def load_csv_filtered(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["syscall_name", "path_type", "label", "session_id", "scenario"],
        dtype=str,
    )
    return df[~df["syscall_name"].isin(NOISE_SYSCALLS)].reset_index(drop=True)


def collect_session_meta(csv_paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in csv_paths:
        df = pd.read_csv(p, usecols=["label", "session_id"], dtype=str, nrows=1)
        rows.append({"session_id": str(df["session_id"].iloc[0]),
                     "label":      str(df["label"].iloc[0]),
                     "path":       p})
    return pd.DataFrame(rows)


# ── vocabulary ────────────────────────────────────────────────────────────────

def build_vocab(csv_paths: list[str]) -> dict:
    syscalls: set = set()
    bigrams:  set = set()
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
        "syscalls":   sorted(syscalls),
        "bigrams":    sorted(bigrams),
        "trigrams":   sorted(trigrams),
        "path_types": sorted(path_types),
    }


# ── feature extraction (numpy path for memory efficiency) ─────────────────────

def _entropy(counts_dict: Counter) -> float:
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts_dict.values()), dtype=float) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _get_feature_names(vocab: dict) -> list[str]:
    names = ["syscall_entropy", "transition_entropy", "self_loop_ratio"]
    names += [f"freq_count::{s}"              for s in vocab["syscalls"]]
    names += [f"seq2_norm::{b[0]}->{b[1]}"   for b in vocab["bigrams"]]
    names += [f"seq3_norm::{t[0]}->{t[1]}->{t[2]}" for t in vocab["trigrams"]]
    names += [f"trans_norm::{b[0]}->{b[1]}"  for b in vocab["bigrams"]]
    names += [f"ctx_path_type::{pt}"          for pt in vocab["path_types"]]
    return names


def _extract_np(chunk: pd.DataFrame, vocab: dict,
                feat_idx: dict, n_feats: int) -> np.ndarray:
    arr = np.zeros(n_feats, dtype=np.float64)
    seq = chunk["syscall_name"].tolist()
    n   = len(seq)

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


def build_feature_matrix(csv_paths: list[str], vocab: dict,
                         chunk_size: int) -> pd.DataFrame:
    """Build feature matrix; includes chunk_idx (order within session)."""
    feat_names = _get_feature_names(vocab)
    feat_idx   = {name: i for i, name in enumerate(feat_names)}
    n_feats    = len(feat_names)

    arrays:     list[np.ndarray] = []
    session_ids: list[str]       = []
    labels:      list[int]       = []
    chunk_idxs:  list[int]       = []

    for p in csv_paths:
        df = load_csv_filtered(p)
        if df.empty:
            continue
        session_id  = str(df["session_id"].iloc[0])
        is_abnormal = int(df["label"].iloc[0] != NORMAL_LABEL)

        cidx = 0
        for start in range(0, len(df) - chunk_size + 1, chunk_size):
            chunk = df.iloc[start: start + chunk_size]
            arrays.append(_extract_np(chunk, vocab, feat_idx, n_feats))
            session_ids.append(session_id)
            labels.append(is_abnormal)
            chunk_idxs.append(cidx)
            cidx += 1
        del df

    X      = np.vstack(arrays)
    result = pd.DataFrame(X, columns=feat_names)
    result["session_id"] = session_ids
    result["label"]      = labels
    result["chunk_idx"]  = chunk_idxs
    return result


# ── column selection by mode ──────────────────────────────────────────────────

def select_columns(df: pd.DataFrame, mode: str) -> list[str]:
    meta = {"session_id", "label", "chunk_idx"}
    all_cols = df.columns.tolist()
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

def early_detection_eval(test_df: pd.DataFrame, y_pred_all: np.ndarray,
                         K: int, threshold: float = 0.5) -> dict:
    """
    Session-level accuracy using ONLY the first K chunks per session.
    Sessions with fewer than K chunks use all available chunks.
    """
    session_true: dict = defaultdict(list)
    session_pred: dict = defaultdict(list)

    for i, (sid, cidx, yt) in enumerate(zip(
        test_df["session_id"].values,
        test_df["chunk_idx"].values,
        test_df["label"].values,
    )):
        if cidx < K:
            session_true[sid].append(int(yt))
            session_pred[sid].append(int(y_pred_all[i]))

    y_true_s, y_pred_s = [], []
    for sid in session_true:
        y_true_s.append(int(any(v == 1 for v in session_true[sid])))
        vote = sum(session_pred[sid]) / len(session_pred[sid])
        y_pred_s.append(int(vote >= threshold))

    if not y_true_s:
        return {"session_accuracy": float("nan"), "session_fpr": float("nan"),
                "n_sessions": 0}

    y_true_arr = np.array(y_true_s)
    y_pred_arr = np.array(y_pred_s)
    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # detection rate: fraction of ANOMALOUS sessions correctly identified
    anom_mask = y_true_arr == 1
    if anom_mask.sum() > 0:
        det_rate = float(y_pred_arr[anom_mask].mean())
    else:
        det_rate = float("nan")

    return {
        "session_accuracy":  acc,
        "session_fpr":       fpr,
        "detection_rate":    det_rate,
        "n_sessions":        len(y_true_s),
        "n_anom_sessions":   int(anom_mask.sum()),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Early detection study")
    parser.add_argument("--raw-dir",      required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--test-size",    type=float, default=0.3)
    parser.add_argument("--random-state", type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = list_csvs(args.raw_dir)
    print(f"Found {len(csv_paths)} CSV files in {args.raw_dir}")

    # Fixed session-level train/test split
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

    # Vocab built once from ALL data (independent of chunk_size)
    print("Building vocabulary (full data) …")
    vocab = build_vocab(csv_paths)
    print(f"  syscalls={len(vocab['syscalls'])}  bigrams={len(vocab['bigrams'])}  "
          f"trigrams={len(vocab['trigrams'])}")

    all_results = []

    for chunk_size in CHUNK_SIZES:
        print(f"\n{'='*60}")
        print(f"chunk_size = {chunk_size}")

        print("  Building feature matrices …")
        train_df = build_feature_matrix(train_paths, vocab, chunk_size)
        test_df  = build_feature_matrix(test_paths,  vocab, chunk_size)
        print(f"  train_chunks={len(train_df)}  test_chunks={len(test_df)}")

        # Max chunks any test session has — cap K at this
        max_chunks_in_session = int(test_df.groupby("session_id")["chunk_idx"].max().max()) + 1
        effective_K = [k for k in K_VALUES if k <= max_chunks_in_session]
        if max_chunks_in_session < max(K_VALUES):
            print(f"  [info] max chunks/session={max_chunks_in_session}; "
                  f"K capped at {max_chunks_in_session}")

        for mode in ISOLATION_MODES:
            feat_cols = select_columns(train_df, mode)
            print(f"  [{mode}] {len(feat_cols)} features …", end="", flush=True)

            X_train = train_df[feat_cols].values
            y_train = train_df["label"].values
            X_test  = test_df[feat_cols].values

            clf = RandomForestClassifier(**RF_PARAMS)
            clf.fit(X_train, y_train)
            y_pred_all = clf.predict(X_test)

            for K in effective_K:
                metrics = early_detection_eval(test_df, y_pred_all, K)
                all_results.append({
                    "chunk_size":        chunk_size,
                    "mode":              mode,
                    "K":                 K,
                    "events_observed":   K * chunk_size,
                    **metrics,
                })

            # Print summary at K=1, 5, full (last effective K)
            brief = {K: early_detection_eval(test_df, y_pred_all, K)
                     for K in [1, 5, effective_K[-1]]}
            parts = "  ".join(f"K={k}: {v['session_accuracy']:.3f}" for k, v in brief.items())
            print(f" {parts}")

        del train_df, test_df

    # ── save ──
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, "early_detection_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    summary = {
        "chunk_sizes": CHUNK_SIZES,
        "K_values":    K_VALUES,
        "modes":       ISOLATION_MODES,
        "n_train_sessions": len(train_sids),
        "n_test_sessions":  len(test_sids),
        "results": results_df.to_dict(orient="records"),
    }
    json_path = os.path.join(args.output_dir, "early_detection_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {json_path}")

    # ── console table: K=1,3,5,full at chunk_size=1000 ──
    print("\n" + "=" * 65)
    print(f"chunk_size=1000 — session accuracy vs K (first K chunks)")
    print("-" * 65)
    sub = results_df[results_df["chunk_size"] == 1000]
    K_show = [k for k in [1, 3, 5, 10, max(K_VALUES)]
              if k in sub["K"].values]
    header = f"{'mode':<10}" + "".join(f"  K={k:>3}" for k in K_show)
    print(header)
    for mode in ISOLATION_MODES:
        row_str = f"{mode:<10}"
        for k in K_show:
            val = sub[(sub["mode"] == mode) & (sub["K"] == k)]["session_accuracy"]
            row_str += f"  {val.values[0]:.3f}" if len(val) else "      —"
        print(row_str)
    print("=" * 65)


if __name__ == "__main__":
    main()
