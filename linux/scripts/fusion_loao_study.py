#!/usr/bin/env python3
"""
Leave-One-Anomaly-Out (LOAO) generalization study.

Simulates zero-shot anomaly detection: the model is trained without ever seeing
one anomaly type, then tested on whether it can still detect that unseen type.

For each held-out anomaly type C ∈ {bulk_export, source_copy, project_scan}:
  - Train on: all normal sessions + all sessions of the OTHER two anomaly types
  - Test on:  held-out type C sessions only (treated as positive class)
              + a fixed set of normal sessions (negative class)

This is the hardest evaluation: a classifier that simply memorised type A and B
patterns must generalise to type C without any prior exposure.

Hypothesis:
  - Single-feature modes (esp. seq/trans) learn attack-specific patterns and
    fail to generalise to unseen attack types.
  - Fusion, by combining entropy/path-type (attack-agnostic) with sequence
    (attack-specific) features, maintains detection capability on unseen types.

Output per software:
  loao_results.csv    — one row per (mode, held_out_type)
  loao_summary.json   — structured results

Usage:
  python fusion_loao_study.py --raw-dir data/raw/freecad_only \
      --output-dir models/fusion_advantage/loao/freecad
  python fusion_loao_study.py --raw-dir data/raw/kicad_only \
      --output-dir models/fusion_advantage/loao/kicad
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
    accuracy_score, confusion_matrix, recall_score, f1_score
)

# ── constants ─────────────────────────────────────────────────────────────────

NOISE_SYSCALLS  = {"mmap", "munmap", "mprotect"}
NORMAL_LABEL    = "normal"
CHUNK_SIZE      = 1000

ISOLATION_MODES = ["scalars", "freq", "seq", "trans", "ctx", "fusion"]

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=9,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# ── data loading ──────────────────────────────────────────────────────────────

def list_csvs(raw_dir: str) -> list:
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


def _scenario_to_subtype(scenario: str) -> str:
    return re.sub(r"_linux_r\d+$", "", scenario)


def collect_session_meta(csv_paths: list) -> pd.DataFrame:
    rows = []
    for p in csv_paths:
        df = pd.read_csv(
            p, usecols=["label", "session_id", "scenario"], dtype=str, nrows=1
        )
        label    = str(df["label"].iloc[0])
        scenario = str(df["scenario"].iloc[0])
        subtype  = _scenario_to_subtype(scenario) if label != NORMAL_LABEL else NORMAL_LABEL
        rows.append({
            "session_id": str(df["session_id"].iloc[0]),
            "label":      label,
            "subtype":    subtype,
            "path":       p,
        })
    return pd.DataFrame(rows)


# ── vocabulary ────────────────────────────────────────────────────────────────

def build_vocab(csv_paths: list) -> dict:
    syscalls:   set = set()
    bigrams:    set = set()
    trigrams:   set = set()
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


# ── feature extraction ────────────────────────────────────────────────────────

def _entropy(counts_dict: Counter) -> float:
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts_dict.values()), dtype=float) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _get_feature_names(vocab: dict) -> list:
    names = ["syscall_entropy", "transition_entropy", "self_loop_ratio"]
    names += [f"freq_count::{s}"                     for s in vocab["syscalls"]]
    names += [f"seq2_norm::{b[0]}->{b[1]}"          for b in vocab["bigrams"]]
    names += [f"seq3_norm::{t[0]}->{t[1]}->{t[2]}" for t in vocab["trigrams"]]
    names += [f"trans_norm::{b[0]}->{b[1]}"         for b in vocab["bigrams"]]
    names += [f"ctx_path_type::{pt}"                for pt in vocab["path_types"]]
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


def build_feature_matrix(csv_paths: list, vocab: dict,
                         subtype_map: dict) -> pd.DataFrame:
    """
    Build feature matrix; adds anomaly_subtype column using the pre-built map.
    subtype_map: {session_id -> subtype}
    """
    feat_names = _get_feature_names(vocab)
    feat_idx   = {name: i for i, name in enumerate(feat_names)}
    n_feats    = len(feat_names)

    arrays:      list = []
    session_ids: list = []
    labels:      list = []
    subtypes:    list = []

    for p in csv_paths:
        df = load_csv_filtered(p)
        if df.empty:
            continue
        session_id  = str(df["session_id"].iloc[0])
        is_abnormal = int(df["label"].iloc[0] != NORMAL_LABEL)
        subtype     = subtype_map.get(session_id, NORMAL_LABEL)

        for start in range(0, len(df) - CHUNK_SIZE + 1, CHUNK_SIZE):
            chunk = df.iloc[start: start + CHUNK_SIZE]
            arrays.append(_extract_np(chunk, vocab, feat_idx, n_feats))
            session_ids.append(session_id)
            labels.append(is_abnormal)
            subtypes.append(subtype)
        del df

    X      = np.vstack(arrays)
    result = pd.DataFrame(X, columns=feat_names)
    result["session_id"]     = session_ids
    result["label"]          = labels
    result["anomaly_subtype"] = subtypes
    return result


# ── column selection ──────────────────────────────────────────────────────────

def select_columns(df: pd.DataFrame, mode: str) -> list:
    meta     = {"session_id", "label", "anomaly_subtype"}
    all_cols = df.columns.tolist()
    if mode == "scalars":
        return [c for c in ["syscall_entropy", "transition_entropy", "self_loop_ratio"]
                if c in all_cols]
    if mode == "freq":
        return [c for c in all_cols if c.startswith("freq_count::")]
    if mode == "seq":
        return [c for c in all_cols
                if c.startswith("seq2_norm::") or c.startswith("seq3_norm::")]
    if mode == "trans":
        return [c for c in all_cols if c.startswith("trans_norm::")]
    if mode == "ctx":
        return [c for c in all_cols if c.startswith("ctx_path_type::")]
    if mode == "fusion":
        return [c for c in all_cols if c not in meta]
    raise ValueError(f"Unknown mode: {mode}")


# ── evaluation helpers ────────────────────────────────────────────────────────

def session_level_eval(y_test_chunk, y_pred_chunk,
                       session_ids, threshold: float = 0.5) -> dict:
    session_true: dict = defaultdict(list)
    session_pred: dict = defaultdict(list)
    for sid, yt, yp in zip(session_ids, y_test_chunk, y_pred_chunk):
        session_true[sid].append(int(yt))
        session_pred[sid].append(int(yp))

    y_true_s, y_pred_s = [], []
    for sid in session_true:
        y_true_s.append(int(any(v == 1 for v in session_true[sid])))
        vote = sum(session_pred[sid]) / len(session_pred[sid])
        y_pred_s.append(int(vote >= threshold))

    y_true_arr = np.array(y_true_s)
    y_pred_arr = np.array(y_pred_s)
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    cm  = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    fpr    = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return {
        "session_accuracy": acc,
        "session_fpr":      fpr,
        "session_recall":   recall,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Leave-One-Anomaly-Out study")
    parser.add_argument("--raw-dir",    required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = list_csvs(args.raw_dir)
    print(f"Found {len(csv_paths)} CSV files in {args.raw_dir}")

    # Collect session metadata
    meta = collect_session_meta(csv_paths)
    subtype_map = dict(zip(meta["session_id"], meta["subtype"]))

    normal_meta  = meta[meta["label"] == NORMAL_LABEL].reset_index(drop=True)
    anomaly_meta = meta[meta["label"] != NORMAL_LABEL].reset_index(drop=True)
    anomaly_types = sorted(anomaly_meta["subtype"].unique().tolist())

    print(f"  Normal sessions:  {len(normal_meta)}")
    print(f"  Anomaly sessions: {len(anomaly_meta)}")
    print(f"  Anomaly types:    {anomaly_types}")

    # Vocab from ALL sessions (ensures all possible feature columns exist)
    print("Building vocabulary from all data …")
    vocab = build_vocab(csv_paths)
    print(f"  syscalls={len(vocab['syscalls'])}  bigrams={len(vocab['bigrams'])}  "
          f"trigrams={len(vocab['trigrams'])}")

    # Build full feature matrix for ALL sessions once
    print("Building full feature matrix …")
    full_df = build_feature_matrix(csv_paths, vocab, subtype_map)
    print(f"  total chunks={len(full_df)}")

    all_results = []

    for held_out in anomaly_types:
        print(f"\n{'='*65}")
        print(f"Held-out anomaly type: {held_out}")

        # Training sessions: all normal + anomaly sessions NOT of held_out type
        other_anomaly_types = [t for t in anomaly_types if t != held_out]
        train_sids = set(normal_meta["session_id"].tolist()) | set(
            anomaly_meta.loc[
                anomaly_meta["subtype"].isin(other_anomaly_types), "session_id"
            ].tolist()
        )
        # Test sessions: a FIXED HALF of normal + ALL held-out sessions
        # (Use same normal sessions as the standard 30% test split: last 30% by index)
        n_normal_test = max(2, int(len(normal_meta) * 0.3))
        normal_test_sids = set(
            normal_meta.sort_values("session_id").tail(n_normal_test)["session_id"].tolist()
        )
        held_out_sids = set(
            anomaly_meta.loc[anomaly_meta["subtype"] == held_out, "session_id"].tolist()
        )
        test_sids = normal_test_sids | held_out_sids

        # Remove test normal sessions from training to avoid leakage
        train_sids -= normal_test_sids

        print(f"  train sessions: {len(train_sids)}  "
              f"(normal + {'+'.join(other_anomaly_types)})")
        print(f"  test sessions:  {len(test_sids)}  "
              f"(normal_test={len(normal_test_sids)}, {held_out}={len(held_out_sids)})")

        train_df = full_df[full_df["session_id"].isin(train_sids)].reset_index(drop=True)
        test_df  = full_df[full_df["session_id"].isin(test_sids)].reset_index(drop=True)

        print(f"  train chunks={len(train_df)}  test chunks={len(test_df)}")

        # Verify test set label sanity
        test_labels = test_df[["session_id","label","anomaly_subtype"]].drop_duplicates("session_id")
        n_pos = (test_labels["label"] == 1).sum()
        n_neg = (test_labels["label"] == 0).sum()
        print(f"  test: {n_pos} anomalous (all {held_out}), {n_neg} normal")

        for mode in ISOLATION_MODES:
            feat_cols = select_columns(train_df, mode)
            X_train   = train_df[feat_cols].values
            y_train   = train_df["label"].values
            X_test    = test_df[feat_cols].values
            y_test    = test_df["label"].values
            sids_test = test_df["session_id"].values

            clf = RandomForestClassifier(**RF_PARAMS)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            m = session_level_eval(y_test, y_pred, sids_test)

            print(f"  [{mode:<7}] acc={m['session_accuracy']:.3f}  "
                  f"recall={m['session_recall']:.3f}  fpr={m['session_fpr']:.3f}  "
                  f"(tp={m['tp']} fn={m['fn']} fp={m['fp']})")

            all_results.append({
                "held_out_type":    held_out,
                "mode":             mode,
                "n_train_sessions": len(train_sids),
                "n_test_held_out":  len(held_out_sids),
                "n_test_normal":    len(normal_test_sids),
                **{f"session_{k}": v for k, v in m.items()},
            })

    # ── save ──
    results_df = pd.DataFrame(all_results)
    csv_path   = os.path.join(args.output_dir, "loao_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    summary = {
        "anomaly_types":  anomaly_types,
        "modes":          ISOLATION_MODES,
        "results":        results_df.to_dict(orient="records"),
    }
    json_path = os.path.join(args.output_dir, "loao_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {json_path}")

    # ── console pivot: recall per (mode, held_out_type) ──
    print("\n" + "=" * 70)
    print("Session-level RECALL on held-out anomaly type (key metric)")
    print("-" * 70)
    pivot = results_df.pivot(index="mode", columns="held_out_type",
                             values="session_session_recall")
    pivot["mean_recall"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean_recall", ascending=False)
    print(pivot.to_string(float_format="{:.3f}".format))
    print("=" * 70)

    print("\n" + "=" * 70)
    print("Session-level ACCURACY on held-out test set")
    print("-" * 70)
    pivot2 = results_df.pivot(index="mode", columns="held_out_type",
                              values="session_session_accuracy")
    pivot2["mean_acc"] = pivot2.mean(axis=1)
    pivot2 = pivot2.sort_values("mean_acc", ascending=False)
    print(pivot2.to_string(float_format="{:.3f}".format))
    print("=" * 70)


if __name__ == "__main__":
    main()
