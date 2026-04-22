#!/usr/bin/env python3
"""
Fusion confidence study: do all modes achieve the same session accuracy, yet
fusion produces more confident (higher-probability) chunk-level predictions?

Rationale: In a real deployment the classifier must commit to an alert at the
chunk level before enough session chunks arrive for a majority vote. A mode
that is accurate but assigns low probability to its predictions gives less
actionable signal. Fusion's diverse features should drive probabilities toward
0 or 1 even on individual chunks, making it more reliable early.

Metrics (chunk-level, using predict_proba):
  mean_correct_proba — mean P(correct class) across all test chunks
  macro_auc_roc      — multi-class OvR AUC (sklearn multi_class='ovr')
  mean_entropy       — mean -sum(p * log2(p+eps)) over probability vectors
  brier_score        — mean sum((p - one_hot(y))^2) per chunk

Binary variants (normal vs anomalous):
  binary_auc_roc         — AUC using 1 - P(normal) as anomaly score
  binary_mean_correct_proba — mean P(correct binary class) per chunk

Output:
  <output-dir>/confidence_results.csv — one row per mode

Usage:
  python fusion_confidence_study.py --raw-dir data/raw/freecad_only \
      --output-dir models/fusion_advantage/confidence/freecad
"""

import argparse
import glob
import os
import re
import sys
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# ── constants ─────────────────────────────────────────────────────────────────

NOISE_SYSCALLS  = {"mmap", "munmap", "mprotect"}
NORMAL_LABEL    = "normal"
CHUNK_SIZE      = 1000
ISOLATION_MODES = ["scalars", "freq", "seq", "trans", "ctx", "fusion"]
RF_PARAMS = dict(n_estimators=300, max_depth=9, class_weight="balanced",
                 random_state=42, n_jobs=-1)


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


def collect_session_meta(csv_paths: list) -> pd.DataFrame:
    """Read session-level metadata; compute subtype via scenario stripping."""
    rows = []
    for p in csv_paths:
        df = pd.read_csv(
            p, usecols=["label", "session_id", "scenario"], dtype=str, nrows=1
        )
        label    = str(df["label"].iloc[0])
        scenario = str(df["scenario"].iloc[0])
        subtype  = re.sub(r"_linux_r\d+$", "", scenario) if label != NORMAL_LABEL else NORMAL_LABEL
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
    """Build feature matrix with session_id + subtype columns (no label column)."""
    feat_names = _get_feature_names(vocab)
    feat_idx   = {name: i for i, name in enumerate(feat_names)}
    n_feats    = len(feat_names)

    arrays:      list = []
    session_ids: list = []
    subtypes:    list = []

    for p in csv_paths:
        df = load_csv_filtered(p)
        if df.empty:
            continue
        session_id = str(df["session_id"].iloc[0])
        subtype    = subtype_map.get(session_id, NORMAL_LABEL)

        for start in range(0, len(df) - CHUNK_SIZE + 1, CHUNK_SIZE):
            chunk = df.iloc[start: start + CHUNK_SIZE]
            arrays.append(_extract_np(chunk, vocab, feat_idx, n_feats))
            session_ids.append(session_id)
            subtypes.append(subtype)
        del df

    X      = np.vstack(arrays)
    result = pd.DataFrame(X, columns=feat_names)
    result["session_id"] = session_ids
    result["subtype"]    = subtypes
    return result


# ── column selection ──────────────────────────────────────────────────────────

def select_columns(df: pd.DataFrame, mode: str) -> list:
    meta     = {"session_id", "subtype"}
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


# ── confidence metrics ────────────────────────────────────────────────────────

def compute_confidence_metrics(
    proba: np.ndarray,
    y_enc: np.ndarray,
    class_names: list,
    normal_idx: int,
) -> dict:
    """
    Compute chunk-level confidence metrics from probability matrix.

    Parameters
    ----------
    proba       : (N, C) float array of predicted class probabilities
    y_enc       : (N,)   int array of true encoded class labels
    class_names : list of class name strings (length C)
    normal_idx  : integer index corresponding to NORMAL_LABEL in class_names

    Returns
    -------
    dict with keys: mean_correct_proba, macro_auc_roc, mean_entropy,
                    brier_score, binary_auc_roc, binary_mean_correct_proba
    """
    n_chunks   = len(y_enc)
    n_classes  = len(class_names)
    eps        = 1e-12

    # ── mean probability assigned to the correct class ──
    correct_proba = proba[np.arange(n_chunks), y_enc]
    mean_correct_proba = float(np.mean(correct_proba))

    # ── multi-class OvR AUC ──
    y_bin = label_binarize(y_enc, classes=list(range(n_classes)))
    # label_binarize returns (N, 1) for binary; expand to (N, 2) for sklearn compat
    if n_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    macro_auc_roc = float(
        roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
    )

    # ── mean entropy over predicted probability vectors ──
    mean_entropy = float(np.mean(-np.sum(proba * np.log2(proba + eps), axis=1)))

    # ── Brier score: mean sum((p - one_hot(y))^2) per chunk ──
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(n_chunks), y_enc] = 1.0
    brier_score = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))

    # ── binary: normal=0 vs anomalous=1 ──
    # anomaly score = 1 - P(normal)
    anomaly_score = 1.0 - proba[:, normal_idx]
    y_binary      = (y_enc != normal_idx).astype(int)

    try:
        binary_auc_roc = float(roc_auc_score(y_binary, anomaly_score))
    except ValueError:
        # only one class present in y_binary
        binary_auc_roc = float("nan")

    # binary correct proba: P(normal) for truly normal chunks, P(anomalous) for anomalous
    p_anomalous = 1.0 - proba[:, normal_idx]
    binary_correct_proba = np.where(y_binary == 0, proba[:, normal_idx], p_anomalous)
    binary_mean_correct_proba = float(np.mean(binary_correct_proba))

    return {
        "mean_correct_proba":       mean_correct_proba,
        "macro_auc_roc":            macro_auc_roc,
        "mean_entropy":             mean_entropy,
        "brier_score":              brier_score,
        "binary_auc_roc":           binary_auc_roc,
        "binary_mean_correct_proba": binary_mean_correct_proba,
    }


# ── session-level accuracy helper ────────────────────────────────────────────

def session_accuracy(y_enc: np.ndarray, y_pred: np.ndarray,
                     sids: np.ndarray) -> float:
    """Majority-vote session accuracy for multi-class labels."""
    session_true: dict  = {}
    session_preds: dict = defaultdict(list)
    for sid, yt, yp in zip(sids, y_enc, y_pred):
        if sid not in session_true:
            session_true[sid] = yt
        session_preds[sid].append(yp)

    y_true_s, y_pred_s = [], []
    for sid in session_true:
        y_true_s.append(session_true[sid])
        vote = Counter(session_preds[sid]).most_common(1)[0][0]
        y_pred_s.append(vote)

    return float(accuracy_score(np.array(y_true_s), np.array(y_pred_s)))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fusion confidence study")
    parser.add_argument("--raw-dir",      required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--test-size",    type=float, default=0.3)
    parser.add_argument("--random-state", type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = list_csvs(args.raw_dir)
    print(f"Found {len(csv_paths)} CSV files in {args.raw_dir}")

    # Session-level metadata with 4-class subtype labels
    meta = collect_session_meta(csv_paths)
    subtype_map = dict(zip(meta["session_id"], meta["subtype"]))

    class_names = sorted(meta["subtype"].unique().tolist())
    label_enc   = {cls: i for i, cls in enumerate(class_names)}
    normal_idx  = label_enc[NORMAL_LABEL]
    print(f"  Classes ({len(class_names)}): {class_names}")
    print(f"  normal_idx = {normal_idx}")

    # Stratified train/test split on subtype
    train_sids, test_sids = train_test_split(
        meta["session_id"],
        test_size=args.test_size,
        stratify=meta["subtype"],
        random_state=args.random_state,
    )
    train_meta = meta[meta["session_id"].isin(train_sids)].reset_index(drop=True)
    test_meta  = meta[meta["session_id"].isin(test_sids)].reset_index(drop=True)
    print(f"  train sessions={len(train_meta)}  test sessions={len(test_meta)}")

    # Vocabulary from all data
    print("Building vocabulary …")
    vocab = build_vocab(csv_paths)
    print(f"  syscalls={len(vocab['syscalls'])}  bigrams={len(vocab['bigrams'])}  "
          f"trigrams={len(vocab['trigrams'])}")

    # Feature matrices built once
    print("Building training feature matrix …")
    train_df = build_feature_matrix(train_meta["path"].tolist(), vocab, subtype_map)
    print(f"  train_chunks={len(train_df)}")

    print("Building test feature matrix …")
    test_df = build_feature_matrix(test_meta["path"].tolist(), vocab, subtype_map)
    print(f"  test_chunks={len(test_df)}")

    # Encode labels as standalone numpy arrays — NOT added to DataFrame columns
    # to prevent select_columns("fusion") from including them as features.
    y_train_enc = train_df["subtype"].map(label_enc).values
    y_test_enc  = test_df["subtype"].map(label_enc).values
    sids_test   = test_df["session_id"].values

    all_results = []

    print(f"\n{'mode':<10}  {'session_acc':>11}  {'mean_correct_p':>14}  "
          f"{'macro_auc':>9}  {'brier':>7}  {'entropy':>8}")
    print("-" * 70)

    for mode in ISOLATION_MODES:
        feat_cols = select_columns(train_df, mode)
        X_train   = train_df[feat_cols].values
        X_test    = test_df[feat_cols].values

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_train, y_train_enc)

        proba  = clf.predict_proba(X_test)   # (N, C)
        y_pred = clf.predict(X_test)

        sess_acc = session_accuracy(y_test_enc, y_pred, sids_test)
        metrics  = compute_confidence_metrics(proba, y_test_enc, class_names, normal_idx)

        print(f"{mode:<10}  {sess_acc:>11.3f}  {metrics['mean_correct_proba']:>14.4f}  "
              f"{metrics['macro_auc_roc']:>9.4f}  {metrics['brier_score']:>7.4f}  "
              f"{metrics['mean_entropy']:>8.4f}")

        row = {
            "mode":                       mode,
            "n_features":                 len(feat_cols),
            "session_accuracy":           sess_acc,
            "mean_correct_proba":         metrics["mean_correct_proba"],
            "macro_auc_roc":              metrics["macro_auc_roc"],
            "mean_entropy":               metrics["mean_entropy"],
            "brier_score":                metrics["brier_score"],
            "binary_auc_roc":             metrics["binary_auc_roc"],
            "binary_mean_correct_proba":  metrics["binary_mean_correct_proba"],
        }
        all_results.append(row)

    print("-" * 70)

    # ── save results ──
    results_df = pd.DataFrame(all_results)
    csv_path   = os.path.join(args.output_dir, "confidence_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # ── printed table: mode | mean_correct_proba | macro_auc_roc | brier_score | mean_entropy ──
    print("\n" + "=" * 70)
    print(f"{'mode':<10}  {'mean_correct_proba':>18}  {'macro_auc_roc':>13}  "
          f"{'brier_score':>11}  {'mean_entropy':>12}")
    print("-" * 70)
    for row in all_results:
        print(f"{row['mode']:<10}  {row['mean_correct_proba']:>18.4f}  "
              f"{row['macro_auc_roc']:>13.4f}  {row['brier_score']:>11.4f}  "
              f"{row['mean_entropy']:>12.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
