#!/usr/bin/env python3
"""
Multi-class anomaly type identification study.

Changes the task from binary detection (normal vs anomalous) to 4-class
identification: normal / bulk_export / source_copy / project_scan.

Hypothesis:
  Binary detection is trivially solved by any single feature group because
  the normal-vs-anomalous gap is large enough for any signal to catch.
  Type identification is harder: it requires the model to discriminate BETWEEN
  attack types, not just against normal behaviour.

  - scalars (3 entropy features): captures normal vs anomalous divergence,
    but entropy values are similar across attack types → high inter-class confusion
  - freq/seq/trans: encode attack-specific syscall patterns, may confuse types
    that share common syscalls
  - ctx (path_type): highly attack-specific (export_target, project paths,
    copy destinations) → good at typing if trained on all types
  - fusion: combines entropy, sequence, and path signals → highest discriminative
    power across all 4 classes

Key metric: per-class recall on attack types (the hardest cells in the confusion
matrix), plus overall 4-class session accuracy.

Output per software:
  multiclass_results.csv    — one row per mode, with per-class metrics
  multiclass_confmat.json   — confusion matrices per mode

Usage:
  python fusion_multiclass_study.py --raw-dir data/raw/freecad_only \
      --output-dir models/fusion_advantage/multiclass/freecad
  python fusion_multiclass_study.py --raw-dir data/raw/kicad_only \
      --output-dir models/fusion_advantage/multiclass/kicad
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
    accuracy_score, confusion_matrix, classification_report,
    f1_score, recall_score
)
from sklearn.model_selection import train_test_split

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
    """Read session-level metadata: session_id, label, subtype (4-class)."""
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
            "subtype":    subtype,   # one of: normal / bulk_export / source_copy / project_scan
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
    """Build feature matrix with 4-class label (subtype) per chunk."""
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


# ── evaluation ────────────────────────────────────────────────────────────────

def session_level_eval_multiclass(
    y_true_chunk: np.ndarray,
    y_pred_chunk: np.ndarray,
    session_ids:  np.ndarray,
    class_names:  list,
) -> dict:
    """
    Session-level multi-class evaluation using majority vote.
    Returns per-class recall and overall accuracy.
    """
    session_true: dict = {}
    session_preds: dict = defaultdict(list)

    for sid, yt, yp in zip(session_ids, y_true_chunk, y_pred_chunk):
        if sid not in session_true:
            session_true[sid] = yt
        session_preds[sid].append(yp)

    y_true_s, y_pred_s = [], []
    for sid in session_true:
        y_true_s.append(session_true[sid])
        # majority vote: most common prediction
        vote = Counter(session_preds[sid]).most_common(1)[0][0]
        y_pred_s.append(vote)

    y_true_arr = np.array(y_true_s)
    y_pred_arr = np.array(y_pred_s)

    acc   = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro",
                              labels=list(range(len(class_names))), zero_division=0))

    per_class_recall = recall_score(
        y_true_arr, y_pred_arr,
        average=None,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    cm = confusion_matrix(y_true_arr, y_pred_arr,
                          labels=list(range(len(class_names))))

    return {
        "session_accuracy": acc,
        "macro_f1":         macro_f1,
        "per_class_recall": {class_names[i]: float(per_class_recall[i])
                             for i in range(len(class_names))},
        "confusion_matrix": cm.tolist(),
        "n_sessions":       len(y_true_s),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-class anomaly type study")
    parser.add_argument("--raw-dir",    required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size",  type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = list_csvs(args.raw_dir)
    print(f"Found {len(csv_paths)} CSV files in {args.raw_dir}")

    # Collect session-level 4-class labels
    meta = collect_session_meta(csv_paths)
    subtype_map = dict(zip(meta["session_id"], meta["subtype"]))

    class_names = sorted(meta["subtype"].unique().tolist())
    label_enc   = {c: i for i, c in enumerate(class_names)}
    print(f"  Classes ({len(class_names)}): {class_names}")

    # Stratified train/test split on 4-class labels
    train_sids, test_sids = train_test_split(
        meta["session_id"],
        test_size=args.test_size,
        stratify=meta["subtype"],
        random_state=42,
    )
    train_meta = meta[meta["session_id"].isin(train_sids)].reset_index(drop=True)
    test_meta  = meta[meta["session_id"].isin(test_sids)].reset_index(drop=True)
    print(f"  train={len(train_meta)} sessions  test={len(test_meta)} sessions")
    print(f"  test class distribution:")
    for cls, cnt in test_meta["subtype"].value_counts().items():
        print(f"    {cls}: {cnt}")

    # Vocab from ALL data
    print("Building vocabulary …")
    vocab = build_vocab(csv_paths)
    print(f"  syscalls={len(vocab['syscalls'])}  bigrams={len(vocab['bigrams'])}  "
          f"trigrams={len(vocab['trigrams'])}")

    # Feature matrices (built once)
    print("Building feature matrices …")
    train_df = build_feature_matrix(train_meta["path"].tolist(), vocab, subtype_map)
    test_df  = build_feature_matrix(test_meta["path"].tolist(),  vocab, subtype_map)
    print(f"  train_chunks={len(train_df)}  test_chunks={len(test_df)}")

    # Encode 4-class labels as separate arrays — NOT added to DataFrame to
    # prevent label_enc from leaking into the fusion feature set via select_columns.
    y_train_enc = train_df["subtype"].map(label_enc).values
    y_test_enc  = test_df["subtype"].map(label_enc).values
    sids_test   = test_df["session_id"].values

    all_results   = []
    all_confmats  = {}

    print(f"\n{'='*70}")
    print(f"{'Mode':<10}  {'Session Acc':>12}  {'Macro F1':>9}  "
          + "  ".join(f"{c[:10]:>12}" for c in class_names))
    print("-" * 70)

    for mode in ISOLATION_MODES:
        feat_cols = select_columns(train_df, mode)
        X_train   = train_df[feat_cols].values
        y_train   = y_train_enc
        X_test    = test_df[feat_cols].values
        y_test    = y_test_enc

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Chunk-level classification report (for reference)
        chunk_acc = float(accuracy_score(y_test, y_pred))

        # Session-level evaluation
        m = session_level_eval_multiclass(y_test, y_pred, sids_test, class_names)

        per_cls = m["per_class_recall"]
        print(f"{mode:<10}  {m['session_accuracy']:>12.3f}  {m['macro_f1']:>9.3f}  "
              + "  ".join(f"{per_cls.get(c, 0.0):>12.3f}" for c in class_names))

        row = {
            "mode":             mode,
            "n_features":       len(feat_cols),
            "chunk_accuracy":   chunk_acc,
            "session_accuracy": m["session_accuracy"],
            "macro_f1":         m["macro_f1"],
            "n_sessions":       m["n_sessions"],
        }
        for cls in class_names:
            row[f"recall_{cls}"] = per_cls.get(cls, 0.0)
        all_results.append(row)
        all_confmats[mode] = {
            "class_names":     class_names,
            "confusion_matrix": m["confusion_matrix"],
        }

    print("=" * 70)

    # ── attack-type accuracy (restricted to anomalous sessions only) ──
    print("\n--- Attack-type recall (anomalous sessions only) ---")
    attack_classes = [c for c in class_names if c != NORMAL_LABEL]
    print(f"{'Mode':<10}  " + "  ".join(f"{c[:14]:>14}" for c in attack_classes)
          + f"  {'mean_attack':>12}")
    print("-" * 70)
    for row in all_results:
        recalls = [row[f"recall_{c}"] for c in attack_classes]
        mean_r  = float(np.mean(recalls))
        print(f"{row['mode']:<10}  "
              + "  ".join(f"{r:>14.3f}" for r in recalls)
              + f"  {mean_r:>12.3f}")

    # ── save ──
    results_df = pd.DataFrame(all_results)
    csv_path   = os.path.join(args.output_dir, "multiclass_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    json_path = os.path.join(args.output_dir, "multiclass_confmat.json")
    with open(json_path, "w") as f:
        json.dump({
            "class_names":  class_names,
            "label_enc":    label_enc,
            "confmats":     all_confmats,
        }, f, indent=2)
    print(f"Confusion matrices → {json_path}")


if __name__ == "__main__":
    main()
