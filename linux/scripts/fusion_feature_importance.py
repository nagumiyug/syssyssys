#!/usr/bin/env python3
"""
Feature-group contribution analysis.

Quantifies each feature group's contribution to the fusion model with three
complementary methods:

  1. Gini importance (feature_importances_ from the fused RF)
     Summed per feature group; fast, baseline measure.

  2. Permutation importance — global
     For each feature group, shuffle those columns in the test set and measure
     the drop in test accuracy.  Accuracy drop = necessity.

  3. Permutation importance — per class (one-vs-rest)
     Train one binary RF per class (normal-vs-rest), apply permutation importance
     per feature group.  Reveals class-specific feature necessity.

Directly answers the client's questions:
  "各自对于不同的异常的检测效用是怎么样的？" → per-class contribution
  "它们各自的贡献占比又是多少？"            → grouped importance %

Output:
  gini_importance_grouped.csv      — group, sum_importance
  permutation_importance_global.csv — group, mean_acc_drop, std
  permutation_importance_per_class.csv — group, class, mean_acc_drop

Usage:
  python fusion_feature_importance.py --raw-dir data/raw/freecad_only \\
      --output-dir models/fusion_advantage/importance/freecad
  python fusion_feature_importance.py --raw-dir data/raw/kicad_only \\
      --output-dir models/fusion_advantage/importance/kicad
"""

import argparse
import glob
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

NOISE_SYSCALLS = {"mmap", "munmap", "mprotect"}
NORMAL_LABEL   = "normal"
CHUNK_SIZE     = 1000
SCALAR_COLS    = {"syscall_entropy", "transition_entropy", "self_loop_ratio"}

PERMUTATION_SEEDS = list(range(5))

RF_PARAMS = dict(
    n_estimators=300, max_depth=9, class_weight="balanced",
    random_state=42, n_jobs=-1,
)


def list_csvs(raw_dir: str) -> list:
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        sys.exit(f"No CSVs in {raw_dir}")
    return paths


def load_csv_filtered(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, dtype=str,
        usecols=["syscall_name", "path_type", "label", "session_id", "scenario"],
    )
    return df[~df["syscall_name"].isin(NOISE_SYSCALLS)].reset_index(drop=True)


def _scenario_to_subtype(s: str) -> str:
    return re.sub(r"_linux_r\d+$", "", s)


def collect_session_meta(csv_paths: list) -> pd.DataFrame:
    rows = []
    for p in csv_paths:
        df = pd.read_csv(p, usecols=["label", "session_id", "scenario"],
                         dtype=str, nrows=1)
        label    = str(df["label"].iloc[0])
        scenario = str(df["scenario"].iloc[0])
        subtype  = _scenario_to_subtype(scenario) if label != NORMAL_LABEL else NORMAL_LABEL
        rows.append({"session_id": str(df["session_id"].iloc[0]),
                     "label": label, "subtype": subtype, "path": p})
    return pd.DataFrame(rows)


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
    return {"syscalls": sorted(syscalls), "bigrams": sorted(bigrams),
            "trigrams": sorted(trigrams), "path_types": sorted(path_types)}


def _entropy(c: Counter) -> float:
    total = sum(c.values())
    if total == 0:
        return 0.0
    probs = np.array(list(c.values()), dtype=float) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _get_feature_names(vocab: dict) -> list:
    names = ["syscall_entropy", "transition_entropy", "self_loop_ratio"]
    names += [f"freq_count::{s}" for s in vocab["syscalls"]]
    names += [f"seq2_norm::{b[0]}->{b[1]}" for b in vocab["bigrams"]]
    names += [f"seq3_norm::{t[0]}->{t[1]}->{t[2]}" for t in vocab["trigrams"]]
    names += [f"trans_norm::{b[0]}->{b[1]}" for b in vocab["bigrams"]]
    names += [f"ctx_path_type::{pt}" for pt in vocab["path_types"]]
    return names


def _extract_np(chunk, vocab, feat_idx, n_feats):
    arr = np.zeros(n_feats, dtype=np.float64)
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
    arr[feat_idx["syscall_entropy"]]    = _entropy(sc_counts)
    arr[feat_idx["transition_entropy"]] = _entropy(bg_counts)
    arr[feat_idx["self_loop_ratio"]]    = (
        sum(1 for i in range(n - 1) if seq[i] == seq[i + 1]) / (n - 1)
        if n > 1 else 0.0
    )
    for s in vocab["syscalls"]:
        arr[feat_idx[f"freq_count::{s}"]] = sc_counts.get(s, 0) / n
    for b in vocab["bigrams"]:
        arr[feat_idx[f"seq2_norm::{b[0]}->{b[1]}"]] = (
            bg_counts.get(b, 0) / max(bg_total, 1))
    for t in vocab["trigrams"]:
        arr[feat_idx[f"seq3_norm::{t[0]}->{t[1]}->{t[2]}"]] = (
            tg_counts.get(t, 0) / max(tg_total, 1))
    for b in vocab["bigrams"]:
        a_cnt = sc_counts.get(b[0], 0)
        arr[feat_idx[f"trans_norm::{b[0]}->{b[1]}"]] = (
            bg_counts.get(b, 0) / max(a_cnt, 1))
    pt_counts = Counter(chunk["path_type"].dropna().tolist())
    for pt in vocab["path_types"]:
        arr[feat_idx[f"ctx_path_type::{pt}"]] = pt_counts.get(pt, 0) / n
    return arr


def build_feature_matrix(csv_paths, vocab, subtype_map):
    feat_names = _get_feature_names(vocab)
    feat_idx   = {name: i for i, name in enumerate(feat_names)}
    n_feats    = len(feat_names)
    arrays, session_ids, subtypes = [], [], []
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
    X = np.vstack(arrays)
    out = pd.DataFrame(X, columns=feat_names)
    out["session_id"] = session_ids
    out["subtype"]    = subtypes
    return out


# ── feature group assignment ──────────────────────────────────────────────────

def group_of(col: str) -> str:
    if col in SCALAR_COLS: return "scalars"
    if col.startswith("freq_count::"): return "freq"
    if col.startswith("seq2_norm::") or col.startswith("seq3_norm::"): return "seq"
    if col.startswith("trans_norm::"): return "trans"
    if col.startswith("ctx_path_type::"): return "ctx"
    return "other"


# ── permutation importance ────────────────────────────────────────────────────

def permutation_drop_group(clf, X_test: np.ndarray, y_test: np.ndarray,
                            col_indices: list, seed: int = 0) -> float:
    """Shuffle all columns in col_indices jointly, return accuracy drop."""
    rng = np.random.RandomState(seed)
    baseline = accuracy_score(y_test, clf.predict(X_test))
    X_perm = X_test.copy()
    perm = rng.permutation(X_test.shape[0])
    for ci in col_indices:
        X_perm[:, ci] = X_test[perm, ci]
    perturbed = accuracy_score(y_test, clf.predict(X_perm))
    return baseline - perturbed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir",    required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_paths = list_csvs(args.raw_dir)

    meta = collect_session_meta(csv_paths)
    subtype_map = dict(zip(meta["session_id"], meta["subtype"]))
    class_names = sorted(meta["subtype"].unique().tolist())
    label_enc   = {c: i for i, c in enumerate(class_names)}
    n_classes   = len(class_names)
    print(f"Classes: {class_names}")

    train_sids, test_sids = train_test_split(
        meta["session_id"], test_size=0.3,
        stratify=meta["subtype"], random_state=42,
    )
    train_meta = meta[meta["session_id"].isin(train_sids)].reset_index(drop=True)
    test_meta  = meta[meta["session_id"].isin(test_sids)].reset_index(drop=True)

    print("Building vocab + feature matrices …")
    vocab = build_vocab(csv_paths)
    train_df = build_feature_matrix(train_meta["path"].tolist(), vocab, subtype_map)
    test_df  = build_feature_matrix(test_meta["path"].tolist(),  vocab, subtype_map)

    feat_cols = [c for c in train_df.columns if c not in {"session_id", "subtype"}]
    X_train = train_df[feat_cols].values
    y_train = train_df["subtype"].map(label_enc).values
    X_test  = test_df[feat_cols].values
    y_test  = test_df["subtype"].map(label_enc).values

    # Group feature columns
    group_to_cols: dict = {g: [] for g in ["scalars", "freq", "seq", "trans", "ctx"]}
    col_to_idx = {c: i for i, c in enumerate(feat_cols)}
    for c in feat_cols:
        g = group_of(c)
        if g in group_to_cols:
            group_to_cols[g].append(col_to_idx[c])
    for g, idxs in group_to_cols.items():
        print(f"  {g}: {len(idxs)} features")

    # ── 1. Gini importance grouped ────────────────────────────────────────────
    print("\nTraining full-fusion RF …")
    clf_full = RandomForestClassifier(**RF_PARAMS)
    clf_full.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, clf_full.predict(X_test))
    print(f"  full-fusion test acc: {test_acc:.4f}")

    gini_rows = []
    total_gini = clf_full.feature_importances_.sum()
    for g, idxs in group_to_cols.items():
        s = float(clf_full.feature_importances_[idxs].sum())
        gini_rows.append({"group": g, "sum_importance": s,
                          "pct_of_total": s / total_gini if total_gini > 0 else 0.0,
                          "n_features": len(idxs)})
    gini_df = pd.DataFrame(gini_rows).sort_values("sum_importance", ascending=False)
    gini_csv = os.path.join(args.output_dir, "gini_importance_grouped.csv")
    gini_df.to_csv(gini_csv, index=False)
    print(f"\nGini importance grouped →\n{gini_df.to_string(index=False)}")

    # ── 2. Global permutation importance per group ────────────────────────────
    print("\nGlobal permutation importance (shuffle each group, 5 seeds) …")
    perm_rows = []
    for g, idxs in group_to_cols.items():
        if not idxs:
            continue
        drops = [permutation_drop_group(clf_full, X_test, y_test, idxs, seed=s)
                 for s in PERMUTATION_SEEDS]
        perm_rows.append({"group": g,
                          "mean_acc_drop": float(np.mean(drops)),
                          "std_acc_drop":  float(np.std(drops))})
        print(f"  {g:<10}  mean drop={np.mean(drops):.4f}  "
              f"std={np.std(drops):.4f}")
    perm_df = pd.DataFrame(perm_rows).sort_values("mean_acc_drop", ascending=False)
    perm_csv = os.path.join(args.output_dir, "permutation_importance_global.csv")
    perm_df.to_csv(perm_csv, index=False)

    # ── 3. Per-class permutation importance (one-vs-rest binary RFs) ──────────
    print("\nPer-class permutation importance (one-vs-rest binary RFs) …")
    per_class_rows = []
    for cls_idx, cls_name in enumerate(class_names):
        y_train_bin = (y_train == cls_idx).astype(int)
        y_test_bin  = (y_test  == cls_idx).astype(int)
        if y_train_bin.sum() == 0 or y_test_bin.sum() == 0:
            print(f"  skip {cls_name}: no positive samples")
            continue
        clf_bin = RandomForestClassifier(**RF_PARAMS)
        clf_bin.fit(X_train, y_train_bin)
        acc_base = accuracy_score(y_test_bin, clf_bin.predict(X_test))
        print(f"  [{cls_name}] binary RF base acc={acc_base:.4f}")
        for g, idxs in group_to_cols.items():
            if not idxs:
                continue
            drops = [permutation_drop_group(clf_bin, X_test, y_test_bin, idxs, seed=s)
                     for s in PERMUTATION_SEEDS]
            per_class_rows.append({"class": cls_name, "group": g,
                                   "mean_acc_drop": float(np.mean(drops)),
                                   "std_acc_drop":  float(np.std(drops))})
    pc_df = pd.DataFrame(per_class_rows)
    pc_csv = os.path.join(args.output_dir, "permutation_importance_per_class.csv")
    pc_df.to_csv(pc_csv, index=False)

    # Pretty per-class summary
    print("\nPer-class permutation importance (mean acc drop):")
    if not pc_df.empty:
        pivot = pc_df.pivot(index="group", columns="class", values="mean_acc_drop")
        pivot = pivot.reindex(["scalars", "freq", "seq", "trans", "ctx"])
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\nGini    → {gini_csv}")
    print(f"Perm    → {perm_csv}")
    print(f"PerCls  → {pc_csv}")


if __name__ == "__main__":
    main()
