#!/usr/bin/env python3
"""
Low-data leave-one-out ablation for fusion model.

Motivation: in standard 70/30 (~56 train sessions), every feature group alone is
sufficient to hit 100% accuracy, making necessity claims impossible.
Under low-data training (2, 3, 5, 10 sessions per class), feature complementarity
becomes visible — removing any genuinely useful component should cause measurable
degradation.

For each n_per_class ∈ {2, 3, 5, 10}:
  - Sample n sessions per class from the training pool (5 seeds × different samples)
  - Train 6 fusion variants (full + 5 minus-X) on this tiny train set
  - Evaluate on a held-out FIXED test set of 24 sessions (70/30 split test portion)
  - Record per-class precision / recall / F1 + aggregate accuracy

A feature group is "necessary" under low data if removing it causes:
  a) a drop in aggregate accuracy, OR
  b) a drop in specific class recall (revealing class-specific contribution)

Output:
  lowdata_loo_results.csv          — per (software, n_per_class, seed, config, metric)
  lowdata_loo_summary.csv          — mean±std aggregated over seeds
  lowdata_loo_class_breakdown.csv  — per-class recall per config per training size

Usage:
  python fusion_lowdata_loo.py --raw-dir data/raw/freecad_only \\
      --output-dir models/fusion_advantage/lowdata_loo/freecad
  python fusion_lowdata_loo.py --raw-dir data/raw/kicad_only \\
      --output-dir models/fusion_advantage/lowdata_loo/kicad
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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

# ── constants ─────────────────────────────────────────────────────────────────

NOISE_SYSCALLS = {"mmap", "munmap", "mprotect"}
NORMAL_LABEL   = "normal"
CHUNK_SIZE     = 1000

N_PER_CLASS_LIST = [2, 3, 5, 10]
N_SAMPLING_SEEDS = 5

SCALAR_COLS = {"syscall_entropy", "transition_entropy", "self_loop_ratio"}
CONFIGS = [
    "fusion_full",
    "fusion_minus_scalars",
    "fusion_minus_freq",
    "fusion_minus_seq",
    "fusion_minus_trans",
    "fusion_minus_ctx",
]

RF_PARAMS = dict(
    n_estimators=300, max_depth=9, class_weight="balanced",
    random_state=42, n_jobs=-1,
)


# ── data loading (reused from fusion_leave_one_out.py) ────────────────────────

def list_csvs(raw_dir: str) -> list:
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        sys.exit(f"No CSV files in {raw_dir}")
    return paths


def load_csv_filtered(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, dtype=str,
        usecols=["syscall_name", "path_type", "label", "session_id", "scenario"],
    )
    return df[~df["syscall_name"].isin(NOISE_SYSCALLS)].reset_index(drop=True)


def _scenario_to_subtype(scenario: str) -> str:
    return re.sub(r"_linux_r\d+$", "", scenario)


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
        arr[feat_idx[f"seq2_norm::{b[0]}->{b[1]}"]] = (
            bg_counts.get(b, 0) / max(bg_total, 1)
        )
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


def select_columns_for_config(df: pd.DataFrame, config: str) -> list:
    all_cols = [c for c in df.columns if c not in {"session_id", "subtype"}]

    def is_scalar(c):  return c in SCALAR_COLS
    def is_freq(c):    return c.startswith("freq_count::")
    def is_seq(c):     return c.startswith("seq2_norm::") or c.startswith("seq3_norm::")
    def is_trans(c):   return c.startswith("trans_norm::")
    def is_ctx(c):     return c.startswith("ctx_path_type::")

    if config == "fusion_full":          return all_cols
    if config == "fusion_minus_scalars": return [c for c in all_cols if not is_scalar(c)]
    if config == "fusion_minus_freq":    return [c for c in all_cols if not is_freq(c)]
    if config == "fusion_minus_seq":     return [c for c in all_cols if not is_seq(c)]
    if config == "fusion_minus_trans":   return [c for c in all_cols if not is_trans(c)]
    if config == "fusion_minus_ctx":     return [c for c in all_cols if not is_ctx(c)]
    raise ValueError(config)


# ── session-level evaluation with per-class metrics ───────────────────────────

def session_eval_per_class(y_true_chunk: np.ndarray, y_pred_chunk: np.ndarray,
                            sids: np.ndarray, n_classes: int) -> dict:
    session_true: dict  = {}
    session_preds: dict = defaultdict(list)
    for sid, yt, yp in zip(sids, y_true_chunk, y_pred_chunk):
        if sid not in session_true:
            session_true[sid] = yt
        session_preds[sid].append(yp)
    yt_s, yp_s = [], []
    for sid in session_true:
        yt_s.append(session_true[sid])
        yp_s.append(Counter(session_preds[sid]).most_common(1)[0][0])
    yt = np.array(yt_s)
    yp = np.array(yp_s)
    labels = list(range(n_classes))
    acc = float(accuracy_score(yt, yp))
    prec, rec, f1, _ = precision_recall_fscore_support(
        yt, yp, labels=labels, average=None, zero_division=0,
    )
    return {"accuracy": acc,
            "per_class_precision": prec.tolist(),
            "per_class_recall":    rec.tolist(),
            "per_class_f1":        f1.tolist()}


def sample_low_data_train(meta: pd.DataFrame, n_per_class: int,
                           seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    sampled = []
    for subtype, grp in meta.groupby("subtype"):
        n = min(n_per_class, len(grp))
        idx = rng.choice(len(grp), size=n, replace=False)
        sampled.append(grp.iloc[idx])
    return pd.concat(sampled, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir",    required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_paths = list_csvs(args.raw_dir)
    print(f"Found {len(csv_paths)} CSV files")

    meta = collect_session_meta(csv_paths)
    subtype_map = dict(zip(meta["session_id"], meta["subtype"]))
    class_names = sorted(meta["subtype"].unique().tolist())
    label_enc   = {c: i for i, c in enumerate(class_names)}
    n_classes   = len(class_names)
    print(f"  Classes ({n_classes}): {class_names}")

    # Fixed 70/30 split — use the 30% test portion as the held-out evaluation set
    train_sids_pool, test_sids = train_test_split(
        meta["session_id"], test_size=0.3,
        stratify=meta["subtype"], random_state=42,
    )
    train_pool_meta = meta[meta["session_id"].isin(train_sids_pool)].reset_index(drop=True)
    test_meta       = meta[meta["session_id"].isin(test_sids)].reset_index(drop=True)
    print(f"  train pool: {len(train_pool_meta)}  "
          f"held-out test: {len(test_meta)}")

    print("Building vocab …")
    vocab = build_vocab(csv_paths)

    print("Building held-out test feature matrix once …")
    test_df   = build_feature_matrix(test_meta["path"].tolist(), vocab, subtype_map)
    y_test    = test_df["subtype"].map(label_enc).values
    sids_test = test_df["session_id"].values

    results           = []
    class_breakdowns  = []

    for n_per_class in N_PER_CLASS_LIST:
        print(f"\n=== n_per_class={n_per_class} ===")
        for seed in range(N_SAMPLING_SEEDS):
            sampled_meta = sample_low_data_train(train_pool_meta, n_per_class, seed)
            print(f"  seed={seed}: {len(sampled_meta)} train sessions")
            train_df = build_feature_matrix(
                sampled_meta["path"].tolist(), vocab, subtype_map
            )
            if train_df.empty:
                continue
            y_train = train_df["subtype"].map(label_enc).values
            for config in CONFIGS:
                cols = select_columns_for_config(train_df, config)
                clf = RandomForestClassifier(**RF_PARAMS)
                clf.fit(train_df[cols].values, y_train)
                y_pred = clf.predict(test_df[cols].values)
                m = session_eval_per_class(y_test, y_pred, sids_test, n_classes)
                results.append({
                    "n_per_class": n_per_class, "seed": seed, "config": config,
                    "n_features":  len(cols),
                    "accuracy":    m["accuracy"],
                })
                for cls_idx, cls_name in enumerate(class_names):
                    class_breakdowns.append({
                        "n_per_class": n_per_class, "seed": seed, "config": config,
                        "class":       cls_name,
                        "precision":   m["per_class_precision"][cls_idx],
                        "recall":      m["per_class_recall"][cls_idx],
                        "f1":          m["per_class_f1"][cls_idx],
                    })
                print(f"    [{config:<22}] acc={m['accuracy']:.3f}  "
                      f"recall={[f'{r:.2f}' for r in m['per_class_recall']]}")

    raw_df = pd.DataFrame(results)
    cls_df = pd.DataFrame(class_breakdowns)

    raw_csv = os.path.join(args.output_dir, "lowdata_loo_results.csv")
    raw_df.to_csv(raw_csv, index=False)

    summary = (raw_df.groupby(["n_per_class", "config"])["accuracy"]
               .agg(["mean", "std"]).reset_index()
               .rename(columns={"mean": "acc_mean", "std": "acc_std"}))
    summary["acc_std"] = summary["acc_std"].fillna(0.0)
    sum_csv = os.path.join(args.output_dir, "lowdata_loo_summary.csv")
    summary.to_csv(sum_csv, index=False)

    class_summary = (cls_df.groupby(["n_per_class", "config", "class"])
                          [["precision", "recall", "f1"]]
                          .mean().reset_index())
    cls_csv = os.path.join(args.output_dir, "lowdata_loo_class_breakdown.csv")
    class_summary.to_csv(cls_csv, index=False)

    print(f"\nRaw          → {raw_csv}")
    print(f"Summary      → {sum_csv}")
    print(f"Class break  → {cls_csv}")

    # Console summary
    print("\n" + "=" * 80)
    print("Aggregate accuracy (mean ± std over seeds)")
    print("=" * 80)
    pivot = summary.pivot(index="config", columns="n_per_class", values="acc_mean")
    pivot = pivot.reindex(CONFIGS)
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 80)
    print("Per-class recall (mean over seeds) at lowest n_per_class")
    print("=" * 80)
    lowest = min(N_PER_CLASS_LIST)
    sub = class_summary[class_summary["n_per_class"] == lowest]
    recall_pivot = sub.pivot(index="config", columns="class", values="recall")
    recall_pivot = recall_pivot.reindex(CONFIGS)
    print(recall_pivot.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
