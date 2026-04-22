#!/usr/bin/env python3
"""
Leave-One-Run-Out (LOUO) cross-validation experiment.

Each data file is named {software}_{subtype}_r{N}.csv where N=1..20.
Run N contributes exactly 4 sessions (one per class: normal, bulk_export,
project_scan, source_copy).  LOUO folds: for each run N, train on all 76
sessions from runs != N, test on the 4 sessions from run N.  This simulates
deploying the classifier to a completely new user whose activity was never
seen during training.

File naming assumption: basename must end in _r{N}.csv (e.g. freecad_normal_r12.csv).

Inputs:
  --raw-dir     directory containing *.csv session files
  --output-dir  destination for output CSVs

Outputs:
  louo_results.csv    — one row per (run, mode): session_accuracy, macro_f1,
                        recall_{class} for each class
  louo_summary.csv    — mean±std of session_accuracy and macro_f1 per mode
                        across all 20 folds

Usage:
  python fusion_louo.py --raw-dir data/raw/freecad_only \\
      --output-dir models/fusion_advantage/louo/freecad
  python fusion_louo.py --raw-dir data/raw/kicad_only \\
      --output-dir models/fusion_advantage/louo/kicad
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
from sklearn.metrics import accuracy_score, f1_score, recall_score

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


# ── run number extraction ─────────────────────────────────────────────────────

def extract_run_number(path: str) -> int:
    """Extract run number N from filename like freecad_normal_r12.csv → 12."""
    m = re.search(r'_r(\d+)\.csv$', os.path.basename(path))
    if m is None:
        raise ValueError(f"Cannot extract run number from: {path}")
    return int(m.group(1))


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


# ── session evaluation (multi-class) ─────────────────────────────────────────

def session_eval_multiclass(y_true: np.ndarray, y_pred: np.ndarray,
                            sids: np.ndarray, n_classes: int) -> dict:
    session_true: dict  = {}
    session_preds: dict = defaultdict(list)
    for sid, yt, yp in zip(sids, y_true, y_pred):
        if sid not in session_true:
            session_true[sid] = yt
        session_preds[sid].append(yp)
    yt_s, yp_s = [], []
    for sid in session_true:
        yt_s.append(session_true[sid])
        yp_s.append(Counter(session_preds[sid]).most_common(1)[0][0])
    yt_arr = np.array(yt_s)
    yp_arr = np.array(yp_s)
    labels   = list(range(n_classes))
    acc      = float(accuracy_score(yt_arr, yp_arr))
    macro_f1 = float(f1_score(yt_arr, yp_arr, average="macro",
                              labels=labels, zero_division=0))
    per_cls  = recall_score(yt_arr, yp_arr, average=None,
                            labels=labels, zero_division=0)
    return {"session_accuracy": acc, "macro_f1": macro_f1,
            "per_class_recall": per_cls.tolist()}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Leave-One-Run-Out cross-validation")
    parser.add_argument("--raw-dir",    required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = list_csvs(args.raw_dir)
    print(f"Found {len(csv_paths)} CSV files")

    meta        = collect_session_meta(csv_paths)
    meta["run"] = meta["path"].apply(extract_run_number)

    subtype_map = dict(zip(meta["session_id"], meta["subtype"]))
    class_names = sorted(meta["subtype"].unique().tolist())
    label_enc   = {c: i for i, c in enumerate(class_names)}
    n_classes   = len(class_names)
    print(f"  Classes ({n_classes}): {class_names}")

    run_numbers = sorted(meta["run"].unique().tolist())
    print(f"  Runs: {run_numbers}  ({len(run_numbers)} folds)")

    print("Building vocabulary from all CSVs …")
    vocab = build_vocab(csv_paths)

    print("Building full feature matrix for all sessions …")
    full_df = build_feature_matrix(csv_paths, vocab, subtype_map)
    print(f"  Total chunks: {len(full_df)}")

    results = []

    for run_number in run_numbers:
        test_sids  = meta[meta["run"] == run_number]["session_id"].tolist()
        train_sids = meta[meta["run"] != run_number]["session_id"].tolist()

        train_df  = full_df[full_df["session_id"].isin(train_sids)].reset_index(drop=True)
        test_df   = full_df[full_df["session_id"].isin(test_sids)].reset_index(drop=True)

        y_train   = train_df["subtype"].map(label_enc).values
        y_test    = test_df["subtype"].map(label_enc).values
        sids_test = test_df["session_id"].values

        print(f"\nFold run={run_number}  train_chunks={len(train_df)}  "
              f"test_chunks={len(test_df)}  test_sessions={len(test_sids)}")

        for mode in ISOLATION_MODES:
            cols   = select_columns(train_df, mode)
            clf    = RandomForestClassifier(**RF_PARAMS)
            clf.fit(train_df[cols].values, y_train)
            y_pred = clf.predict(test_df[cols].values)

            m = session_eval_multiclass(y_test, y_pred, sids_test, n_classes)
            row = {
                "run":              run_number,
                "mode":             mode,
                "session_accuracy": m["session_accuracy"],
                "macro_f1":         m["macro_f1"],
            }
            for i, cls in enumerate(class_names):
                row[f"recall_{cls}"] = m["per_class_recall"][i]
            results.append(row)
            print(f"  [{mode:>8}] acc={m['session_accuracy']:.3f}  "
                  f"f1={m['macro_f1']:.3f}")

    results_df = pd.DataFrame(results)
    raw_csv = os.path.join(args.output_dir, "louo_results.csv")
    results_df.to_csv(raw_csv, index=False)
    print(f"\nRaw results → {raw_csv}")

    agg_acc = (results_df.groupby("mode")["session_accuracy"]
               .agg(["mean", "std"]).reset_index()
               .rename(columns={"mean": "acc_mean", "std": "acc_std"}))
    agg_f1  = (results_df.groupby("mode")["macro_f1"]
               .agg(["mean", "std"]).reset_index()
               .rename(columns={"mean": "f1_mean", "std": "f1_std"}))
    summary = agg_acc.merge(agg_f1, on="mode")
    summary["acc_std"] = summary["acc_std"].fillna(0.0)
    summary["f1_std"]  = summary["f1_std"].fillna(0.0)
    sum_csv = os.path.join(args.output_dir, "louo_summary.csv")
    summary.to_csv(sum_csv, index=False)
    print(f"Summary → {sum_csv}")

    # Console summary table
    print("\n" + "=" * 64)
    print(f"LOUO 4-class session accuracy — {args.raw_dir.split('/')[-1]}")
    print(f"{'mode':<10}  {'acc_mean':>8}  {'acc_std':>8}  "
          f"{'f1_mean':>8}  {'f1_std':>8}")
    print("-" * 64)
    for mode in ISOLATION_MODES:
        row = summary[summary["mode"] == mode]
        if len(row):
            am = row["acc_mean"].values[0]
            as_ = row["acc_std"].values[0]
            fm = row["f1_mean"].values[0]
            fs = row["f1_std"].values[0]
            print(f"{mode:<10}  {am:>8.3f}  {as_:>8.3f}  {fm:>8.3f}  {fs:>8.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
