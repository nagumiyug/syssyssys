#!/usr/bin/env python3
"""
Data efficiency study: how does each feature mode degrade as training data shrinks?

Hypothesis: fusion degrades more gracefully because its diverse signals are
partially redundant — losing data hurts single-feature modes disproportionately.

Grid:
  train_fraction ∈ {0.10, 0.20, 0.30, 0.50, 0.70, 1.00}
  n_seeds        = 5  (random subsampling seeds → mean ± std)
  mode           ∈ {scalars, freq, seq, trans, ctx, fusion}

Implementation:
  - Vocab and full feature matrices are built ONCE before the experiment loop.
  - Subsampling is done by filtering session IDs from the pre-built matrix,
    not by re-loading CSVs — this avoids O(modes × fractions × seeds) redundant
    feature extraction (critical for KiCad with ~8000 train chunks).
  - fraction=1.0 uses all training sessions; only one seed is run (std=0 by
    definition since there is no randomness in the data selection).

Output per software:
  data_efficiency_results.csv  — one row per (mode, fraction, seed)
  data_efficiency_summary.csv  — aggregated mean ± std per (mode, fraction)

Usage:
  python fusion_data_efficiency.py --raw-dir data/raw/freecad_only \
      --output-dir models/fusion_advantage/data_efficiency/freecad
  python fusion_data_efficiency.py --raw-dir data/raw/kicad_only \
      --output-dir models/fusion_advantage/data_efficiency/kicad
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

# ── constants ─────────────────────────────────────────────────────────────────

NOISE_SYSCALLS  = {"mmap", "munmap", "mprotect"}
NORMAL_LABEL    = "normal"
CHUNK_SIZE      = 1000

TRAIN_FRACTIONS = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
N_SEEDS         = 5
SUBSAMPLE_SEEDS = list(range(N_SEEDS))   # 0 … 4
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
        usecols=["syscall_name", "path_type", "label", "session_id"],
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


# ── feature extraction (numpy, memory-efficient) ──────────────────────────────

def _entropy(counts_dict: Counter) -> float:
    total = sum(counts_dict.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts_dict.values()), dtype=float) / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _get_feature_names(vocab: dict) -> list[str]:
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


def build_feature_matrix(csv_paths: list[str], vocab: dict) -> pd.DataFrame:
    """Build full feature matrix for the given sessions."""
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

        for start in range(0, len(df) - CHUNK_SIZE + 1, CHUNK_SIZE):
            chunk = df.iloc[start: start + CHUNK_SIZE]
            arrays.append(_extract_np(chunk, vocab, feat_idx, n_feats))
            session_ids.append(session_id)
            labels.append(is_abnormal)
        del df

    X      = np.vstack(arrays)
    result = pd.DataFrame(X, columns=feat_names)
    result["session_id"] = session_ids
    result["label"]      = labels
    return result


# ── column selection by mode ──────────────────────────────────────────────────

def select_columns(df: pd.DataFrame, mode: str) -> list[str]:
    meta     = {"session_id", "label"}
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


# ── session-level evaluation ──────────────────────────────────────────────────

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
    tn, fp = cm[0, 0], cm[0, 1]
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {"session_accuracy": acc, "session_fpr": fpr}


# ── session-level subsampling from pre-built matrix ───────────────────────────

def subsample_sessions(full_train_df: pd.DataFrame, fraction: float,
                       seed: int) -> pd.DataFrame:
    """
    Return a subset of full_train_df containing only chunks from a stratified
    random sample of training sessions (sampled at session level, not chunk level).
    """
    session_info = (
        full_train_df[["session_id", "label"]]
        .drop_duplicates("session_id")
        .reset_index(drop=True)
    )
    normal_sids = session_info.loc[session_info["label"] == 0, "session_id"].tolist()
    abnorm_sids = session_info.loc[session_info["label"] == 1, "session_id"].tolist()

    rng = np.random.RandomState(seed)
    rng.shuffle(normal_sids)
    rng.shuffle(abnorm_sids)

    n_keep_normal = max(1, int(len(normal_sids) * fraction))
    n_keep_abnorm = max(1, int(len(abnorm_sids) * fraction))

    kept = set(normal_sids[:n_keep_normal]) | set(abnorm_sids[:n_keep_abnorm])
    return full_train_df[full_train_df["session_id"].isin(kept)].reset_index(drop=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data efficiency study")
    parser.add_argument("--raw-dir",      required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--test-size",    type=float, default=0.3)
    parser.add_argument("--random-state", type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = list_csvs(args.raw_dir)
    print(f"Found {len(csv_paths)} CSV files in {args.raw_dir}")

    # Fixed session-level train/test split (identical across all experiments)
    meta = collect_session_meta(csv_paths)
    train_sids, test_sids = train_test_split(
        meta["session_id"],
        test_size=args.test_size,
        stratify=meta["label"],
        random_state=args.random_state,
    )
    train_meta = meta[meta["session_id"].isin(train_sids)].reset_index(drop=True)
    test_meta  = meta[meta["session_id"].isin(test_sids)].reset_index(drop=True)
    print(f"  train sessions={len(train_meta)}  test sessions={len(test_meta)}")

    # Vocab built on ALL data (no out-of-vocab features at small fractions)
    print("Building vocabulary from all data …")
    vocab = build_vocab(csv_paths)
    print(f"  syscalls={len(vocab['syscalls'])}  bigrams={len(vocab['bigrams'])}  "
          f"trigrams={len(vocab['trigrams'])}")

    # Build full train and test matrices ONCE — subsampling is done by filtering
    # session IDs in-memory, avoiding redundant CSV I/O and feature extraction.
    print("Building full training feature matrix (built once, reused for all fractions) …")
    full_train_df = build_feature_matrix(train_meta["path"].tolist(), vocab)
    print(f"  full_train_chunks={len(full_train_df)}")

    print("Building test feature matrix …")
    test_df = build_feature_matrix(test_meta["path"].tolist(), vocab)
    print(f"  test_chunks={len(test_df)}")

    all_results = []

    for mode in ISOLATION_MODES:
        feat_cols = select_columns(full_train_df, mode)
        X_test    = test_df[feat_cols].values
        y_test    = test_df["label"].values
        sids_test = test_df["session_id"].values
        print(f"\n[{mode}] {len(feat_cols)} features")

        for fraction in TRAIN_FRACTIONS:
            # fraction=1.0 is deterministic (no subsampling randomness) → one seed
            seeds_to_run = [0] if fraction >= 1.0 else SUBSAMPLE_SEEDS

            accs, fprs = [], []
            for seed in seeds_to_run:
                if fraction >= 1.0:
                    sub_df = full_train_df
                else:
                    sub_df = subsample_sessions(full_train_df, fraction, seed)

                X_train = sub_df[feat_cols].values
                y_train = sub_df["label"].values

                clf = RandomForestClassifier(**RF_PARAMS)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                m = session_level_eval(y_test, y_pred, sids_test)
                accs.append(m["session_accuracy"])
                fprs.append(m["session_fpr"])

                n_train_sessions = sub_df["session_id"].nunique()
                all_results.append({
                    "mode":             mode,
                    "train_fraction":   fraction,
                    "seed":             seed,
                    "n_train_sessions": n_train_sessions,
                    "n_train_chunks":   len(sub_df),
                    "session_accuracy": m["session_accuracy"],
                    "session_fpr":      m["session_fpr"],
                })

            mean_acc = float(np.mean(accs))
            std_acc  = float(np.std(accs)) if len(accs) > 1 else 0.0
            n_sess   = int(len(full_train_df["session_id"].unique()) * fraction)
            print(f"  fraction={fraction:.2f}  ~{n_sess:>3} sessions  "
                  f"acc={mean_acc:.3f}±{std_acc:.3f}")

    # ── save raw results ──
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, "data_efficiency_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nRaw results saved → {csv_path}")

    # ── aggregated summary: mean ± std over seeds ──
    agg_acc = (
        results_df.groupby(["mode", "train_fraction"])["session_accuracy"]
        .agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "acc_mean", "std": "acc_std"})
    )
    agg_fpr = (
        results_df.groupby(["mode", "train_fraction"])["session_fpr"]
        .agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "fpr_mean", "std": "fpr_std"})
    )
    summary_df = agg_acc.merge(agg_fpr, on=["mode", "train_fraction"])
    summary_df["acc_std"] = summary_df["acc_std"].fillna(0.0)
    summary_df["fpr_std"] = summary_df["fpr_std"].fillna(0.0)
    summary_csv = os.path.join(args.output_dir, "data_efficiency_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved       → {summary_csv}")

    # ── console table ──
    print("\n" + "=" * 80)
    print("Session accuracy mean (±std over seeds | std=0 at fraction=1.0)")
    header = f"{'mode':<10}" + "".join(f"  {f:.0%}".rjust(12) for f in TRAIN_FRACTIONS)
    print(header)
    print("-" * 80)
    for mode in ISOLATION_MODES:
        row_str = f"{mode:<10}"
        for frac in TRAIN_FRACTIONS:
            row = summary_df[(summary_df["mode"] == mode) &
                             (summary_df["train_fraction"] == frac)]
            if len(row):
                m, s = row["acc_mean"].values[0], row["acc_std"].values[0]
                row_str += f"  {m:.3f}±{s:.3f}"
            else:
                row_str += "              —"
        print(row_str)
    print("=" * 80)


if __name__ == "__main__":
    main()
