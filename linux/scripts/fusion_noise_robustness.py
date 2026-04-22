#!/usr/bin/env python3
"""
Noise robustness study: how well does each feature mode tolerate corrupted syscalls?

Hypothesis: fusion degrades more gracefully than single-feature modes because
diverse signals compensate for each other — noise that destroys one group's
signal leaves other groups largely intact.

Protocol:
  - Train on clean data (no noise).
  - Evaluate on test data where noise_level% of syscalls are randomly replaced
    with other syscalls drawn uniformly from the training vocabulary.
  - 5 independent noise seeds per noise level → mean ± std.

Grid:
  noise_level ∈ {0.00, 0.05, 0.10, 0.20, 0.30, 0.50}
  n_seeds     = 5
  mode        ∈ {scalars, freq, seq, trans, ctx, fusion}

The training feature matrix is built once on clean data.
The test feature matrix is rebuilt for each (noise_level, seed) combination,
but since the test set is smaller (30%), this is fast.

Output per software:
  noise_robustness_results.csv  — one row per (mode, noise_level, seed)
  noise_robustness_summary.csv  — aggregated mean ± std per (mode, noise_level)

Usage:
  python fusion_noise_robustness.py --raw-dir data/raw/freecad_only \
      --output-dir models/fusion_advantage/noise_robustness/freecad
  python fusion_noise_robustness.py --raw-dir data/raw/kicad_only \
      --output-dir models/fusion_advantage/noise_robustness/kicad
"""

import argparse
import glob
import os
import sys
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ── constants ─────────────────────────────────────────────────────────────────

NOISE_SYSCALLS  = {"mmap", "munmap", "mprotect"}
NORMAL_LABEL    = "normal"
CHUNK_SIZE      = 1000

NOISE_LEVELS    = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50]
N_SEEDS         = 5
NOISE_SEEDS     = list(range(N_SEEDS))   # 0 … 4
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


# ── noise injection ───────────────────────────────────────────────────────────

def inject_noise(df: pd.DataFrame, noise_level: float,
                 syscall_vocab: list[str], rng: np.random.RandomState) -> pd.DataFrame:
    """
    Replace `noise_level` fraction of syscall_name values with random entries
    drawn uniformly from syscall_vocab (in-vocabulary noise, not gibberish).
    path_type column is left unchanged — noise targets the syscall sequence only.
    """
    if noise_level <= 0.0:
        return df
    df = df.copy()
    n      = len(df)
    n_noisy = int(n * noise_level)
    if n_noisy == 0:
        return df
    indices = rng.choice(n, size=n_noisy, replace=False)
    replacements = rng.choice(syscall_vocab, size=n_noisy)
    df.iloc[indices, df.columns.get_loc("syscall_name")] = replacements
    return df


# ── feature extraction (numpy) ────────────────────────────────────────────────

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


def build_feature_matrix(csv_paths: list[str], vocab: dict,
                         noise_level: float = 0.0,
                         syscall_vocab: Optional[list[str]] = None,
                         rng: Optional[np.random.RandomState] = None) -> pd.DataFrame:
    """
    Build feature matrix. If noise_level > 0, inject in-vocabulary noise into
    each session before feature extraction.
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
        if noise_level > 0.0 and syscall_vocab is not None and rng is not None:
            df = inject_noise(df, noise_level, syscall_vocab, rng)

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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Noise robustness study")
    parser.add_argument("--raw-dir",      required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--test-size",    type=float, default=0.3)
    parser.add_argument("--random-state", type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = list_csvs(args.raw_dir)
    print(f"Found {len(csv_paths)} CSV files in {args.raw_dir}")

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

    # Vocab from ALL data
    print("Building vocabulary …")
    vocab = build_vocab(csv_paths)
    syscall_vocab = vocab["syscalls"]   # used for noise injection
    print(f"  syscalls={len(syscall_vocab)}  bigrams={len(vocab['bigrams'])}  "
          f"trigrams={len(vocab['trigrams'])}")

    # Train on clean data — one feature matrix, train all mode classifiers once
    print("Building clean training feature matrix …")
    train_df = build_feature_matrix(train_meta["path"].tolist(), vocab)
    print(f"  train_chunks={len(train_df)}")

    # Pre-train one RF per mode on clean data (reused across all noise levels)
    print("Pre-training classifiers on clean data …")
    trained_clfs: dict = {}
    for mode in ISOLATION_MODES:
        feat_cols = select_columns(train_df, mode)
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(train_df[feat_cols].values, train_df["label"].values)
        trained_clfs[mode] = (clf, feat_cols)
        print(f"  [{mode}] trained on {len(feat_cols)} features")

    del train_df   # free memory

    all_results = []

    for noise_level in NOISE_LEVELS:
        print(f"\nnoise_level = {noise_level:.0%}")

        # noise_level=0 is deterministic — run once, std=0
        seeds_to_run = [0] if noise_level == 0.0 else NOISE_SEEDS

        # Accumulate per-mode results across seeds for this noise level
        mode_accs: dict = {m: [] for m in ISOLATION_MODES}
        mode_fprs: dict = {m: [] for m in ISOLATION_MODES}

        for seed in seeds_to_run:
            rng = np.random.RandomState(seed)
            test_df = build_feature_matrix(
                test_meta["path"].tolist(), vocab,
                noise_level=noise_level,
                syscall_vocab=syscall_vocab,
                rng=rng,
            )

            for mode in ISOLATION_MODES:
                clf, feat_cols = trained_clfs[mode]
                y_pred = clf.predict(test_df[feat_cols].values)
                m = session_level_eval(
                    test_df["label"].values, y_pred,
                    test_df["session_id"].values,
                )
                mode_accs[mode].append(m["session_accuracy"])
                mode_fprs[mode].append(m["session_fpr"])

                all_results.append({
                    "mode":             mode,
                    "noise_level":      noise_level,
                    "seed":             seed,
                    "session_accuracy": m["session_accuracy"],
                    "session_fpr":      m["session_fpr"],
                })

            del test_df

        # Print summary for this noise level
        parts = []
        for mode in ISOLATION_MODES:
            m = np.mean(mode_accs[mode])
            s = np.std(mode_accs[mode]) if len(mode_accs[mode]) > 1 else 0.0
            parts.append(f"{mode}={m:.3f}±{s:.3f}")
        print("  " + "  ".join(parts))

    # ── save ──
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, "noise_robustness_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nRaw results saved → {csv_path}")

    agg_acc = (
        results_df.groupby(["mode", "noise_level"])["session_accuracy"]
        .agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "acc_mean", "std": "acc_std"})
    )
    agg_fpr = (
        results_df.groupby(["mode", "noise_level"])["session_fpr"]
        .agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "fpr_mean", "std": "fpr_std"})
    )
    summary_df = agg_acc.merge(agg_fpr, on=["mode", "noise_level"])
    summary_df["acc_std"] = summary_df["acc_std"].fillna(0.0)
    summary_df["fpr_std"] = summary_df["fpr_std"].fillna(0.0)
    summary_csv = os.path.join(args.output_dir, "noise_robustness_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved       → {summary_csv}")

    # ── console table ──
    print("\n" + "=" * 80)
    print("Session accuracy mean (±std over noise seeds)")
    header = f"{'mode':<10}" + "".join(f"  {nl:.0%}".rjust(11) for nl in NOISE_LEVELS)
    print(header)
    print("-" * 80)
    for mode in ISOLATION_MODES:
        row_str = f"{mode:<10}"
        for nl in NOISE_LEVELS:
            row = summary_df[(summary_df["mode"] == mode) &
                             (summary_df["noise_level"] == nl)]
            if len(row):
                m, s = row["acc_mean"].values[0], row["acc_std"].values[0]
                row_str += f"  {m:.3f}±{s:.3f}"
            else:
                row_str += "             —"
        print(row_str)
    print("=" * 80)


if __name__ == "__main__":
    main()
