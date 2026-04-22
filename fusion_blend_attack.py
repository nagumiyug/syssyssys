#!/usr/bin/env python3
"""
Mixed attack concentration study (blend attack).

Creates "partially infected" test sessions by taking normal test sessions and
replacing `blend_ratio` fraction of their syscalls with syscalls sampled from
real attack sessions.  Measures detection rate (fraction of sessions classified
as non-normal) at each blend level.  Tests whether fusion detects attacks
earlier than single feature modes.

Blend model: for a given attack_type and blend_ratio, every normal test session
is copied; then `blend_ratio` fraction of its (already-filtered) syscall_name
entries are replaced with syscalls drawn uniformly from a flat pool of all
syscalls observed in real sessions of that attack_type.  The ctx (path_type)
feature group is unaffected, so fusion/ctx are expected to stay robust at low
blend ratios while syscall-heavy modes diverge.

Grid:
  attack_type  ∈ {all non-normal subtypes discovered in raw_dir}
  blend_ratio  ∈ {0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00}
  n_seeds      = 5  (blend randomness; blend_ratio=0 → 1 seed)
  mode         ∈ {scalars, freq, seq, trans, ctx, fusion}

Full vocabulary built from all data; classifiers pre-trained once per mode on
clean training chunks; blended test features re-extracted per
(attack_type, blend_ratio, seed).

Output:
  blend_attack_results.csv   — raw (attack_type, mode, blend_ratio, seed) rows
  blend_attack_summary.csv   — aggregated mean±std per (attack_type, mode, blend_ratio)

Usage:
  python fusion_blend_attack.py --raw-dir data/raw/freecad_only \\
      --output-dir models/fusion_advantage/blend_attack/freecad
  python fusion_blend_attack.py --raw-dir data/raw/kicad_only \\
      --output-dir models/fusion_advantage/blend_attack/kicad
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
from sklearn.model_selection import train_test_split

# ── constants ─────────────────────────────────────────────────────────────────

NOISE_SYSCALLS  = {"mmap", "munmap", "mprotect"}
NORMAL_LABEL    = "normal"
CHUNK_SIZE      = 1000

BLEND_RATIOS    = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
N_SEEDS         = 5
BLEND_SEEDS     = list(range(N_SEEDS))
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


# ── blend-specific helpers ────────────────────────────────────────────────────

def build_attack_syscall_pool(csv_paths: list, subtype_filter: str) -> list:
    """Return a flat list of syscall_name values from all sessions of subtype_filter.

    Only sessions whose subtype (derived from scenario) matches subtype_filter
    are included.  NOISE_SYSCALLS are already removed by load_csv_filtered.
    """
    pool: list = []
    for p in csv_paths:
        # Peek at first row to check subtype before loading the full file
        peek = pd.read_csv(
            p, usecols=["label", "scenario"], dtype=str, nrows=1
        )
        label    = str(peek["label"].iloc[0])
        scenario = str(peek["scenario"].iloc[0])
        subtype  = _scenario_to_subtype(scenario) if label != NORMAL_LABEL else NORMAL_LABEL
        if subtype != subtype_filter:
            continue
        df = load_csv_filtered(p)
        pool.extend(df["syscall_name"].tolist())
        del df
    return pool


def build_blend_test_matrix(normal_raw_dfs: list, attack_pool: list,
                             vocab: dict, subtype_map: dict,
                             blend_ratio: float, seed: int) -> pd.DataFrame:
    """Re-extract features from blend-corrupted normal test DataFrames.

    Each normal session has `blend_ratio` fraction of its syscall_name entries
    replaced by syscalls drawn uniformly from `attack_pool`.  The subtype column
    in the returned DataFrame remains "normal" so that at blend_ratio=0 we can
    verify zero false-alarm rate; detection_rate is computed externally by
    checking whether prediction != normal_label_idx.

    Parameters
    ----------
    normal_raw_dfs : list of pd.DataFrame
        Raw (NOISE_SYSCALLS-filtered) DataFrames for normal test sessions.
    attack_pool : list of str
        Flat list of syscall_name strings drawn from real attack sessions of the
        target attack_type.
    vocab : dict
        Vocabulary dict as returned by build_vocab.
    subtype_map : dict
        session_id → subtype mapping (used to fill the subtype column).
    blend_ratio : float
        Fraction of syscalls to replace with attack_pool samples.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Chunk-level feature DataFrame with session_id and subtype columns.
    """
    feat_names = _get_feature_names(vocab)
    feat_idx   = {name: i for i, name in enumerate(feat_names)}
    n_feats    = len(feat_names)
    attack_pool_arr = np.array(attack_pool)
    arrays, session_ids, subtypes = [], [], []
    rng = np.random.RandomState(seed)
    for df_clean in normal_raw_dfs:
        if df_clean.empty:
            continue
        if blend_ratio > 0.0 and len(attack_pool_arr) > 0:
            df = df_clean.copy()
            n = len(df)
            n_inject = int(n * blend_ratio)
            idx          = rng.choice(n, size=n_inject, replace=False)
            replacements = rng.choice(len(attack_pool_arr), size=n_inject)
            sc_col = df.columns.get_loc("syscall_name")
            for ii, ri in zip(idx, replacements):
                df.iat[ii, sc_col] = attack_pool_arr[ri]
        else:
            df = df_clean
        session_id = str(df["session_id"].iloc[0])
        subtype    = subtype_map.get(session_id, NORMAL_LABEL)
        for start in range(0, len(df) - CHUNK_SIZE + 1, CHUNK_SIZE):
            chunk = df.iloc[start: start + CHUNK_SIZE]
            arrays.append(_extract_np(chunk, vocab, feat_idx, n_feats))
            session_ids.append(session_id)
            subtypes.append(subtype)
    if not arrays:
        return pd.DataFrame(columns=feat_names + ["session_id", "subtype"])
    X      = np.vstack(arrays)
    result = pd.DataFrame(X, columns=feat_names)
    result["session_id"] = session_ids
    result["subtype"]    = subtypes
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mixed attack concentration study")
    parser.add_argument("--raw-dir",    required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size",  type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_paths = list_csvs(args.raw_dir)
    print(f"Found {len(csv_paths)} CSV files")

    meta        = collect_session_meta(csv_paths)
    subtype_map = dict(zip(meta["session_id"], meta["subtype"]))
    class_names = sorted(meta["subtype"].unique().tolist())
    label_enc   = {c: i for i, c in enumerate(class_names)}
    n_classes   = len(class_names)
    print(f"  Classes ({n_classes}): {class_names}")

    train_sids, test_sids = train_test_split(
        meta["session_id"],
        test_size=args.test_size,
        stratify=meta["subtype"],
        random_state=42,
    )
    train_meta = meta[meta["session_id"].isin(train_sids)].reset_index(drop=True)
    test_meta  = meta[meta["session_id"].isin(test_sids)].reset_index(drop=True)
    print(f"  train={len(train_meta)}  test={len(test_meta)}")

    print("Building vocabulary …")
    vocab = build_vocab(csv_paths)

    print("Building clean training feature matrix (built once) …")
    train_df = build_feature_matrix(train_meta["path"].tolist(), vocab, subtype_map)
    y_train  = train_df["subtype"].map(label_enc).values
    print(f"  train_chunks={len(train_df)}")

    # Pre-load ONLY normal test raw DataFrames once — blend injected in-memory
    print("Pre-loading normal test raw DataFrames …")
    normal_test_meta = test_meta[test_meta["label"] == NORMAL_LABEL].reset_index(drop=True)
    normal_raw_dfs: list = []
    for p in normal_test_meta["path"].tolist():
        normal_raw_dfs.append(load_csv_filtered(p))
    print(f"  normal test sessions pre-loaded: {len(normal_raw_dfs)}")

    # Pre-train one classifier per mode on clean data
    print("Pre-training classifiers on clean data …")
    trained_clfs:  dict = {}
    feat_cols_map: dict = {}
    for mode in ISOLATION_MODES:
        cols = select_columns(train_df, mode)
        feat_cols_map[mode] = cols
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(train_df[cols].values, y_train)
        trained_clfs[mode] = clf
        print(f"  [{mode}] trained on {len(cols)} features")
    del train_df   # free memory

    # Build attack syscall pool per attack_type using ALL sessions (train+test)
    attack_types = [c for c in class_names if c != NORMAL_LABEL]
    print(f"\nBuilding attack syscall pools for: {attack_types}")
    attack_pools: dict = {}
    for at in attack_types:
        pool = build_attack_syscall_pool(csv_paths, at)
        attack_pools[at] = pool
        print(f"  [{at}] pool size: {len(pool)} syscalls")

    normal_label_idx = label_enc[NORMAL_LABEL]

    all_results = []

    for attack_type in attack_types:
        pool = attack_pools[attack_type]
        if not pool:
            print(f"\n[WARNING] Empty attack pool for {attack_type}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"attack_type={attack_type}")

        for blend_ratio in BLEND_RATIOS:
            seeds_to_run = [0] if blend_ratio == 0.0 else BLEND_SEEDS
            print(f"  blend_ratio={blend_ratio:.2f}  seeds={seeds_to_run}")

            for seed in seeds_to_run:
                blend_df = build_blend_test_matrix(
                    normal_raw_dfs, pool, vocab, subtype_map, blend_ratio, seed
                )
                if blend_df.empty:
                    continue

                sids_blend = blend_df["session_id"].values

                for mode in ISOLATION_MODES:
                    cols   = feat_cols_map[mode]
                    X_blend = blend_df[cols].values
                    y_pred  = trained_clfs[mode].predict(X_blend)

                    # Majority-vote per session, then count detections
                    session_preds: dict = defaultdict(list)
                    for sid, yp in zip(sids_blend, y_pred):
                        session_preds[sid].append(yp)

                    n_sessions   = len(session_preds)
                    n_detected   = sum(
                        1 for sid in session_preds
                        if Counter(session_preds[sid]).most_common(1)[0][0]
                        != normal_label_idx
                    )
                    detection_rate = n_detected / n_sessions if n_sessions > 0 else 0.0

                    all_results.append({
                        "attack_type":    attack_type,
                        "mode":           mode,
                        "blend_ratio":    blend_ratio,
                        "seed":           seed,
                        "detection_rate": detection_rate,
                        "n_sessions":     n_sessions,
                    })
                    print(f"    [{mode:>8}]  seed={seed}  "
                          f"detection_rate={detection_rate:.3f}  "
                          f"n_sessions={n_sessions}")

    results_df = pd.DataFrame(all_results)
    raw_csv = os.path.join(args.output_dir, "blend_attack_results.csv")
    results_df.to_csv(raw_csv, index=False)
    print(f"\nRaw results → {raw_csv}")

    agg = (
        results_df.groupby(["attack_type", "mode", "blend_ratio"])["detection_rate"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "detection_rate_mean", "std": "detection_rate_std"})
    )
    agg["detection_rate_std"] = agg["detection_rate_std"].fillna(0.0)
    sum_csv = os.path.join(args.output_dir, "blend_attack_summary.csv")
    agg.to_csv(sum_csv, index=False)
    print(f"Summary → {sum_csv}")

    # Console table
    print("\n" + "=" * 90)
    print(f"Detection rate mean — {args.raw_dir.split('/')[-1]}")
    for attack_type in attack_types:
        print(f"\n  attack_type={attack_type}")
        header = f"  {'mode':<10}" + "".join(
            f"  {br:.0%}".rjust(10) for br in BLEND_RATIOS
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for mode in ISOLATION_MODES:
            row_str = f"  {mode:<10}"
            for br in BLEND_RATIOS:
                row = agg[
                    (agg["attack_type"] == attack_type)
                    & (agg["mode"] == mode)
                    & (agg["blend_ratio"] == br)
                ]
                if len(row):
                    m = row["detection_rate_mean"].values[0]
                    s = row["detection_rate_std"].values[0]
                    row_str += f"  {m:.3f}±{s:.3f}"
                else:
                    row_str += "              —"
            print(row_str)
    print("=" * 90)


if __name__ == "__main__":
    main()
