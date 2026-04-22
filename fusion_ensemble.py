#!/usr/bin/env python3
"""
Ensemble fusion study.

Addresses the core weakness of feature-concatenation fusion: in a 436-dim RF,
the 3 entropy (scalars) features have very low selection probability, so their
anomaly signal is suppressed.  Ensemble fusion trains one RF per feature mode
and combines predictions — each mode's detector has full voice.

Ensemble variants:
  soft_vote   — average predicted class probabilities across 5 mode-RFs → argmax
  hard_vote   — plurality vote of the 5 mode-RF class predictions per chunk,
                then session-level majority vote
  any_attack  — flag a session as non-normal if ANY mode-RF's session-level
                majority vote predicts a non-normal class

Reference strategies also included in output:
  scalars / freq / seq / trans / ctx  — single-mode baselines
  concat                              — feature-concatenation (existing approach)

Experiments:
  A. Standard 70/30 accuracy on clean test data
     (all single modes + concat + soft_vote + hard_vote)
  B. Blend attack detection rate across blend_ratio 0–100%
     (concat + soft_vote + hard_vote + any_attack + scalars as reference)

Output (in --output-dir):
  ensemble_accuracy.csv          — strategy → session accuracy on clean test
  ensemble_blend_results.csv     — raw (attack_type, strategy, blend_ratio, seed)
  ensemble_blend_summary.csv     — aggregated mean±std

Usage:
  python fusion_ensemble.py --raw-dir /root/s410/data/raw/freecad_only \\
      --output-dir /root/s410/models/fusion_advantage/ensemble/freecad
  python fusion_ensemble.py --raw-dir /root/s410/data/raw/kicad_only \\
      --output-dir /root/s410/models/fusion_advantage/ensemble/kicad
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ── constants ─────────────────────────────────────────────────────────────────

NOISE_SYSCALLS  = {"mmap", "munmap", "mprotect"}
NORMAL_LABEL    = "normal"
CHUNK_SIZE      = 1000

BLEND_RATIOS    = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
N_SEEDS         = 5
BLEND_SEEDS     = list(range(N_SEEDS))
SINGLE_MODES    = ["scalars", "freq", "seq", "trans", "ctx"]

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


# ── session-level evaluation ──────────────────────────────────────────────────

def session_majority_vote(y_pred: np.ndarray, sids: np.ndarray) -> dict:
    """Return {sid: session_prediction} via chunk majority vote."""
    session_preds: dict = defaultdict(list)
    for sid, yp in zip(sids, y_pred):
        session_preds[sid].append(yp)
    return {
        sid: Counter(preds).most_common(1)[0][0]
        for sid, preds in session_preds.items()
    }


def session_accuracy(y_pred: np.ndarray, y_true_map: dict,
                     sids: np.ndarray) -> float:
    votes = session_majority_vote(y_pred, sids)
    yt, yp = [], []
    for sid, pred in votes.items():
        if sid in y_true_map:
            yt.append(y_true_map[sid])
            yp.append(pred)
    return float(accuracy_score(yt, yp)) if yt else 0.0


# ── ensemble prediction helpers ───────────────────────────────────────────────

def get_mode_features(full_df: pd.DataFrame, feat_cols_map: dict) -> dict:
    """Extract per-mode feature arrays from the full feature DataFrame."""
    return {mode: full_df[cols].values for mode, cols in feat_cols_map.items()}


def predict_soft_vote(X_by_mode: dict, trained_clfs: dict) -> np.ndarray:
    """Average predicted class probabilities across all mode-RFs → argmax."""
    proba_sum = None
    for mode in SINGLE_MODES:
        proba = trained_clfs[mode].predict_proba(X_by_mode[mode])
        if proba_sum is None:
            proba_sum = proba.copy()
        else:
            proba_sum += proba
    return np.argmax(proba_sum, axis=1)


def predict_hard_vote(X_by_mode: dict, trained_clfs: dict) -> np.ndarray:
    """Plurality vote of 5 mode-RF predictions per chunk."""
    all_preds = np.stack(
        [trained_clfs[mode].predict(X_by_mode[mode]) for mode in SINGLE_MODES],
        axis=1,
    )  # shape: (n_chunks, 5)
    result = np.empty(len(all_preds), dtype=int)
    for i, row in enumerate(all_preds):
        result[i] = Counter(row.tolist()).most_common(1)[0][0]
    return result


def detection_rate_any_attack(X_by_mode: dict, trained_clfs: dict,
                               sids: np.ndarray,
                               normal_label_idx: int) -> float:
    """
    Flag session as anomalous if ANY mode-RF's session-level majority vote
    predicts non-normal.  Returns fraction of sessions flagged.
    """
    # Per-mode session votes
    session_flagged: dict = defaultdict(bool)
    for mode in SINGLE_MODES:
        y_pred = trained_clfs[mode].predict(X_by_mode[mode])
        votes  = session_majority_vote(y_pred, sids)
        for sid, pred in votes.items():
            if pred != normal_label_idx:
                session_flagged[sid] = True
    # Count sessions present in sids
    all_sids = list(dict.fromkeys(sids))  # unique, insertion-ordered
    n_sessions = len(all_sids)
    n_detected = sum(1 for sid in all_sids if session_flagged.get(sid, False))
    return n_detected / n_sessions if n_sessions > 0 else 0.0


# ── blend helpers (identical to fusion_blend_attack.py) ───────────────────────

def build_attack_syscall_pool(csv_paths: list, subtype_filter: str) -> list:
    pool: list = []
    for p in csv_paths:
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
    parser = argparse.ArgumentParser(description="Ensemble fusion study")
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

    print("Building full feature matrix (train) …")
    train_df = build_feature_matrix(train_meta["path"].tolist(), vocab, subtype_map)
    y_train  = train_df["subtype"].map(label_enc).values

    print("Building full feature matrix (test) …")
    test_df  = build_feature_matrix(test_meta["path"].tolist(), vocab, subtype_map)
    y_true_map = {
        sid: label_enc[sub]
        for sid, sub in zip(test_df["session_id"], test_df["subtype"])
    }
    sids_test = test_df["session_id"].values

    # Column maps
    feat_cols_map: dict = {
        mode: select_columns(train_df, mode) for mode in SINGLE_MODES
    }
    feat_cols_map["concat"] = select_columns(train_df, "fusion")

    print("\nPre-training classifiers …")
    trained_clfs: dict = {}
    for key, cols in feat_cols_map.items():
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(train_df[cols].values, y_train)
        trained_clfs[key] = clf
        print(f"  [{key}] {len(cols)} features")
    del train_df

    # Verify all single-mode RFs have same class ordering
    ref_classes = trained_clfs[SINGLE_MODES[0]].classes_
    for mode in SINGLE_MODES[1:]:
        assert np.array_equal(trained_clfs[mode].classes_, ref_classes), \
            f"Class order mismatch for mode={mode}"
    normal_label_idx = label_enc[NORMAL_LABEL]

    # ── Experiment A: standard 70/30 accuracy ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment A: standard 70/30 accuracy")
    print("=" * 60)

    X_by_mode_test = get_mode_features(test_df, feat_cols_map)
    acc_results = []

    # Single modes + concat
    for key in list(SINGLE_MODES) + ["concat"]:
        y_pred = trained_clfs[key].predict(X_by_mode_test[key])
        acc    = session_accuracy(y_pred, y_true_map, sids_test)
        print(f"  [{key:>10}]  acc={acc:.4f}")
        acc_results.append({"strategy": key, "session_accuracy": acc})

    # Ensemble soft_vote
    y_pred_soft = predict_soft_vote(X_by_mode_test, trained_clfs)
    acc_soft    = session_accuracy(y_pred_soft, y_true_map, sids_test)
    print(f"  [{'soft_vote':>10}]  acc={acc_soft:.4f}")
    acc_results.append({"strategy": "soft_vote", "session_accuracy": acc_soft})

    # Ensemble hard_vote
    y_pred_hard = predict_hard_vote(X_by_mode_test, trained_clfs)
    acc_hard    = session_accuracy(y_pred_hard, y_true_map, sids_test)
    print(f"  [{'hard_vote':>10}]  acc={acc_hard:.4f}")
    acc_results.append({"strategy": "hard_vote", "session_accuracy": acc_hard})

    # any_attack: binary anomaly detector — FP rate on NORMAL test sessions only.
    # Using full test set would conflate true positives with false positives.
    normal_test_sids = set(
        test_meta[test_meta["label"] == NORMAL_LABEL]["session_id"].tolist()
    )
    normal_mask = np.isin(sids_test, list(normal_test_sids))
    X_normal_modes = {mode: X[normal_mask] for mode, X in X_by_mode_test.items()}
    sids_normal    = sids_test[normal_mask]
    fp_any = detection_rate_any_attack(
        X_normal_modes, trained_clfs, sids_normal, normal_label_idx
    )
    print(f"  [{'any_attack':>10}]  false_positive_rate={fp_any:.4f}  (normal sessions only, 1-fp={1-fp_any:.4f})")
    acc_results.append({"strategy": "any_attack", "session_accuracy": 1.0 - fp_any})

    acc_df = pd.DataFrame(acc_results)
    acc_csv = os.path.join(args.output_dir, "ensemble_accuracy.csv")
    acc_df.to_csv(acc_csv, index=False)
    print(f"\nAccuracy results → {acc_csv}")

    del test_df  # free memory

    # ── Experiment B: blend attack detection ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment B: blend attack detection")
    print("=" * 60)

    # Pre-load normal test raw DataFrames
    normal_test_meta = test_meta[test_meta["label"] == NORMAL_LABEL].reset_index(drop=True)
    normal_raw_dfs: list = []
    for p in normal_test_meta["path"].tolist():
        normal_raw_dfs.append(load_csv_filtered(p))
    print(f"  normal test sessions pre-loaded: {len(normal_raw_dfs)}")

    # Build attack pools
    attack_types = [c for c in class_names if c != NORMAL_LABEL]
    print(f"  attack types: {attack_types}")
    attack_pools: dict = {}
    for at in attack_types:
        pool = build_attack_syscall_pool(csv_paths, at)
        attack_pools[at] = pool
        print(f"  [{at}] pool size: {len(pool)}")

    blend_results = []

    for attack_type in attack_types:
        pool = attack_pools[attack_type]
        if not pool:
            continue

        print(f"\nattack_type={attack_type}")

        for blend_ratio in BLEND_RATIOS:
            seeds_to_run = [0] if blend_ratio == 0.0 else BLEND_SEEDS
            print(f"  blend_ratio={blend_ratio:.2f}  seeds={seeds_to_run}")

            for seed in seeds_to_run:
                blend_df = build_blend_test_matrix(
                    normal_raw_dfs, pool, vocab, subtype_map, blend_ratio, seed
                )
                if blend_df.empty:
                    continue

                sids_blend    = blend_df["session_id"].values
                X_blend_modes = get_mode_features(blend_df, feat_cols_map)
                n_sessions    = len(set(sids_blend))

                def _dr(y_pred):
                    votes = session_majority_vote(y_pred, sids_blend)
                    n_det = sum(1 for v in votes.values() if v != normal_label_idx)
                    return n_det / len(votes) if votes else 0.0

                # scalars (reference single mode)
                dr_scalars = _dr(trained_clfs["scalars"].predict(X_blend_modes["scalars"]))
                # concat fusion
                dr_concat  = _dr(trained_clfs["concat"].predict(X_blend_modes["concat"]))
                # soft_vote ensemble
                dr_soft    = _dr(predict_soft_vote(X_blend_modes, trained_clfs))
                # hard_vote ensemble
                dr_hard    = _dr(predict_hard_vote(X_blend_modes, trained_clfs))
                # any_attack ensemble
                dr_any     = detection_rate_any_attack(
                    X_blend_modes, trained_clfs, sids_blend, normal_label_idx
                )

                for strategy, dr in [
                    ("scalars", dr_scalars),
                    ("concat",  dr_concat),
                    ("soft_vote", dr_soft),
                    ("hard_vote", dr_hard),
                    ("any_attack", dr_any),
                ]:
                    blend_results.append({
                        "attack_type":    attack_type,
                        "strategy":       strategy,
                        "blend_ratio":    blend_ratio,
                        "seed":           seed,
                        "detection_rate": dr,
                        "n_sessions":     n_sessions,
                    })

                print(f"    scalars={dr_scalars:.3f}  concat={dr_concat:.3f}  "
                      f"soft={dr_soft:.3f}  hard={dr_hard:.3f}  any={dr_any:.3f}")

    blend_df_out = pd.DataFrame(blend_results)
    raw_csv = os.path.join(args.output_dir, "ensemble_blend_results.csv")
    blend_df_out.to_csv(raw_csv, index=False)
    print(f"\nRaw blend results → {raw_csv}")

    agg = (
        blend_df_out.groupby(["attack_type", "strategy", "blend_ratio"])["detection_rate"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "detection_rate_mean", "std": "detection_rate_std"})
    )
    agg["detection_rate_std"] = agg["detection_rate_std"].fillna(0.0)
    sum_csv = os.path.join(args.output_dir, "ensemble_blend_summary.csv")
    agg.to_csv(sum_csv, index=False)
    print(f"Blend summary → {sum_csv}")

    # Console summary table
    print("\n" + "=" * 90)
    print("Detection rate mean (blend attack) — " + args.raw_dir.split("/")[-1])
    strategies_order = ["scalars", "concat", "soft_vote", "hard_vote", "any_attack"]
    for attack_type in attack_types:
        print(f"\n  attack_type={attack_type}")
        header = f"  {'strategy':<12}" + "".join(
            f"{br:.0%}".rjust(9) for br in BLEND_RATIOS
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for strat in strategies_order:
            row_str = f"  {strat:<12}"
            for br in BLEND_RATIOS:
                row = agg[
                    (agg["attack_type"] == attack_type)
                    & (agg["strategy"] == strat)
                    & (agg["blend_ratio"] == br)
                ]
                if len(row):
                    m = row["detection_rate_mean"].values[0]
                    row_str += f"  {m:.3f}  "
                else:
                    row_str += "    —    "
            print(row_str)
    print("=" * 90)


if __name__ == "__main__":
    main()
