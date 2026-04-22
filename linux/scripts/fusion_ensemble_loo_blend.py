#!/usr/bin/env python3
"""
Ensemble leave-one-out under blend attack.

Extends Section 12's any_attack ensemble: train 5 single-mode RFs, then compute
detection rate for any_attack *and* for each "any_attack minus mode X" variant.

If mode X is the only effective detector for a given (attack_type, blend_ratio),
removing X drops the ensemble detection rate to 0 — proving X is NECESSARY for
detecting that attack at that concentration.

Expected results:
  FreeCAD bulk_export @ 10% blend: only scalars detects (86.7%).
    - any_attack full:        86.7%
    - any_attack - scalars:    0.0%   ← proves scalars necessary
    - any_attack - {freq,seq,trans,ctx}: 86.7% (no change)

  KiCad project_scan @ 70% blend: only freq detects (100%).
    - any_attack full:        100%
    - any_attack - freq:        0%    ← proves freq necessary
    - any_attack - {scalars,seq,trans,ctx}: 100% (no change)

Strategies tested per (attack_type, blend_ratio, seed):
  ens_full         all 5 modes
  ens_minus_scalars  4 modes (no scalars)
  ens_minus_freq     4 modes
  ens_minus_seq      4 modes
  ens_minus_trans    4 modes
  ens_minus_ctx      4 modes

Output:
  ensemble_loo_blend_results.csv   — raw rows
  ensemble_loo_blend_summary.csv   — mean±std

Usage:
  python fusion_ensemble_loo_blend.py --raw-dir data/raw/freecad_only \\
      --output-dir models/fusion_advantage/ensemble_loo/freecad
  python fusion_ensemble_loo_blend.py --raw-dir data/raw/kicad_only \\
      --output-dir models/fusion_advantage/ensemble_loo/kicad
"""

import argparse
import glob
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ── constants ─────────────────────────────────────────────────────────────────

NOISE_SYSCALLS = {"mmap", "munmap", "mprotect"}
NORMAL_LABEL   = "normal"
CHUNK_SIZE     = 1000

BLEND_RATIOS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
N_SEEDS      = 5
BLEND_SEEDS  = list(range(N_SEEDS))
SINGLE_MODES = ["scalars", "freq", "seq", "trans", "ctx"]

LOO_VARIANTS = {
    "ens_full":             SINGLE_MODES,
    "ens_minus_scalars":    [m for m in SINGLE_MODES if m != "scalars"],
    "ens_minus_freq":       [m for m in SINGLE_MODES if m != "freq"],
    "ens_minus_seq":        [m for m in SINGLE_MODES if m != "seq"],
    "ens_minus_trans":      [m for m in SINGLE_MODES if m != "trans"],
    "ens_minus_ctx":        [m for m in SINGLE_MODES if m != "ctx"],
}

RF_PARAMS = dict(
    n_estimators=300, max_depth=9, class_weight="balanced",
    random_state=42, n_jobs=-1,
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
        df = pd.read_csv(p, usecols=["label", "session_id", "scenario"],
                         dtype=str, nrows=1)
        label    = str(df["label"].iloc[0])
        scenario = str(df["scenario"].iloc[0])
        subtype  = _scenario_to_subtype(scenario) if label != NORMAL_LABEL else NORMAL_LABEL
        rows.append({"session_id": str(df["session_id"].iloc[0]),
                     "label":      label, "subtype": subtype, "path": p})
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
    X = np.vstack(arrays)
    out = pd.DataFrame(X, columns=feat_names)
    out["session_id"] = session_ids
    out["subtype"]    = subtypes
    return out


def select_columns(df: pd.DataFrame, mode: str) -> list:
    all_cols = [c for c in df.columns if c not in {"session_id", "subtype"}]
    if mode == "scalars":
        return [c for c in ["syscall_entropy", "transition_entropy", "self_loop_ratio"]
                if c in all_cols]
    if mode == "freq":  return [c for c in all_cols if c.startswith("freq_count::")]
    if mode == "seq":   return [c for c in all_cols
                                 if c.startswith("seq2_norm::") or c.startswith("seq3_norm::")]
    if mode == "trans": return [c for c in all_cols if c.startswith("trans_norm::")]
    if mode == "ctx":   return [c for c in all_cols if c.startswith("ctx_path_type::")]
    raise ValueError(mode)


def session_majority_vote(y_pred: np.ndarray, sids: np.ndarray) -> dict:
    session_preds: dict = defaultdict(list)
    for sid, yp in zip(sids, y_pred):
        session_preds[sid].append(yp)
    return {sid: Counter(preds).most_common(1)[0][0]
            for sid, preds in session_preds.items()}


def detection_rate_loo(X_by_mode: dict, trained_clfs: dict,
                        sids: np.ndarray, normal_label_idx: int,
                        active_modes: list) -> float:
    """Detection rate via any_attack over `active_modes` subset."""
    session_flagged: dict = defaultdict(bool)
    for mode in active_modes:
        y_pred = trained_clfs[mode].predict(X_by_mode[mode])
        votes  = session_majority_vote(y_pred, sids)
        for sid, pred in votes.items():
            if pred != normal_label_idx:
                session_flagged[sid] = True
    all_sids = list(dict.fromkeys(sids))
    return (sum(1 for sid in all_sids if session_flagged.get(sid, False))
            / len(all_sids)) if all_sids else 0.0


def build_attack_syscall_pool(csv_paths: list, subtype_filter: str) -> list:
    pool: list = []
    for p in csv_paths:
        peek = pd.read_csv(p, usecols=["label", "scenario"], dtype=str, nrows=1)
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
    X = np.vstack(arrays)
    out = pd.DataFrame(X, columns=feat_names)
    out["session_id"] = session_ids
    out["subtype"]    = subtypes
    return out


def main():
    parser = argparse.ArgumentParser()
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
    print(f"  Classes: {class_names}")

    train_sids, test_sids = train_test_split(
        meta["session_id"], test_size=args.test_size,
        stratify=meta["subtype"], random_state=42,
    )
    train_meta = meta[meta["session_id"].isin(train_sids)].reset_index(drop=True)
    test_meta  = meta[meta["session_id"].isin(test_sids)].reset_index(drop=True)

    print("Building vocab …")
    vocab = build_vocab(csv_paths)

    print("Building train feature matrix …")
    train_df = build_feature_matrix(train_meta["path"].tolist(), vocab, subtype_map)
    y_train  = train_df["subtype"].map(label_enc).values

    print("Pre-training single-mode RFs …")
    trained_clfs: dict = {}
    for mode in SINGLE_MODES:
        cols = select_columns(train_df, mode)
        clf  = RandomForestClassifier(**RF_PARAMS)
        clf.fit(train_df[cols].values, y_train)
        trained_clfs[mode] = clf
        print(f"  [{mode}] {len(cols)} features")
    del train_df

    normal_label_idx = label_enc[NORMAL_LABEL]
    # all single-mode RFs must share the same class index order
    ref_classes = trained_clfs[SINGLE_MODES[0]].classes_
    for mode in SINGLE_MODES[1:]:
        assert np.array_equal(trained_clfs[mode].classes_, ref_classes)

    # Pre-load normal test raw sessions + build attack pools
    normal_test_meta = test_meta[test_meta["label"] == NORMAL_LABEL].reset_index(drop=True)
    normal_raw_dfs = [load_csv_filtered(p) for p in normal_test_meta["path"].tolist()]
    print(f"Normal test sessions: {len(normal_raw_dfs)}")

    attack_types = [c for c in class_names if c != NORMAL_LABEL]
    attack_pools = {at: build_attack_syscall_pool(csv_paths, at)
                    for at in attack_types}
    for at, pool in attack_pools.items():
        print(f"  attack pool [{at}] size: {len(pool)}")

    # Probe columns once by building a tiny feature matrix from one session
    sample_df = build_feature_matrix(
        [normal_test_meta["path"].iloc[0]], vocab, subtype_map
    )
    cols_map = {mode: select_columns(sample_df, mode) for mode in SINGLE_MODES}
    del sample_df

    results = []
    print("\n" + "=" * 60)
    print("Running ensemble LOO across blend attacks")
    print("=" * 60)

    for attack_type in attack_types:
        pool = attack_pools[attack_type]
        if not pool:
            continue
        print(f"\nattack_type={attack_type}")
        for blend_ratio in BLEND_RATIOS:
            seeds = [0] if blend_ratio == 0.0 else BLEND_SEEDS
            for seed in seeds:
                blend_df = build_blend_test_matrix(
                    normal_raw_dfs, pool, vocab, subtype_map, blend_ratio, seed
                )
                if blend_df.empty:
                    continue
                sids = blend_df["session_id"].values
                X_by_mode = {m: blend_df[cols_map[m]].values for m in SINGLE_MODES}

                row_vals = {}
                for variant_name, active_modes in LOO_VARIANTS.items():
                    dr = detection_rate_loo(
                        X_by_mode, trained_clfs, sids, normal_label_idx, active_modes,
                    )
                    row_vals[variant_name] = dr
                    results.append({
                        "attack_type":    attack_type,
                        "variant":        variant_name,
                        "blend_ratio":    blend_ratio,
                        "seed":           seed,
                        "detection_rate": dr,
                    })
                print(f"  br={blend_ratio:.2f} seed={seed}  "
                      f"full={row_vals['ens_full']:.3f}  "
                      f"-sc={row_vals['ens_minus_scalars']:.3f}  "
                      f"-fq={row_vals['ens_minus_freq']:.3f}  "
                      f"-sq={row_vals['ens_minus_seq']:.3f}  "
                      f"-tr={row_vals['ens_minus_trans']:.3f}  "
                      f"-cx={row_vals['ens_minus_ctx']:.3f}")

    df = pd.DataFrame(results)
    raw_csv = os.path.join(args.output_dir, "ensemble_loo_blend_results.csv")
    df.to_csv(raw_csv, index=False)

    agg = (df.groupby(["attack_type", "variant", "blend_ratio"])["detection_rate"]
             .agg(["mean", "std"]).reset_index()
             .rename(columns={"mean": "dr_mean", "std": "dr_std"}))
    agg["dr_std"] = agg["dr_std"].fillna(0.0)
    sum_csv = os.path.join(args.output_dir, "ensemble_loo_blend_summary.csv")
    agg.to_csv(sum_csv, index=False)
    print(f"\nRaw → {raw_csv}\nSummary → {sum_csv}")

    # Console summary
    print("\n" + "=" * 90)
    print("Ensemble LOO blend detection — drop when removing each mode")
    for at in attack_types:
        print(f"\n  attack_type={at}")
        for variant in LOO_VARIANTS:
            row_str = f"    {variant:<22}"
            for br in BLEND_RATIOS:
                row = agg[(agg["attack_type"] == at)
                          & (agg["variant"] == variant)
                          & (agg["blend_ratio"] == br)]
                if len(row):
                    row_str += f" {row['dr_mean'].values[0]:.3f}"
                else:
                    row_str += "   -  "
            print(row_str)


if __name__ == "__main__":
    main()
