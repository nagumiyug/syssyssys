#!/usr/bin/env python3
"""
Leave-one-component-out ablation for the fusion model.

Purpose: prove the NECESSITY of each feature group in the fusion architecture.
For each feature group G ∈ {freq, seq, trans, ctx, scalars}, train a fusion RF
with G removed, and evaluate under:

  - Standard 70/30 split (baseline; expected to be saturated)
  - Noise robustness (5%, 10%, 30% noise, 5 seeds each)
  - Cross-software transfer (train on one software, test on the other)

A feature group is "necessary" if removing it causes significant performance
degradation in AT LEAST ONE scenario. Context (ctx) is expected to be the
most clearly necessary component (noise-immune, cross-software transferable).

Configurations tested:
  fusion_full         all features
  fusion_minus_freq   fusion without freq_count::*
  fusion_minus_seq    fusion without seq2_norm::* / seq3_norm::*
  fusion_minus_trans  fusion without trans_norm::*
  fusion_minus_ctx    fusion without ctx_path_type::*
  fusion_minus_scalars fusion without syscall_entropy / transition_entropy / self_loop_ratio

Outputs (in --output-dir):
  leave_one_out_standard.csv       — (software, config, session_accuracy, macro_f1)
  leave_one_out_noise.csv          — (software, config, noise_level, seed, session_accuracy)
  leave_one_out_noise_summary.csv  — aggregated mean±std
  leave_one_out_cross.csv          — (scenario, config, session_accuracy, macro_f1)

Usage:
  python fusion_leave_one_out.py \\
      --freecad-dir data/raw/freecad_only \\
      --kicad-dir   data/raw/kicad_only \\
      --output-dir  models/fusion_advantage/leave_one_out
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ── constants ─────────────────────────────────────────────────────────────────

NOISE_SYSCALLS = {"mmap", "munmap", "mprotect"}
NORMAL_LABEL   = "normal"
CHUNK_SIZE     = 1000

NOISE_LEVELS   = [0.00, 0.05, 0.10, 0.30]
N_NOISE_SEEDS  = 5

# The configurations tested. "fusion_full" is the baseline; the rest remove one
# feature group from the fusion model.
CONFIGS = [
    "fusion_full",
    "fusion_minus_freq",
    "fusion_minus_seq",
    "fusion_minus_trans",
    "fusion_minus_ctx",
    "fusion_minus_scalars",
]

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


# ── noise injection ───────────────────────────────────────────────────────────

def inject_noise(df: pd.DataFrame, noise_level: float,
                 syscall_vocab: list, rng: np.random.RandomState) -> pd.DataFrame:
    if noise_level <= 0.0:
        return df
    df = df.copy()
    n = len(df)
    n_noisy = int(n * noise_level)
    if n_noisy == 0:
        return df
    indices      = rng.choice(n, size=n_noisy, replace=False)
    replacements = rng.choice(syscall_vocab, size=n_noisy)
    df.iloc[indices, df.columns.get_loc("syscall_name")] = replacements
    return df


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
                         subtype_map: dict,
                         noise_level: float = 0.0,
                         syscall_vocab: Optional[list] = None,
                         rng: Optional[np.random.RandomState] = None) -> pd.DataFrame:
    feat_names = _get_feature_names(vocab)
    feat_idx   = {name: i for i, name in enumerate(feat_names)}
    n_feats    = len(feat_names)
    arrays, session_ids, subtypes = [], [], []
    for p in csv_paths:
        df = load_csv_filtered(p)
        if df.empty:
            continue
        if noise_level > 0.0 and syscall_vocab is not None and rng is not None:
            df = inject_noise(df, noise_level, syscall_vocab, rng)
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


# ── column selection (leave-one-out aware) ────────────────────────────────────

SCALAR_COLS = {"syscall_entropy", "transition_entropy", "self_loop_ratio"}


def select_columns_for_config(df: pd.DataFrame, config: str) -> list:
    """Return feature columns for a given leave-one-out config."""
    meta = {"session_id", "subtype"}
    all_cols = [c for c in df.columns if c not in meta]

    def is_scalar(c):  return c in SCALAR_COLS
    def is_freq(c):    return c.startswith("freq_count::")
    def is_seq(c):     return c.startswith("seq2_norm::") or c.startswith("seq3_norm::")
    def is_trans(c):   return c.startswith("trans_norm::")
    def is_ctx(c):     return c.startswith("ctx_path_type::")

    if config == "fusion_full":
        return all_cols
    if config == "fusion_minus_scalars":
        return [c for c in all_cols if not is_scalar(c)]
    if config == "fusion_minus_freq":
        return [c for c in all_cols if not is_freq(c)]
    if config == "fusion_minus_seq":
        return [c for c in all_cols if not is_seq(c)]
    if config == "fusion_minus_trans":
        return [c for c in all_cols if not is_trans(c)]
    if config == "fusion_minus_ctx":
        return [c for c in all_cols if not is_ctx(c)]
    raise ValueError(f"Unknown config: {config}")


# ── session-level evaluation ──────────────────────────────────────────────────

def session_eval(y_true: np.ndarray, y_pred: np.ndarray,
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
    labels = list(range(n_classes))
    acc      = float(accuracy_score(yt_arr, yp_arr))
    macro_f1 = float(f1_score(yt_arr, yp_arr, average="macro",
                              labels=labels, zero_division=0))
    return {"session_accuracy": acc, "macro_f1": macro_f1}


# ── experiment runners ────────────────────────────────────────────────────────

def run_standard(software_name: str, csv_paths: list, vocab: dict,
                 subtype_map: dict, label_enc: dict, n_classes: int) -> list:
    meta = collect_session_meta(csv_paths)
    train_sids, test_sids = train_test_split(
        meta["session_id"],
        test_size=0.3,
        stratify=meta["subtype"],
        random_state=42,
    )
    train_meta = meta[meta["session_id"].isin(train_sids)].reset_index(drop=True)
    test_meta  = meta[meta["session_id"].isin(test_sids)].reset_index(drop=True)

    print(f"  [{software_name}] Building train feature matrix …")
    train_df = build_feature_matrix(
        train_meta["path"].tolist(), vocab, subtype_map
    )
    y_train  = train_df["subtype"].map(label_enc).values

    print(f"  [{software_name}] Building test feature matrix …")
    test_df  = build_feature_matrix(
        test_meta["path"].tolist(), vocab, subtype_map
    )
    y_test   = test_df["subtype"].map(label_enc).values
    sids_test = test_df["session_id"].values

    results = []
    for config in CONFIGS:
        cols = select_columns_for_config(train_df, config)
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(train_df[cols].values, y_train)
        y_pred = clf.predict(test_df[cols].values)
        m = session_eval(y_test, y_pred, sids_test, n_classes)
        results.append({
            "software":         software_name,
            "config":           config,
            "n_features":       len(cols),
            "session_accuracy": m["session_accuracy"],
            "macro_f1":         m["macro_f1"],
        })
        print(f"    [{config:>22}]  n_feat={len(cols):>4}  "
              f"acc={m['session_accuracy']:.3f}  f1={m['macro_f1']:.3f}")
    return results


def run_noise(software_name: str, csv_paths: list, vocab: dict,
              subtype_map: dict, label_enc: dict, n_classes: int) -> list:
    meta = collect_session_meta(csv_paths)
    train_sids, test_sids = train_test_split(
        meta["session_id"],
        test_size=0.3,
        stratify=meta["subtype"],
        random_state=42,
    )
    train_meta = meta[meta["session_id"].isin(train_sids)].reset_index(drop=True)
    test_meta  = meta[meta["session_id"].isin(test_sids)].reset_index(drop=True)

    # Train on clean data once
    print(f"  [{software_name}] Building clean train feature matrix …")
    train_df = build_feature_matrix(
        train_meta["path"].tolist(), vocab, subtype_map
    )
    y_train  = train_df["subtype"].map(label_enc).values

    # Pre-train one RF per config on clean train
    print(f"  [{software_name}] Pre-training one RF per config on clean data …")
    trained: dict = {}
    for config in CONFIGS:
        cols = select_columns_for_config(train_df, config)
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(train_df[cols].values, y_train)
        trained[config] = (clf, cols)
    del train_df

    syscall_vocab = vocab["syscalls"]

    results = []
    for noise_level in NOISE_LEVELS:
        seeds = [0] if noise_level == 0.0 else list(range(N_NOISE_SEEDS))
        for seed in seeds:
            rng = np.random.RandomState(seed)
            noise_label = f"noise={noise_level:.2f}  seed={seed}"
            print(f"  [{software_name}] Building noisy test {noise_label} …")
            test_df = build_feature_matrix(
                test_meta["path"].tolist(), vocab, subtype_map,
                noise_level=noise_level, syscall_vocab=syscall_vocab, rng=rng,
            )
            y_test   = test_df["subtype"].map(label_enc).values
            sids_test = test_df["session_id"].values
            for config in CONFIGS:
                clf, cols = trained[config]
                y_pred = clf.predict(test_df[cols].values)
                m = session_eval(y_test, y_pred, sids_test, n_classes)
                results.append({
                    "software":         software_name,
                    "config":           config,
                    "noise_level":      noise_level,
                    "seed":             seed,
                    "session_accuracy": m["session_accuracy"],
                    "macro_f1":         m["macro_f1"],
                })
                print(f"    [{config:>22}] acc={m['session_accuracy']:.3f}  "
                      f"f1={m['macro_f1']:.3f}")
            del test_df
    return results


def run_cross_software(csv_paths_a: list, csv_paths_b: list,
                       vocab: dict,
                       subtype_map_a: dict, subtype_map_b: dict,
                       label_enc: dict, n_classes: int,
                       name_a: str, name_b: str) -> list:
    # Build feature matrices for full A and full B
    print(f"  Building full feature matrix for {name_a} …")
    df_a = build_feature_matrix(csv_paths_a, vocab, subtype_map_a)
    print(f"  Building full feature matrix for {name_b} …")
    df_b = build_feature_matrix(csv_paths_b, vocab, subtype_map_b)

    y_a = df_a["subtype"].map(label_enc).values
    y_b = df_b["subtype"].map(label_enc).values
    sids_a = df_a["session_id"].values
    sids_b = df_b["session_id"].values

    results = []
    scenarios = [
        (f"{name_a}_to_{name_b}", df_a, y_a, df_b, y_b, sids_b),
        (f"{name_b}_to_{name_a}", df_b, y_b, df_a, y_a, sids_a),
    ]

    for scenario_label, train_df, y_tr, test_df, y_te, sids_te in scenarios:
        print(f"\n  Scenario: {scenario_label}")
        for config in CONFIGS:
            cols_tr = select_columns_for_config(train_df, config)
            cols_te = select_columns_for_config(test_df,  config)
            clf = RandomForestClassifier(**RF_PARAMS)
            clf.fit(train_df[cols_tr].values, y_tr)
            y_pred = clf.predict(test_df[cols_te].values)
            m = session_eval(y_te, y_pred, sids_te, n_classes)
            results.append({
                "scenario":         scenario_label,
                "config":           config,
                "n_features":       len(cols_tr),
                "session_accuracy": m["session_accuracy"],
                "macro_f1":         m["macro_f1"],
            })
            print(f"    [{config:>22}]  acc={m['session_accuracy']:.3f}  "
                  f"f1={m['macro_f1']:.3f}")
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Leave-one-component-out ablation for fusion model"
    )
    parser.add_argument("--freecad-dir", required=True)
    parser.add_argument("--kicad-dir",   required=True)
    parser.add_argument("--output-dir",  required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_freecad = list_csvs(args.freecad_dir)
    csv_kicad   = list_csvs(args.kicad_dir)
    print(f"Found {len(csv_freecad)} FreeCAD + {len(csv_kicad)} KiCad CSV files")

    # ── vocabularies ──────────────────────────────────────────────────────────
    print("\nBuilding per-software and shared vocabularies …")
    vocab_freecad = build_vocab(csv_freecad)
    vocab_kicad   = build_vocab(csv_kicad)
    vocab_shared  = build_vocab(csv_freecad + csv_kicad)
    print(f"  FreeCAD:  {len(vocab_freecad['syscalls'])} syscalls, "
          f"{len(vocab_freecad['bigrams'])} bigrams, "
          f"{len(vocab_freecad['trigrams'])} trigrams, "
          f"{len(vocab_freecad['path_types'])} path_types")
    print(f"  KiCad:    {len(vocab_kicad['syscalls'])} syscalls, "
          f"{len(vocab_kicad['bigrams'])} bigrams, "
          f"{len(vocab_kicad['trigrams'])} trigrams, "
          f"{len(vocab_kicad['path_types'])} path_types")
    print(f"  Shared:   {len(vocab_shared['syscalls'])} syscalls, "
          f"{len(vocab_shared['bigrams'])} bigrams, "
          f"{len(vocab_shared['trigrams'])} trigrams, "
          f"{len(vocab_shared['path_types'])} path_types")

    # ── session meta and label encoding ───────────────────────────────────────
    meta_freecad = collect_session_meta(csv_freecad)
    meta_kicad   = collect_session_meta(csv_kicad)
    subtype_map_freecad = dict(zip(meta_freecad["session_id"], meta_freecad["subtype"]))
    subtype_map_kicad   = dict(zip(meta_kicad["session_id"],   meta_kicad["subtype"]))

    class_names = sorted(
        set(meta_freecad["subtype"].unique())
        | set(meta_kicad["subtype"].unique())
    )
    label_enc   = {c: i for i, c in enumerate(class_names)}
    n_classes   = len(class_names)
    print(f"  Classes ({n_classes}): {class_names}")

    # ── Experiment 1: standard 70/30 per software ────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 1: Standard 70/30")
    print("=" * 60)
    std_results = []
    std_results += run_standard(
        "freecad", csv_freecad, vocab_freecad, subtype_map_freecad,
        label_enc, n_classes,
    )
    std_results += run_standard(
        "kicad",   csv_kicad,   vocab_kicad,   subtype_map_kicad,
        label_enc, n_classes,
    )
    std_df = pd.DataFrame(std_results)
    std_csv = os.path.join(args.output_dir, "leave_one_out_standard.csv")
    std_df.to_csv(std_csv, index=False)
    print(f"\nStandard results → {std_csv}")

    # ── Experiment 2: noise robustness per software ──────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 2: Noise robustness")
    print("=" * 60)
    noise_results = []
    noise_results += run_noise(
        "freecad", csv_freecad, vocab_freecad, subtype_map_freecad,
        label_enc, n_classes,
    )
    noise_results += run_noise(
        "kicad",   csv_kicad,   vocab_kicad,   subtype_map_kicad,
        label_enc, n_classes,
    )
    noise_df = pd.DataFrame(noise_results)
    noise_csv = os.path.join(args.output_dir, "leave_one_out_noise.csv")
    noise_df.to_csv(noise_csv, index=False)

    agg = (
        noise_df.groupby(["software", "config", "noise_level"])
        ["session_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "acc_mean", "std": "acc_std"})
    )
    agg["acc_std"] = agg["acc_std"].fillna(0.0)
    summary_csv = os.path.join(args.output_dir, "leave_one_out_noise_summary.csv")
    agg.to_csv(summary_csv, index=False)
    print(f"\nNoise raw     → {noise_csv}")
    print(f"Noise summary → {summary_csv}")

    # ── Experiment 3: cross-software transfer ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Experiment 3: Cross-software transfer")
    print("=" * 60)
    cross_results = run_cross_software(
        csv_freecad, csv_kicad, vocab_shared,
        subtype_map_freecad, subtype_map_kicad,
        label_enc, n_classes,
        name_a="freecad", name_b="kicad",
    )
    cross_df = pd.DataFrame(cross_results)
    cross_csv = os.path.join(args.output_dir, "leave_one_out_cross.csv")
    cross_df.to_csv(cross_csv, index=False)
    print(f"\nCross-software results → {cross_csv}")

    # ── Console summary tables ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY: Standard 70/30 accuracy")
    pivot_std = std_df.pivot(index="config", columns="software",
                              values="session_accuracy")
    pivot_std = pivot_std.reindex(CONFIGS)
    print(pivot_std.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 90)
    print("SUMMARY: Noise robustness (session accuracy, mean over seeds)")
    for sw in ["freecad", "kicad"]:
        sub = agg[agg["software"] == sw]
        pivot = sub.pivot(index="config", columns="noise_level", values="acc_mean")
        pivot = pivot.reindex(CONFIGS)
        print(f"\n{sw}:")
        print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 90)
    print("SUMMARY: Cross-software transfer")
    pivot_cross = cross_df.pivot(index="config", columns="scenario",
                                  values="session_accuracy")
    pivot_cross = pivot_cross.reindex(CONFIGS)
    print(pivot_cross.to_string(float_format=lambda x: f"{x:.3f}"))
    print("=" * 90)


if __name__ == "__main__":
    main()
