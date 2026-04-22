#!/usr/bin/env python3
"""
Multi-class data efficiency study.

Tests how 4-class accuracy (normal/bulk_export/source_copy/project_scan)
degrades as training data shrinks. Binary session-level accuracy saturates
to 100% for all modes, making training fraction irrelevant there.
4-class type identification is harder: scalars (3 features) is expected
to need more data to learn all 4 class boundaries, while fusion (all
features) should maintain high type-ID accuracy with less data.

Grid:
  train_fraction ∈ {0.10, 0.20, 0.30, 0.50, 0.70, 1.00}
  n_seeds = 5  (random subsampling; fraction=1.0 → 1 seed)
  mode ∈ {scalars, freq, seq, trans, ctx, fusion}

Full feature matrix built ONCE; subsampling via session-ID filtering.

Output:
  multiclass_efficiency_results.csv   — raw (mode, fraction, seed) rows
  multiclass_efficiency_summary.csv   — aggregated mean±std per (mode, fraction)

Usage:
  python fusion_multiclass_efficiency.py --raw-dir data/raw/freecad_only \
      --output-dir models/fusion_advantage/multiclass_efficiency/freecad
  python fusion_multiclass_efficiency.py --raw-dir data/raw/kicad_only \
      --output-dir models/fusion_advantage/multiclass_efficiency/kicad
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
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ── constants ─────────────────────────────────────────────────────────────────

NOISE_SYSCALLS  = {"mmap", "munmap", "mprotect"}
NORMAL_LABEL    = "normal"
CHUNK_SIZE      = 1000

TRAIN_FRACTIONS = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
N_SEEDS         = 5
SUBSAMPLE_SEEDS = list(range(N_SEEDS))
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
        label   = str(df["label"].iloc[0])
        scenario = str(df["scenario"].iloc[0])
        subtype  = _scenario_to_subtype(scenario) if label != NORMAL_LABEL else NORMAL_LABEL
        rows.append({"session_id": str(df["session_id"].iloc[0]),
                     "label":      label,
                     "subtype":    subtype,
                     "path":       p})
    return pd.DataFrame(rows)


# ── vocabulary ────────────────────────────────────────────────────────────────

def build_vocab(csv_paths: list) -> dict:
    syscalls: set = set()
    bigrams:  set = set()
    trigrams: set = set()
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
    feat_names = _get_feature_names(vocab)
    feat_idx   = {name: i for i, name in enumerate(feat_names)}
    n_feats    = len(feat_names)
    arrays: list = []
    session_ids: list = []
    subtypes: list = []
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
    session_true: dict = {}
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
    labels  = list(range(n_classes))
    acc     = float(accuracy_score(yt_arr, yp_arr))
    macro_f1 = float(f1_score(yt_arr, yp_arr, average="macro",
                               labels=labels, zero_division=0))
    per_cls  = recall_score(yt_arr, yp_arr, average=None,
                            labels=labels, zero_division=0)
    return {"session_accuracy": acc, "macro_f1": macro_f1,
            "per_class_recall": per_cls.tolist()}


# ── session subsampling ───────────────────────────────────────────────────────

def subsample_sessions(full_train_df: pd.DataFrame, class_names: list,
                       fraction: float, seed: int) -> pd.DataFrame:
    """Stratified subsample at session level (by subtype), min 1 per class."""
    info = (full_train_df[["session_id", "subtype"]]
            .drop_duplicates("session_id").reset_index(drop=True))
    rng  = np.random.RandomState(seed)
    kept: list = []
    for cls in class_names:
        sids = info.loc[info["subtype"] == cls, "session_id"].tolist()
        rng.shuffle(sids)
        n_keep = max(1, int(len(sids) * fraction))
        kept.extend(sids[:n_keep])
    return full_train_df[full_train_df["session_id"].isin(kept)].reset_index(drop=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-class data efficiency study")
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

    print("Building full training matrix (built once) …")
    full_train_df = build_feature_matrix(train_meta["path"].tolist(), vocab, subtype_map)
    print(f"  full_train_chunks={len(full_train_df)}")

    print("Building test matrix …")
    test_df = build_feature_matrix(test_meta["path"].tolist(), vocab, subtype_map)
    y_test     = test_df["subtype"].map(label_enc).values
    sids_test  = test_df["session_id"].values
    print(f"  test_chunks={len(test_df)}")

    all_results = []

    for mode in ISOLATION_MODES:
        feat_cols = select_columns(full_train_df, mode)
        X_test    = test_df[feat_cols].values
        print(f"\n[{mode}] {len(feat_cols)} features")

        for fraction in TRAIN_FRACTIONS:
            seeds_to_run = [0] if fraction >= 1.0 else SUBSAMPLE_SEEDS
            accs, f1s = [], []
            per_cls_all = []

            for seed in seeds_to_run:
                sub_df  = full_train_df if fraction >= 1.0 else \
                          subsample_sessions(full_train_df, class_names, fraction, seed)
                X_train = sub_df[feat_cols].values
                y_train = sub_df["subtype"].map(label_enc).values

                clf = RandomForestClassifier(**RF_PARAMS)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                m = session_eval_multiclass(y_test, y_pred, sids_test, n_classes)
                accs.append(m["session_accuracy"])
                f1s.append(m["macro_f1"])
                per_cls_all.append(m["per_class_recall"])

                all_results.append({
                    "mode": mode, "train_fraction": fraction, "seed": seed,
                    "n_train_sessions": sub_df["session_id"].nunique(),
                    "session_accuracy": m["session_accuracy"],
                    "macro_f1":         m["macro_f1"],
                    **{f"recall_{class_names[i]}": m["per_class_recall"][i]
                       for i in range(n_classes)},
                })

            mean_acc = float(np.mean(accs))
            std_acc  = float(np.std(accs)) if len(accs) > 1 else 0.0
            mean_f1  = float(np.mean(f1s))
            print(f"  frac={fraction:.2f}  acc={mean_acc:.3f}±{std_acc:.3f}  "
                  f"f1={mean_f1:.3f}")

    results_df = pd.DataFrame(all_results)
    raw_csv = os.path.join(args.output_dir, "multiclass_efficiency_results.csv")
    results_df.to_csv(raw_csv, index=False)
    print(f"\nRaw results → {raw_csv}")

    agg_acc = (results_df.groupby(["mode", "train_fraction"])["session_accuracy"]
               .agg(["mean", "std"]).reset_index()
               .rename(columns={"mean": "acc_mean", "std": "acc_std"}))
    agg_f1  = (results_df.groupby(["mode", "train_fraction"])["macro_f1"]
               .agg(["mean", "std"]).reset_index()
               .rename(columns={"mean": "f1_mean", "std": "f1_std"}))
    summary = agg_acc.merge(agg_f1, on=["mode", "train_fraction"])
    summary["acc_std"] = summary["acc_std"].fillna(0.0)
    summary["frac_std"]= summary["f1_std"].fillna(0.0)
    sum_csv = os.path.join(args.output_dir, "multiclass_efficiency_summary.csv")
    summary.to_csv(sum_csv, index=False)
    print(f"Summary → {sum_csv}")

    # Console table
    print("\n" + "=" * 80)
    print(f"4-class session accuracy mean (±std) — {args.raw_dir.split('/')[-1]}")
    header = f"{'mode':<10}" + "".join(f"  {f:.0%}".rjust(12) for f in TRAIN_FRACTIONS)
    print(header)
    print("-" * 80)
    for mode in ISOLATION_MODES:
        row_str = f"{mode:<10}"
        for frac in TRAIN_FRACTIONS:
            row = summary[(summary["mode"] == mode) &
                          (summary["train_fraction"] == frac)]
            if len(row):
                m, s = row["acc_mean"].values[0], row["acc_std"].values[0]
                row_str += f"  {m:.3f}±{s:.3f}"
            else:
                row_str += "              —"
        print(row_str)
    print("=" * 80)


if __name__ == "__main__":
    main()
