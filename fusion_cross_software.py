#!/usr/bin/env python3
"""
Cross-software generalization experiment.

Builds a SHARED vocabulary from two software datasets (e.g. FreeCAD and KiCad),
then evaluates how well each feature mode transfers from one software environment
to the other.

Key insight:
  - seq / freq / trans depend on specific syscall names, which differ between
    software environments → these modes may NOT transfer well.
  - scalars (entropy, self-loop ratio) and ctx (path_type ratios) are
    ratio-based / path-type-based → should transfer across environments.
  - fusion combines both groups → expected graceful degradation.

Shared vocab approach:
  The vocabulary (syscalls, bigrams, trigrams, path_types) is built from ALL
  CSV files in BOTH --raw-dir-a and --raw-dir-b.  This ensures that a syscall
  seen only in software B still has a valid column when classifying software A
  sessions (its freq/seq/trans features will be 0 for A sessions but present
  in the feature matrix).  Without the shared vocab, the feature matrices for A
  and B would have different column sets, making cross-software prediction
  impossible.

Inputs:
  --raw-dir-a   directory of CSV files for software A (e.g. FreeCAD)
  --raw-dir-b   directory of CSV files for software B (e.g. KiCad)
  --output-dir  where to write results
  --name-a      short label for software A (default "software_a")
  --name-b      short label for software B (default "software_b")

Outputs:
  cross_software_results.csv   — raw rows: scenario, mode, session_accuracy, macro_f1
  Console table                — accuracy per mode × scenario (4 columns)
  Degradation report           — (within_A − A_to_B) and (within_B − B_to_A) per mode

Scenarios:
  within_A   train 56 / test 24 sessions of A (70/30 stratified split)
  within_B   train 56 / test 24 sessions of B (70/30 stratified split)
  A_to_B     train on all 80 A sessions, test on all 80 B sessions
  B_to_A     train on all 80 B sessions, test on all 80 A sessions

Usage:
  python fusion_cross_software.py \\
      --raw-dir-a data/raw/freecad_only \\
      --raw-dir-b data/raw/kicad_only \\
      --output-dir models/fusion_advantage/cross_software \\
      --name-a freecad --name-b kicad
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


def build_shared_vocab(csv_paths_a: list, csv_paths_b: list) -> dict:
    """Build vocabulary from both source and target software datasets."""
    return build_vocab(csv_paths_a + csv_paths_b)


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


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_scenario(scenario_label: str,
                  train_df: pd.DataFrame,
                  test_df: pd.DataFrame,
                  label_enc: dict,
                  n_classes: int) -> list:
    """Train+eval all modes for one scenario; return list of result dicts."""
    results = []
    y_train  = train_df["subtype"].map(label_enc).values
    y_test   = test_df["subtype"].map(label_enc).values
    sids_test = test_df["session_id"].values
    for mode in ISOLATION_MODES:
        cols_train = select_columns(train_df, mode)
        cols_test  = select_columns(test_df, mode)
        # cols must match; they do because both DFs share the same shared vocab
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(train_df[cols_train].values, y_train)
        y_pred = clf.predict(test_df[cols_test].values)
        m = session_eval_multiclass(y_test, y_pred, sids_test, n_classes)
        results.append({
            "scenario":         scenario_label,
            "mode":             mode,
            "session_accuracy": m["session_accuracy"],
            "macro_f1":         m["macro_f1"],
        })
        print(f"  [{scenario_label:>8}][{mode:>8}]  "
              f"acc={m['session_accuracy']:.3f}  f1={m['macro_f1']:.3f}")
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-software generalization experiment"
    )
    parser.add_argument("--raw-dir-a",   required=True,
                        help="Source software raw CSV directory (e.g. freecad_only)")
    parser.add_argument("--raw-dir-b",   required=True,
                        help="Target software raw CSV directory (e.g. kicad_only)")
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--name-a",      default="software_a",
                        help="Short label for software A")
    parser.add_argument("--name-b",      default="software_b",
                        help="Short label for software B")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load paths and meta ───────────────────────────────────────────────────
    csv_paths_a = list_csvs(args.raw_dir_a)
    csv_paths_b = list_csvs(args.raw_dir_b)
    print(f"Found {len(csv_paths_a)} CSV files for {args.name_a}")
    print(f"Found {len(csv_paths_b)} CSV files for {args.name_b}")

    meta_a = collect_session_meta(csv_paths_a)
    meta_b = collect_session_meta(csv_paths_b)
    subtype_map_a = dict(zip(meta_a["session_id"], meta_a["subtype"]))
    subtype_map_b = dict(zip(meta_b["session_id"], meta_b["subtype"]))

    class_names_a = sorted(meta_a["subtype"].unique().tolist())
    class_names_b = sorted(meta_b["subtype"].unique().tolist())
    # Use union of class names so label encodings are consistent
    all_class_names = sorted(set(class_names_a) | set(class_names_b))
    label_enc  = {c: i for i, c in enumerate(all_class_names)}
    n_classes  = len(all_class_names)

    print(f"\n{args.name_a} classes ({len(class_names_a)}): {class_names_a}")
    print(f"{args.name_b} classes ({len(class_names_b)}): {class_names_b}")
    print(f"Union classes ({n_classes}): {all_class_names}")
    if set(class_names_a) != set(class_names_b):
        only_a = sorted(set(class_names_a) - set(class_names_b))
        only_b = sorted(set(class_names_b) - set(class_names_a))
        if only_a:
            print(f"  [WARNING] Classes only in {args.name_a}: {only_a} "
                  f"— RF trained on {args.name_a} cannot predict these when testing on {args.name_b}")
        if only_b:
            print(f"  [WARNING] Classes only in {args.name_b}: {only_b} "
                  f"— RF trained on {args.name_b} cannot predict these when testing on {args.name_a}")
        print(f"  macro_f1 is averaged over all {n_classes} union classes; "
              f"phantom classes score 0 recall.")

    # ── build shared vocabulary ───────────────────────────────────────────────
    print("\nBuilding shared vocabulary from both datasets …")
    vocab = build_shared_vocab(csv_paths_a, csv_paths_b)
    print(f"  syscalls:   {len(vocab['syscalls'])}")
    print(f"  bigrams:    {len(vocab['bigrams'])}")
    print(f"  trigrams:   {len(vocab['trigrams'])}")
    print(f"  path_types: {len(vocab['path_types'])}")

    # ── build feature matrices ────────────────────────────────────────────────
    # subtype_map must cover all session_ids; merge both maps
    combined_subtype_map = {**subtype_map_a, **subtype_map_b}

    print(f"\nBuilding feature matrix for {args.name_a} …")
    feat_df_a = build_feature_matrix(csv_paths_a, vocab, combined_subtype_map)
    print(f"  {args.name_a}: {len(meta_a)} sessions, {len(feat_df_a)} chunks")

    print(f"\nBuilding feature matrix for {args.name_b} …")
    feat_df_b = build_feature_matrix(csv_paths_b, vocab, combined_subtype_map)
    print(f"  {args.name_b}: {len(meta_b)} sessions, {len(feat_df_b)} chunks")

    feature_dim = len([c for c in feat_df_a.columns
                       if c not in {"session_id", "subtype"}])
    print(f"\nFeature dimension (total columns): {feature_dim}")

    # ── within-software baselines (70/30 stratified split) ───────────────────
    all_results = []

    # within A
    print(f"\n=== within_{args.name_a} (70/30 split) ===")
    train_sids_a, test_sids_a = train_test_split(
        meta_a["session_id"],
        test_size=0.3,
        stratify=meta_a["subtype"],
        random_state=42,
    )
    within_train_a = feat_df_a[feat_df_a["session_id"].isin(train_sids_a)].reset_index(drop=True)
    within_test_a  = feat_df_a[feat_df_a["session_id"].isin(test_sids_a)].reset_index(drop=True)
    print(f"  train={len(train_sids_a)} sessions / "
          f"test={len(test_sids_a)} sessions")
    all_results.extend(_run_scenario(
        f"within_{args.name_a}", within_train_a, within_test_a, label_enc, n_classes
    ))

    # within B
    print(f"\n=== within_{args.name_b} (70/30 split) ===")
    train_sids_b, test_sids_b = train_test_split(
        meta_b["session_id"],
        test_size=0.3,
        stratify=meta_b["subtype"],
        random_state=42,
    )
    within_train_b = feat_df_b[feat_df_b["session_id"].isin(train_sids_b)].reset_index(drop=True)
    within_test_b  = feat_df_b[feat_df_b["session_id"].isin(test_sids_b)].reset_index(drop=True)
    print(f"  train={len(train_sids_b)} sessions / "
          f"test={len(test_sids_b)} sessions")
    all_results.extend(_run_scenario(
        f"within_{args.name_b}", within_train_b, within_test_b, label_enc, n_classes
    ))

    # ── cross-software transfer ───────────────────────────────────────────────
    print(f"\n=== {args.name_a}_to_{args.name_b} (all sessions) ===")
    print(f"  train={len(meta_a)} {args.name_a} sessions / "
          f"test={len(meta_b)} {args.name_b} sessions")
    all_results.extend(_run_scenario(
        f"{args.name_a}_to_{args.name_b}", feat_df_a, feat_df_b, label_enc, n_classes
    ))

    print(f"\n=== {args.name_b}_to_{args.name_a} (all sessions) ===")
    print(f"  train={len(meta_b)} {args.name_b} sessions / "
          f"test={len(meta_a)} {args.name_a} sessions")
    all_results.extend(_run_scenario(
        f"{args.name_b}_to_{args.name_a}", feat_df_b, feat_df_a, label_enc, n_classes
    ))

    # ── save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    out_csv = os.path.join(args.output_dir, "cross_software_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nResults → {out_csv}")

    # ── console table ─────────────────────────────────────────────────────────
    scenario_labels = [
        f"within_{args.name_a}",
        f"within_{args.name_b}",
        f"{args.name_a}_to_{args.name_b}",
        f"{args.name_b}_to_{args.name_a}",
    ]
    col_w = max(len(s) for s in scenario_labels) + 2

    print("\n" + "=" * (10 + col_w * 4))
    header = f"{'mode':<10}" + "".join(f"{s:>{col_w}}" for s in scenario_labels)
    print(header)
    print("-" * (10 + col_w * 4))

    # pivot: mode -> scenario -> accuracy
    pivot: dict = {}
    for row in all_results:
        pivot.setdefault(row["mode"], {})[row["scenario"]] = row["session_accuracy"]

    for mode in ISOLATION_MODES:
        row_str = f"{mode:<10}"
        for sc in scenario_labels:
            acc = pivot.get(mode, {}).get(sc, float("nan"))
            row_str += f"{acc:>{col_w}.3f}"
        print(row_str)
    print("=" * (10 + col_w * 4))

    # ── degradation report ────────────────────────────────────────────────────
    print(f"\nDegradation report  "
          f"(positive = accuracy drops when crossing software boundaries)")
    print(f"  {'mode':<10}  "
          f"{'within_A - A_to_B':>20}  "
          f"{'within_B - B_to_A':>20}")
    print("  " + "-" * 54)
    for mode in ISOLATION_MODES:
        m = pivot.get(mode, {})
        wa  = m.get(f"within_{args.name_a}", float("nan"))
        atb = m.get(f"{args.name_a}_to_{args.name_b}", float("nan"))
        wb  = m.get(f"within_{args.name_b}", float("nan"))
        bta = m.get(f"{args.name_b}_to_{args.name_a}", float("nan"))
        deg_a = wa - atb
        deg_b = wb - bta
        print(f"  {mode:<10}  {deg_a:>+20.3f}  {deg_b:>+20.3f}")


if __name__ == "__main__":
    main()
