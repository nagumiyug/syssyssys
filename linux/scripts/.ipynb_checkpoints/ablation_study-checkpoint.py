"""
Ablation study: proving feature engineering + preprocessing matters more
than classifier choice.

Usage:
  python linux/scripts/ablation_study.py \
      --raw-dir data/raw/freecad_only \
      --output-dir models/ablation/freecad

Pipeline:
  1. Build features TWICE: once without noise filter, once with
  2. Define 5 feature tiers by selecting column subsets
  3. For each (tier, classifier) combo, train + evaluate at session level
  4. Record 5x4=20 results per software
  5. Save CSV + JSON + summary
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.syscall_anomaly.features import (
    NOISE_SYSCALLS,
    build_session_features,
    _ngrams,
    _top_keys,
)
from src.syscall_anomaly.models import _align_features
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


META_COLS = ["session_id", "software", "scenario", "label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation study: feature engineering vs classifier impact.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_vocab_streaming(files: list[Path], apply_noise_filter: bool) -> dict:
    """Build global vocabulary. Optionally apply noise filter."""
    sc = Counter(); tr = Counter(); bg = Counter(); tg = Counter()
    op = Counter(); pt = Counter(); ex = Counter(); rc = Counter()
    for f in files:
        df = pd.read_csv(f)
        if apply_noise_filter:
            df = df[~df["syscall_name"].isin(NOISE_SYSCALLS)]
        syscalls = df["syscall_name"].astype(str).tolist()
        sc.update(syscalls)
        tr.update(f"{a}->{b}" for a, b in zip(syscalls[:-1], syscalls[1:]))
        bg.update("__".join(t) for t in _ngrams(syscalls, 2))
        tg.update("__".join(t) for t in _ngrams(syscalls, 3))
        op.update(df["op_category"].fillna("unknown").astype(str))
        pt.update(df["path_type"].fillna("unknown").astype(str))
        ex.update(df["file_ext"].fillna("unknown").astype(str))
        rc.update(df["return_code"].fillna("unknown").astype(str))
        del df, syscalls
        gc.collect()
    return {
        "syscalls": _top_keys(sc, 32),
        "transitions": _top_keys(tr, 64),
        "bigrams": _top_keys(bg, 64),
        "trigrams": _top_keys(tg, 64),
        "ctx_ops": _top_keys(op, 16),
        "ctx_path_types": _top_keys(pt, 16),
        "ctx_exts": _top_keys(ex, 16),
        "ctx_rcs": _top_keys(rc, 16),
    }


def build_features_streaming(files: list[Path], vocab: dict, apply_noise_filter: bool) -> pd.DataFrame:
    """Build session features with full context columns."""
    rows = []
    for f in files:
        df = pd.read_csv(f)
        if apply_noise_filter:
            df = df[~df["syscall_name"].isin(NOISE_SYSCALLS)]
        rows.append(build_session_features(df, include_context=True, vocab=vocab))
        del df
        gc.collect()
    full_df = pd.concat(rows, ignore_index=True).fillna(0.0)
    del rows
    gc.collect()
    return full_df


def select_tier_columns(all_cols: list[str], tier: str) -> list[str]:
    """Return the subset of feature columns for the given tier.

    Tier design (deliberately progressive to show feature engineering impact):
      F1: 3 scalar features only (entropies + self_loop_ratio) — intentionally weak
      F2: F1 + raw syscall frequency counts (freq_count::*)
      F3: F2 + bigram + trigram (seq2_norm::*, seq3_norm::*)
      F4: F3 + transitions + full context (trans_norm::*, ctx_*)
      F5: F4 columns applied to noise-filtered data
    """
    def by_prefix(*prefixes):
        return [c for c in all_cols if any(c.startswith(p) for p in prefixes)]

    scalar_seq = [c for c in ("syscall_entropy", "transition_entropy", "self_loop_ratio") if c in all_cols]

    if tier == "F1":
        return scalar_seq
    if tier == "F2":
        return scalar_seq + by_prefix("freq_count::")
    if tier == "F3":
        return scalar_seq + by_prefix("freq_count::", "seq2_norm::", "seq3_norm::")
    if tier == "F4":
        return scalar_seq + by_prefix("freq_count::", "seq2_norm::", "seq3_norm::", "trans_norm::", "ctx_")
    if tier == "F5":
        return scalar_seq + by_prefix("freq_count::", "seq2_norm::", "seq3_norm::", "trans_norm::", "ctx_")
    raise ValueError(f"Unknown tier: {tier}")


def stratified_split(feature_df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = feature_df.copy()
    feature_df["base_session"] = feature_df["session_id"].apply(lambda x: str(x).split("_chunk_")[0])
    session_meta = feature_df[["base_session", "software", "label"]].drop_duplicates()
    train_sessions, test_sessions = [], []
    for _, g in session_meta.groupby(["software", "label"], sort=True):
        s = g.sample(frac=1.0, random_state=random_state)
        tc = max(1, int(round(len(s) * test_size)))
        test_sessions.extend(s["base_session"].iloc[:tc].tolist())
        train_sessions.extend(s["base_session"].iloc[tc:].tolist())
    test_df = feature_df[feature_df["base_session"].isin(test_sessions)].drop(columns=["base_session"])
    train_df = feature_df[feature_df["base_session"].isin(train_sessions)].drop(columns=["base_session"])
    return train_df, test_df


def train_classifier(name: str, x: np.ndarray, y: np.ndarray, random_state: int):
    """Train a classifier by name. Returns (model, needs_scaling_bool)."""
    if name == "LogReg":
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, random_state=random_state,
        )
        clf.fit(x_scaled, y)
        return clf, scaler
    if name == "DecisionTree":
        clf = DecisionTreeClassifier(
            max_depth=9, class_weight="balanced", random_state=random_state,
        )
        clf.fit(x, y)
        return clf, None
    if name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=9, class_weight="balanced",
            random_state=random_state, n_jobs=-1,
        )
        clf.fit(x, y)
        return clf, None
    if name == "GradientBoosting":
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=random_state,
        )
        clf.fit(x, y)
        return clf, None
    raise ValueError(f"Unknown classifier: {name}")


def predict_proba(clf, scaler, x: np.ndarray) -> np.ndarray:
    if scaler is not None:
        x = scaler.transform(x)
    return clf.predict_proba(x)[:, 1]


def _metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = max(len(y_true), 1)
    accuracy = float((tp + tn) / n)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
    fpr = float(fp / max(fp + tn, 1))
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "fpr": fpr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
    }


def evaluate_both_levels(clf, scaler, test_df: pd.DataFrame, feature_cols: list[str]) -> tuple[dict, dict]:
    """Return (chunk_level_metrics, session_level_metrics)."""
    x_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    y_score = predict_proba(clf, scaler, x_test)
    chunk_pred = (y_score >= 0.5).astype(np.int32)
    chunk_true = (test_df["label"] == "abnormal").astype(int).to_numpy()

    # Chunk-level metrics
    chunk_metrics = _metrics_from_predictions(chunk_true, chunk_pred)

    # Session-level aggregation (majority vote)
    df_out = pd.DataFrame({
        "base_session": test_df["session_id"].apply(lambda x: str(x).split("_chunk_")[0]).values,
        "label": chunk_true,
        "pred": chunk_pred,
    })
    sess = df_out.groupby("base_session").agg(
        true_label=("label", "max"),
        n_chunks=("pred", "size"),
        n_abnormal=("pred", "sum"),
    ).reset_index()
    sess["pred"] = (sess["n_abnormal"] / sess["n_chunks"] > 0.5).astype(int)
    session_metrics = _metrics_from_predictions(
        sess["true_label"].to_numpy(), sess["pred"].to_numpy()
    )
    return chunk_metrics, session_metrics


def _run_experiments_for_dataset(feat: pd.DataFrame, tiers_to_run: list[str],
                                  classifiers: list[str], software: str,
                                  args: argparse.Namespace, results: list) -> None:
    """Train/eval for a set of tiers on a single feature DataFrame.

    Memory-optimized: caller passes in a single `feat` DataFrame (either raw or
    filtered) and only the tiers that use that dataset. Avoids holding both
    raw and filtered features in memory at once.
    """
    train_df, test_df = stratified_split(feat, args.test_size, args.random_state)
    for tier in tiers_to_run:
        feature_cols = select_tier_columns([c for c in feat.columns if c not in META_COLS], tier)
        n_feat = len(feature_cols)
        x_train = train_df[feature_cols].to_numpy(dtype=np.float64)
        y_train = (train_df["label"] == "abnormal").astype(np.int32).to_numpy()

        for clf_name in classifiers:
            try:
                clf, scaler = train_classifier(clf_name, x_train, y_train, args.random_state)
                chunk_m, sess_m = evaluate_both_levels(clf, scaler, test_df, feature_cols)
                print(f"      {tier} | {clf_name:<18s} | feat={n_feat:3d} | chunk_acc={chunk_m['accuracy']:.4f} chunk_f1={chunk_m['f1']:.4f} | sess_acc={sess_m['accuracy']:.4f}")
                results.append({
                    "software": software,
                    "tier": tier,
                    "classifier": clf_name,
                    "n_features": n_feat,
                    "chunk_accuracy": chunk_m["accuracy"],
                    "chunk_precision": chunk_m["precision"],
                    "chunk_recall": chunk_m["recall"],
                    "chunk_f1": chunk_m["f1"],
                    "chunk_fpr": chunk_m["fpr"],
                    "chunk_tp": chunk_m["tp"], "chunk_tn": chunk_m["tn"],
                    "chunk_fp": chunk_m["fp"], "chunk_fn": chunk_m["fn"],
                    "session_accuracy": sess_m["accuracy"],
                    "session_f1": sess_m["f1"],
                    "session_fpr": sess_m["fpr"],
                    "session_tp": sess_m["tp"], "session_tn": sess_m["tn"],
                    "session_fp": sess_m["fp"], "session_fn": sess_m["fn"],
                })
                del clf, scaler
                gc.collect()
            except Exception as e:
                print(f"      {tier} | {clf_name:<18s} | ERROR: {e}")
                results.append({
                    "software": software, "tier": tier, "classifier": clf_name,
                    "n_features": n_feat,
                    "chunk_accuracy": float("nan"), "chunk_precision": float("nan"),
                    "chunk_recall": float("nan"), "chunk_f1": float("nan"), "chunk_fpr": float("nan"),
                    "chunk_tp": 0, "chunk_tn": 0, "chunk_fp": 0, "chunk_fn": 0,
                    "session_accuracy": float("nan"), "session_f1": float("nan"), "session_fpr": float("nan"),
                    "session_tp": 0, "session_tn": 0, "session_fp": 0, "session_fn": 0,
                })
        del x_train, y_train
        gc.collect()


def run_ablation(files: list[Path], output_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    tiers = ["F1", "F2", "F3", "F4", "F5"]
    classifiers = ["LogReg", "DecisionTree", "RandomForest", "GradientBoosting"]
    results = []

    # Phase A: Raw features (F1-F4)
    print("[1/4] Building vocab + features WITHOUT noise filter (for F1-F4)...")
    vocab_raw = build_vocab_streaming(files, apply_noise_filter=False)
    feat_raw = build_features_streaming(files, vocab_raw, apply_noise_filter=False)
    software = str(feat_raw["software"].iloc[0])
    print(f"      Chunks raw:      {len(feat_raw)}, Features: {len([c for c in feat_raw.columns if c not in META_COLS])}")

    print("[2/4] Running experiments F1-F4 on raw features...")
    _run_experiments_for_dataset(feat_raw, tiers[:-1], classifiers, software, args, results)

    # Free raw features before loading filtered
    del feat_raw, vocab_raw
    gc.collect()

    # Phase B: Filtered features (F5)
    print("[3/4] Building vocab + features WITH noise filter (for F5)...")
    vocab_filt = build_vocab_streaming(files, apply_noise_filter=True)
    feat_filt = build_features_streaming(files, vocab_filt, apply_noise_filter=True)
    print(f"      Chunks filtered: {len(feat_filt)}, Features: {len([c for c in feat_filt.columns if c not in META_COLS])}")

    print("[4/4] Running experiments F5 on filtered features...")
    _run_experiments_for_dataset(feat_filt, ["F5"], classifiers, software, args, results)

    del feat_filt, vocab_filt
    gc.collect()

    results_df = pd.DataFrame(results)

    # Step 4: Save
    print("[4/4] Saving results...")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "ablation_results.csv", index=False)

    # Compute delta metrics for BOTH chunk-level and session-level
    def compute_deltas(level: str) -> dict:
        col = f"{level}_accuracy"
        matrix = results_df.pivot(index="classifier", columns="tier", values=col)
        dfeat = (matrix.max(axis=1) - matrix.min(axis=1)).to_dict()
        dclf = (matrix.max(axis=0) - matrix.min(axis=0)).to_dict()
        return {
            "matrix": matrix,
            "delta_features_per_classifier": dfeat,
            "delta_classifiers_per_tier": dclf,
            "max_delta_features": float(max(dfeat.values())),
            "max_delta_classifiers": float(max(dclf.values())),
            "hypothesis_confirmed": max(dfeat.values()) > max(dclf.values()),
        }

    chunk_analysis = compute_deltas("chunk")
    session_analysis = compute_deltas("session")

    summary = {
        "software": software,
        "n_experiments": len(results_df),
        "chunk_level": {
            "accuracy_matrix": chunk_analysis["matrix"].to_dict(),
            "delta_features_per_classifier": chunk_analysis["delta_features_per_classifier"],
            "delta_classifiers_per_tier": chunk_analysis["delta_classifiers_per_tier"],
            "max_delta_features": chunk_analysis["max_delta_features"],
            "max_delta_classifiers": chunk_analysis["max_delta_classifiers"],
            "hypothesis_confirmed": chunk_analysis["hypothesis_confirmed"],
        },
        "session_level": {
            "accuracy_matrix": session_analysis["matrix"].to_dict(),
            "delta_features_per_classifier": session_analysis["delta_features_per_classifier"],
            "delta_classifiers_per_tier": session_analysis["delta_classifiers_per_tier"],
            "max_delta_features": session_analysis["max_delta_features"],
            "max_delta_classifiers": session_analysis["max_delta_classifiers"],
            "hypothesis_confirmed": session_analysis["hypothesis_confirmed"],
        },
    }

    (output_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    def print_analysis(title: str, analysis: dict) -> None:
        print(f"\n--- {title} ---")
        print("Accuracy matrix (rows=classifier, cols=tier):")
        print(analysis["matrix"].to_string(float_format="%.4f"))
        print("\nDelta_features per classifier (range across tiers):")
        for k, v in analysis["delta_features_per_classifier"].items():
            print(f"  {k:<18s}: {v:.4f}")
        print("Delta_classifiers per tier (range across classifiers):")
        for k, v in analysis["delta_classifiers_per_tier"].items():
            print(f"  {k:<4s}: {v:.4f}")
        print(f"max_delta_features    = {analysis['max_delta_features']:.4f}")
        print(f"max_delta_classifiers = {analysis['max_delta_classifiers']:.4f}")
        print(f"Hypothesis confirmed  = {analysis['hypothesis_confirmed']}")

    print(f"\n=== {software} Ablation Summary ===")
    print_analysis("CHUNK-LEVEL (primary for ablation)", chunk_analysis)
    print_analysis("SESSION-LEVEL (reference)", session_analysis)

    return results_df


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {raw_dir}")
    print(f"Found {len(files)} raw CSV files in {raw_dir}")

    run_ablation(files, output_dir, args)
    print(f"\nDone. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
