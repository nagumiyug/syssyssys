"""
P2 case study extractor: pick one normal + one anomalous session, run them
through the F5 pipeline (with section_7_1 default hyperparameters), and dump
per-window features + per-window predicted probabilities + session-vote stats
so we can illustrate the detection mechanism end-to-end.

Output (under <output-dir>/<case_id>/):
  - per_window_predictions.csv  (chunk_id, syscalls_in_chunk_top, freq features,
                                 ngram features, transition features, ctx features,
                                 predicted_proba_abnormal, predicted_label)
  - session_vote_summary.json   (n_chunks, n_predicted_abnormal, vote_threshold,
                                 final_session_pred, true_label)

Usage:
    python case_study_extract.py \\
        --raw-dir data/raw/freecad_only \\
        --normal-case freecad_normal_r5 \\
        --abnormal-case freecad_source_copy_r5 \\
        --output-dir models/case_study/freecad

Both case names must match the prefix of CSV files (without .csv) under raw-dir.
The two case sessions are forcibly held out from training; everything else
follows section_7_1's random_state=42 stratified train/test partition.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.syscall_anomaly.models import make_random_forest_model, _align_features
from linux.scripts.eval_per_software import (
    build_vocab_streaming,
    build_features_streaming,
    stratified_split,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=str, required=True)
    p.add_argument("--normal-case", type=str, required=True,
                   help="Base CSV stem for normal case, e.g. freecad_normal_r5")
    p.add_argument("--abnormal-case", type=str, default=None,
                   help="(Deprecated, single-case form) Base CSV stem for an anomalous case")
    p.add_argument("--abnormal-cases", type=str, default=None,
                   help="Comma-separated list of anomalous case stems, e.g. "
                        "'freecad_bulk_export_r5,freecad_source_copy_r5,freecad_project_scan_r5'")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=9)
    p.add_argument("--random-state", type=int, default=42)
    # vocab limits (matching run_7_1_evaluation.py defaults)
    p.add_argument("--syscall-limit", type=int, default=32)
    p.add_argument("--transition-limit", type=int, default=64)
    p.add_argument("--bigram-limit", type=int, default=64)
    p.add_argument("--trigram-limit", type=int, default=64)
    p.add_argument("--context-value-limit", type=int, default=16)
    return p.parse_args()


def find_session_id_for_case(ctx_df: pd.DataFrame, case_stem: str) -> str:
    """Map a case CSV stem to the actual base session_id used by the feature
    pipeline. The stem is something like 'freecad_normal_r5'; the pipeline
    creates session_id values like '<uuid>_chunk_0' that share a base session
    derived from the CSV.

    We use the 'scenario' field + a deterministic ordering by base_session to
    pick the right one. Simpler: each raw CSV produces exactly one base session
    (one UUID), and we have one CSV per case. So we group ctx_df by base_session
    and rely on the file we singled out.
    """
    # ctx_df does not carry the source filename, so we need an alternative.
    # Trick: count base sessions per (software, scenario, label). For each
    # (scenario, label) bucket the run number r1..r20 maps deterministically
    # via the *order in which CSV files were globbed*, but that order is not
    # stable across machines. Therefore we rely on the caller passing the
    # actual session_id resolved against raw CSVs (see resolve_session_ids()).
    raise NotImplementedError("Use resolve_session_ids instead.")


def resolve_session_ids(raw_dir: Path, case_stems: list[str]) -> dict[str, str]:
    """Read each case CSV's first row to extract its session_id; return
    {case_stem: base_session_id}.
    """
    out = {}
    for stem in case_stems:
        f = raw_dir / f"{stem}.csv"
        if not f.exists():
            raise FileNotFoundError(f"Case CSV not found: {f}")
        # Read just the first row to grab session_id
        head = pd.read_csv(f, nrows=1)
        if "session_id" not in head.columns:
            raise ValueError(f"{f} missing session_id column")
        out[stem] = str(head["session_id"].iloc[0])
    return out


def stratified_split_holdout(feature_df: pd.DataFrame, test_size: float,
                             random_state: int, holdout_sessions: list[str]
                             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Like stratified_split but force `holdout_sessions` into the test set."""
    feature_df = feature_df.copy()
    feature_df["base_session"] = feature_df["session_id"].apply(
        lambda x: str(x).split("_chunk_")[0]
    )
    holdout_set = set(holdout_sessions)

    # Stratified split on the remainder
    rest_df = feature_df[~feature_df["base_session"].isin(holdout_set)].copy()
    session_meta = rest_df[["base_session", "software", "label"]].drop_duplicates()
    train_sessions, test_sessions = [], []
    for _, g in session_meta.groupby(["software", "label"], sort=True):
        s = g.sample(frac=1.0, random_state=random_state)
        tc = max(1, int(round(len(s) * test_size)))
        test_sessions.extend(s["base_session"].iloc[:tc].tolist())
        train_sessions.extend(s["base_session"].iloc[tc:].tolist())

    train_df = feature_df[feature_df["base_session"].isin(train_sessions)].drop(
        columns=["base_session"]
    )
    test_df = feature_df[feature_df["base_session"].isin(test_sessions)].drop(
        columns=["base_session"]
    )
    case_df = feature_df[feature_df["base_session"].isin(holdout_set)].drop(
        columns=["base_session"]
    )
    if train_df.empty:
        raise SystemExit("Train split is empty.")
    if case_df.empty:
        raise SystemExit("Case holdout is empty — session_id resolution failed.")
    return train_df, test_df, case_df


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the list of case stems: 1 normal + 1..N abnormal cases.
    # --abnormal-cases (plural, comma-separated) takes precedence over the
    # legacy --abnormal-case (single) for backward compatibility.
    if args.abnormal_cases:
        abnormal_stems = [s.strip() for s in args.abnormal_cases.split(",") if s.strip()]
    elif args.abnormal_case:
        abnormal_stems = [args.abnormal_case]
    else:
        raise SystemExit("Must provide --abnormal-case or --abnormal-cases")
    case_stems = [args.normal_case] + abnormal_stems
    case_session_ids = resolve_session_ids(raw_dir, case_stems)
    print(f"[case_study] resolved session ids:")
    for stem, sid in case_session_ids.items():
        print(f"   {stem} -> {sid}")

    files = sorted(raw_dir.glob("*.csv"))
    print(f"[case_study] {len(files)} CSV files in {raw_dir}")

    # 1. vocab
    print("[case_study] step 1/4: build vocab (F5: noise filter on)")
    vocab, raw_events, filtered_events = build_vocab_streaming(files, args)
    print(f"   raw_events={raw_events:,}  filtered={filtered_events:,}")

    # 2. features
    print("[case_study] step 2/4: build F5 features (with context)")
    _, ctx_df = build_features_streaming(files, vocab)
    print(f"   ctx_df rows={len(ctx_df):,}  cols={len(ctx_df.columns)}")

    # 3. split with case sessions held out
    print("[case_study] step 3/4: stratified split with case holdout")
    train_df, test_df, case_df = stratified_split_holdout(
        ctx_df, args.test_size, args.random_state,
        list(case_session_ids.values()),
    )
    print(f"   train={len(train_df)} chunks   test={len(test_df)} chunks   "
          f"case={len(case_df)} chunks")
    del ctx_df
    gc.collect()

    # 4. train + predict on case
    print("[case_study] step 4/4: train RF + predict on case sessions")
    model = make_random_forest_model(
        train_df,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    x_case = _align_features(model, case_df)
    clf = model["sklearn_model"]
    proba_abn = clf.predict_proba(x_case)[:, 1]
    pred_label = (proba_abn >= 0.5).astype(int)

    # 5. dump per-window predictions for each case
    case_df = case_df.reset_index(drop=True).copy()
    case_df["proba_abnormal"] = proba_abn
    case_df["pred_label"] = pred_label
    case_df["base_session"] = case_df["session_id"].apply(
        lambda x: str(x).split("_chunk_")[0]
    )
    case_df["chunk_index"] = case_df["session_id"].apply(
        lambda x: int(str(x).split("_chunk_")[-1]) if "_chunk_" in str(x) else 0
    )

    # Reverse map base_session -> case_stem for nicer output
    sid_to_stem = {sid: stem for stem, sid in case_session_ids.items()}

    summaries = []
    for stem, sid in case_session_ids.items():
        sub = case_df[case_df["base_session"] == sid].sort_values("chunk_index")
        out_csv = out_dir / f"{stem}__per_window_predictions.csv"
        # Pick a manageable column subset for the dump (avoid 200+ feature cols)
        feat_cols = [c for c in sub.columns if c.startswith((
            "freq_count::", "freq_norm::", "seq2_norm::", "seq3_norm::",
            "trans_norm::", "ctx_op::", "ctx_path_type::", "ctx_ext::",
            "ctx_rc::", "ctx_gap::"
        ))]
        # global scalars
        feat_cols += [c for c in sub.columns if c in (
            "syscall_entropy", "transition_entropy", "self_loop_ratio",
            "ctx_design_file_ratio", "ctx_source_copy_target_ratio",
            "ctx_has_project_source", "ctx_read_write_balance",
            "ctx_path_type_switch_ratio",
        )]
        keep_cols = [
            "base_session", "chunk_index", "session_id",
            "software", "scenario", "label",
            "proba_abnormal", "pred_label",
        ] + feat_cols
        keep_cols = [c for c in keep_cols if c in sub.columns]
        sub[keep_cols].to_csv(out_csv, index=False)
        print(f"[case_study] wrote {out_csv}  ({len(sub)} chunks, "
              f"{len(keep_cols)} columns)")

        # session vote summary
        n_chunks = int(len(sub))
        n_abn_pred = int(sub["pred_label"].sum())
        true_label_int = int((sub["label"].iloc[0] == "abnormal"))
        sess_pred = int((n_abn_pred / n_chunks) > 0.5) if n_chunks > 0 else 0
        summaries.append({
            "case_stem": stem,
            "base_session_id": sid,
            "scenario": str(sub["scenario"].iloc[0]) if n_chunks else "",
            "true_label": "abnormal" if true_label_int else "normal",
            "n_chunks": n_chunks,
            "n_predicted_abnormal_chunks": n_abn_pred,
            "fraction_abnormal": n_abn_pred / max(n_chunks, 1),
            "vote_threshold": 0.5,
            "final_session_pred": "abnormal" if sess_pred else "normal",
            "correct": (sess_pred == true_label_int),
            "min_proba_abnormal": float(sub["proba_abnormal"].min()),
            "max_proba_abnormal": float(sub["proba_abnormal"].max()),
            "mean_proba_abnormal": float(sub["proba_abnormal"].mean()),
        })

    out_json = out_dir / "session_vote_summary.json"
    out_json.write_text(json.dumps(summaries, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[case_study] wrote {out_json}")
    print("[case_study] done.")


if __name__ == "__main__":
    main()
