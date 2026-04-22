from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.syscall_anomaly.features import build_session_features, infer_feature_vocab
from src.syscall_anomaly.models import (
    _align_features,
    evaluate_binary_classifier,
    make_random_forest_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Linux syscall anomaly baselines.")
    parser.add_argument("--input", required=True, help="Path to raw events CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and trained models.")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split ratio.")
    parser.add_argument("--syscall-limit", type=int, default=32)
    parser.add_argument("--transition-limit", type=int, default=64)
    parser.add_argument("--bigram-limit", type=int, default=64)
    parser.add_argument("--trigram-limit", type=int, default=64)
    parser.add_argument("--context-value-limit", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees for RandomForest.")
    parser.add_argument("--max-depth", type=int, default=9, help="Max tree depth for RandomForest.")
    return parser.parse_args()


def stratified_split(feature_df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = feature_df.copy()
    feature_df["base_session"] = feature_df["session_id"].apply(lambda x: str(x).split("_chunk_")[0])
    session_meta = feature_df[["base_session", "software", "label"]].drop_duplicates()

    train_sessions: list[str] = []
    test_sessions: list[str] = []

    for _, group_df in session_meta.groupby(["software", "label"], sort=True):
        shuffled = group_df.sample(frac=1.0, random_state=42)
        test_count = max(1, int(round(len(shuffled) * test_size)))
        test_sessions.extend(shuffled["base_session"].iloc[:test_count].tolist())
        train_sessions.extend(shuffled["base_session"].iloc[test_count:].tolist())

    test_df = feature_df[feature_df["base_session"].isin(test_sessions)].drop(columns=["base_session"])
    train_df = feature_df[feature_df["base_session"].isin(train_sessions)].drop(columns=["base_session"])

    if train_df.empty or test_df.empty:
        raise SystemExit("Train/test split produced an empty partition.")

    return train_df, test_df


def evaluate_session_level(model: dict, test_df: pd.DataFrame) -> dict:
    """
    Aggregate chunk-level predictions to session level via majority vote.
    A session is flagged abnormal if more than 50% of its chunks are predicted abnormal.
    """
    x_test = _align_features(model, test_df)
    clf = model["sklearn_model"]
    chunk_pred = clf.predict(x_test).astype(np.int32)
    chunk_score = clf.predict_proba(x_test)[:, 1]

    df_out = pd.DataFrame({
        "base_session": test_df["session_id"].apply(lambda x: str(x).split("_chunk_")[0]).values,
        "label": (test_df["label"] == "abnormal").astype(int).values,
        "pred": chunk_pred,
        "score": chunk_score,
    })

    sess = df_out.groupby("base_session").agg(
        true_label=("label", "max"),
        n_chunks=("pred", "size"),
        n_abnormal_chunks=("pred", "sum"),
        max_score=("score", "max"),
        mean_score=("score", "mean"),
    ).reset_index()

    sess["pred"] = (sess["n_abnormal_chunks"] / sess["n_chunks"] > 0.5).astype(int)

    y_true = sess["true_label"].to_numpy()
    y_pred = sess["pred"].to_numpy()
    y_score = sess["mean_score"].to_numpy()

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

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        roc_auc = 0.5
    else:
        sorted_neg = np.sort(neg_scores)
        wins = 0.0
        for pos in pos_scores:
            left = np.searchsorted(sorted_neg, pos, side="left")
            right = np.searchsorted(sorted_neg, pos, side="right")
            wins += float(left) + 0.5 * float(right - left)
        roc_auc = float(wins / (len(pos_scores) * len(neg_scores)))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "roc_auc": roc_auc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def group_distribution_counts(feature_df: pd.DataFrame) -> dict[str, int]:
    counts = feature_df.groupby(["software", "label"], sort=True).size().to_dict()
    return {f"{software}/{label}": int(count) for (software, label), count in counts.items()}


def write_summary(output_dir: Path, input_csv: Path, metrics: dict, session_count: int, event_count: int) -> None:
    summary = "\n".join(
        [
            "# Evaluation Summary",
            "",
            f"- Input data: `{input_csv}`",
            f"- {session_count} sessions",
            f"- {event_count} events",
            "",
            "## Sequence-only model (session-level)",
            f"- Accuracy: {metrics['base_model']['accuracy']:.3f}",
            f"- FPR: {metrics['base_model']['fpr']:.3f}",
            f"- Recall: {metrics['base_model']['recall']:.3f}",
            f"- F1: {metrics['base_model']['f1']:.3f}",
            "",
            "## Sequence + context model (session-level)",
            f"- Accuracy: {metrics['context_model']['accuracy']:.3f}",
            f"- FPR: {metrics['context_model']['fpr']:.3f}",
            f"- Recall: {metrics['context_model']['recall']:.3f}",
            f"- F1: {metrics['context_model']['f1']:.3f}",
            "",
            "## Delta",
            f"- Accuracy gain: {metrics['accuracy_gain']:.3f}",
            f"- FPR drop: {metrics['fpr_drop']:.3f}",
            "",
        ]
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")


def serialize_model(model: dict[str, object], output_dir: Path, stem: str) -> dict[str, object]:
    data = {"model_type": model.get("model_type", "unknown"), "features": list(model["features"])}
    if model.get("model_type") == "random_forest":
        joblib_path = output_dir / f"{stem}.joblib"
        joblib.dump(model["sklearn_model"], joblib_path)
        data["joblib_file"] = str(joblib_path.name)
    else:
        data.update({
            "mean": list(map(float, model.get("mean", []))),
            "std": list(map(float, model.get("std", []))),
            "normal_centroid": list(map(float, model["normal_centroid"])),
            "threshold": float(model.get("threshold", 1.0)),
        })
    return data


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input)
    vocab = infer_feature_vocab(
        raw_df,
        syscall_limit=args.syscall_limit,
        transition_limit=args.transition_limit,
        bigram_limit=args.bigram_limit,
        trigram_limit=args.trigram_limit,
        context_value_limit=args.context_value_limit,
    )
    base_df = build_session_features(raw_df, include_context=False, vocab=vocab)
    ctx_df = build_session_features(raw_df, include_context=True, vocab=vocab)

    if base_df["label"].nunique() < 2 or len(base_df) < 4:
        raise SystemExit("Need at least 4 sessions and both normal/abnormal labels for train/eval.")

    base_train, base_test = stratified_split(base_df, args.test_size)
    ctx_train, ctx_test = stratified_split(ctx_df, args.test_size)

    base_model = make_random_forest_model(base_train, n_estimators=args.n_estimators, max_depth=args.max_depth)
    ctx_model = make_random_forest_model(ctx_train, n_estimators=args.n_estimators, max_depth=args.max_depth)

    # Session-level evaluation (majority vote across chunks per session)
    base_session_metrics = evaluate_session_level(base_model, base_test)
    ctx_session_metrics = evaluate_session_level(ctx_model, ctx_test)

    # Also keep chunk-level metrics for reference
    base_chunk_result = evaluate_binary_classifier(base_model, base_train, base_test)
    ctx_chunk_result = evaluate_binary_classifier(ctx_model, ctx_train, ctx_test)

    metrics = {
        "base_model": base_session_metrics,
        "context_model": ctx_session_metrics,
        "accuracy_gain": ctx_session_metrics["accuracy"] - base_session_metrics["accuracy"],
        "fpr_drop": base_session_metrics["fpr"] - ctx_session_metrics["fpr"],
        "chunk_level_base": base_chunk_result.metrics,
        "chunk_level_context": ctx_chunk_result.metrics,
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "base_model.json").write_text(
        json.dumps(serialize_model(base_model, output_dir, "base_model"), ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "context_model.json").write_text(
        json.dumps(serialize_model(ctx_model, output_dir, "context_model"), ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary(output_dir, Path(args.input), metrics, raw_df["session_id"].nunique(), len(raw_df))

    print("train_distribution:", json.dumps(group_distribution_counts(base_train), ensure_ascii=False))
    print("test_distribution:", json.dumps(group_distribution_counts(base_test), ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
