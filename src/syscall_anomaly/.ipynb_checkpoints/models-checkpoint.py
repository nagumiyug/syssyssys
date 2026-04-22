from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


META_COLUMNS = ["session_id", "software", "scenario", "label"]


@dataclass
class TrainResult:
    model: dict[str, object]
    metrics: dict[str, float]


# 数据清洗与特征分离
def _prepare_matrix(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    x_df = feature_df.drop(columns=["label"]).copy()
    categorical = [col for col in ["software"] if col in x_df.columns]
    if categorical:
        x_df = pd.get_dummies(x_df, columns=categorical, dummy_na=False)
    x_df = x_df.drop(columns=["session_id", "scenario"], errors="ignore").fillna(0.0)
    x_df = x_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = (feature_df["label"] == "abnormal").astype(int).to_numpy(dtype=np.int32)
    return x_df, y


def make_centroid_model(train_df: pd.DataFrame) -> dict[str, object]:
    x_df, y = _prepare_matrix(train_df)
    features = x_df.columns.tolist()
    x = x_df.to_numpy(dtype=np.float64)

    normal_x = x[y == 0]
    if len(normal_x) == 0:
        raise ValueError("Training data must contain normal sessions to build a baseline.")

    # 标准化缩放（均值为 0，标准差为 1），避免不同特征尺度对距离计算的影响
    mean = normal_x.mean(axis=0)
    std = normal_x.std(axis=0)
    std[std < 1e-8] = 1.0
    scaled_normal = (normal_x - mean) / std

    # 计算标准空间下的质心
    normal_centroid = scaled_normal.mean(axis=0)

    # 计算每个正常样本到质心的距离
    distances = np.linalg.norm(scaled_normal - normal_centroid, axis=1)

    # 使用 99% 分位数作为阈值
    threshold = float(np.percentile(distances, 99)) if len(distances) > 0 else 1.0

    return {
        "model_type": "centroid",
        "features": features,
        "mean": mean,  # 保存缩放参数
        "std": std,    # 保存缩放参数
        "normal_centroid": normal_centroid,
        "threshold": threshold,
    }


def make_random_forest_model(
    train_df: pd.DataFrame,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> dict[str, object]:
    x_df, y = _prepare_matrix(train_df)
    features = x_df.columns.tolist()
    x = x_df.to_numpy(dtype=np.float64)
    if np.unique(y).size < 2:
        raise ValueError("Training data must contain both normal and abnormal sessions.")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(x, y)
    return {
        "model_type": "random_forest",
        "features": features,
        "sklearn_model": clf,
    }


def _align_features(model: dict[str, object], feature_df: pd.DataFrame) -> np.ndarray:
    x_df, _ = _prepare_matrix(feature_df)
    expected = list(model["features"])
    aligned = x_df.reindex(columns=expected, fill_value=0.0)
    return aligned.to_numpy(dtype=np.float64)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50, 50)))


def evaluate_binary_classifier(
    model: dict[str, object],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> TrainResult:
    trained = make_centroid_model(train_df) if not model else model
    x_test = _align_features(trained, test_df)
    y_test = (test_df["label"] == "abnormal").astype(int).to_numpy(dtype=np.int32)
    model_type = str(trained.get("model_type", "centroid"))

    if model_type == "random_forest":
        clf = trained["sklearn_model"]
        y_pred = clf.predict(x_test).astype(np.int32)
        y_score = clf.predict_proba(x_test)[:, 1]
    else:
        # 【新增】提取训练时的缩放参数
        mean = np.asarray(trained.get("mean", np.zeros(x_test.shape[1])), dtype=np.float64)
        std = np.asarray(trained.get("std", np.ones(x_test.shape[1])), dtype=np.float64)
        normal_centroid = np.asarray(trained["normal_centroid"], dtype=np.float64)
        threshold = float(trained.get("threshold", 1.0))

        # 【新增】对测试集切片进行完全一致的缩放
        x_scaled = (x_test - mean) / std

        # 计算距离并与阈值对比
        distances = np.linalg.norm(x_scaled - normal_centroid, axis=1)
        y_pred = (distances > threshold).astype(np.int32)
        y_score = _sigmoid(distances - threshold)

    tp = int(np.sum((y_test == 1) & (y_pred == 1)))
    tn = int(np.sum((y_test == 0) & (y_pred == 0)))
    fp = int(np.sum((y_test == 0) & (y_pred == 1)))
    fn = int(np.sum((y_test == 1) & (y_pred == 0)))

    accuracy = float((tp + tn) / max(len(y_test), 1))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
    fpr = float(fp / max(fp + tn, 1))

    pos_scores = y_score[y_test == 1]
    neg_scores = y_score[y_test == 0]
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

    return TrainResult(
        model=trained,
        metrics={
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
        },
    )
