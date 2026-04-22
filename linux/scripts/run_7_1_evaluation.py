"""
生成毕设论文 7.1 节（常规性能评估）的专属实验数据。
使用最优配置：F5 档位特征（序列+上下文） + 噪声过滤 + 随机森林
同时输出 Chunk 级别 和 Session 级别的评估指标。
"""
import argparse
import json
import gc
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# 引入原始仓库模块
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.syscall_anomaly.models import make_random_forest_model, _align_features
from linux.scripts.eval_per_software import (
    build_vocab_streaming,
    build_features_streaming,
    stratified_split
)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """通用的指标计算函数"""
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
        "f1": f1, "fpr": fpr, "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

def evaluate_software(raw_dir: Path, software_name: str, args: argparse.Namespace) -> dict:
    print(f"\n[{software_name}] 开始进行 7.1 节常规性能评估...")
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"未在 {raw_dir} 找到 CSV 文件！")

    # 1. 构建词表（开启噪声过滤 -> F5）
    print(f"[{software_name}] 步骤 1/4: 构建流式词表与特征过滤...")
    vocab, raw_events, filtered_events = build_vocab_streaming(files, args)

    # 2. 提取特征（开启上下文 -> F5）
    print(f"[{software_name}] 步骤 2/4: 提取 F5 档位融合特征...")
    _, ctx_df = build_features_streaming(files, vocab)

    # 3. 划分数据集
    print(f"[{software_name}] 步骤 3/4: 划分训练集与测试集...")
    train_df, test_df = stratified_split(ctx_df, args.test_size, args.random_state)
    del ctx_df
    gc.collect()

    # 4. 训练模型
    print(f"[{software_name}] 步骤 4/4: 训练随机森林并进行双层级评估...")
    model = make_random_forest_model(
        train_df,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    
    # 获取 Chunk 级别原始预测
    x_test = _align_features(model, test_df)
    clf = model["sklearn_model"]
    chunk_pred = clf.predict(x_test).astype(np.int32)
    chunk_true = (test_df["label"] == "abnormal").astype(int).to_numpy()
    
    # 4.1 Chunk 级评估
    chunk_metrics = calculate_metrics(chunk_true, chunk_pred)

    # 4.2 Session 级评估 (多数投票)
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
    
    session_metrics = calculate_metrics(sess["true_label"].to_numpy(), sess["pred"].to_numpy())

    print(f"[{software_name}] Chunk级准确率: {chunk_metrics['accuracy']:.4f} | Session级准确率: {session_metrics['accuracy']:.4f}")
    
    return {
        "software": software_name,
        "events_raw": raw_events,
        "events_filtered": filtered_events,
        "chunk_metrics": chunk_metrics,
        "session_metrics": session_metrics
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freecad-dir", type=str, default="data/raw/freecad_only")
    parser.add_argument("--kicad-dir", type=str, default="data/raw/kicad_only")
    parser.add_argument("--output-dir", type=str, default="models/section_7_1")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--random-state", type=int, default=42)
    # Vocab limits
    parser.add_argument("--syscall-limit", type=int, default=32)
    parser.add_argument("--transition-limit", type=int, default=64)
    parser.add_argument("--bigram-limit", type=int, default=64)
    parser.add_argument("--trigram-limit", type=int, default=64)
    parser.add_argument("--context-value-limit", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "FreeCAD": evaluate_software(Path(args.freecad_dir), "FreeCAD", args),
        "KiCad": evaluate_software(Path(args.kicad_dir), "KiCad", args)
    }

    out_file = out_dir / "7_1_evaluation_results.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n所有双层级评估数据已保存至: {out_file}")

if __name__ == "__main__":
    main()