"""
读取 7.1 节生成的实验数据 JSON，绘制包含 Chunk 和 Session 双层级的高清对比图表。
"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def plot_dual_level_performance(results: dict, out_dir: Path):
    """绘制 Chunk vs Session 性能对比柱状图"""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # 提取四个系列的成绩
    fc_chunk = [results["FreeCAD"]["chunk_metrics"][m] * 100 for m in metrics_to_plot]
    fc_sess = [results["FreeCAD"]["session_metrics"][m] * 100 for m in metrics_to_plot]
    kc_chunk = [results["KiCad"]["chunk_metrics"][m] * 100 for m in metrics_to_plot]
    kc_sess = [results["KiCad"]["session_metrics"][m] * 100 for m in metrics_to_plot]

    x = np.arange(len(metric_labels))
    width = 0.2  # 柱子变窄以容纳4个

    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    rects1 = ax.bar(x - 1.5*width, fc_chunk, width, label='FreeCAD (Chunk)', color='#A0CBE8', edgecolor='black')
    rects2 = ax.bar(x - 0.5*width, fc_sess, width, label='FreeCAD (Session)', color='#4E79A7', edgecolor='black')
    rects3 = ax.bar(x + 0.5*width, kc_chunk, width, label='KiCad (Chunk)', color='#B2DF8A', edgecolor='black')
    rects4 = ax.bar(x + 1.5*width, kc_sess, width, label='KiCad (Session)', color='#33A02C', edgecolor='black')

    ax.set_ylabel('Score (%)')
    ax.set_title('Chunk-level vs Session-level Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(75, 105) 
    ax.legend(loc='lower right', ncol=2)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=0)

    for r in [rects1, rects2, rects3, rects4]:
        autolabel(r)

    fig.tight_layout()
    plt.savefig(out_dir / "7_1_dual_level_performance.png")
    plt.close()

def plot_dual_confusion_matrices(results: dict, out_dir: Path):
    """绘制 4 个混淆矩阵图 (FreeCAD Chunk/Session, KiCad Chunk/Session)"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    
    configs = [
        ("FreeCAD", "chunk_metrics", axes[0, 0], "FreeCAD - Chunk Level"),
        ("FreeCAD", "session_metrics", axes[0, 1], "FreeCAD - Session Level"),
        ("KiCad", "chunk_metrics", axes[1, 0], "KiCad - Chunk Level"),
        ("KiCad", "session_metrics", axes[1, 1], "KiCad - Session Level"),
    ]
    
    for software, metric_key, ax, title in configs:
        metrics = results[software][metric_key]
        cm = np.array([
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]]
        ])
        
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax, cbar=False, annot_kws={"size": 14})
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticklabels(['Normal', 'Abnormal'])
        ax.set_yticklabels(['Normal', 'Abnormal'])

    fig.tight_layout()
    plt.savefig(out_dir / "7_1_dual_confusion_matrices.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", type=str, default="models/section_7_1/7_1_evaluation_results.json")
    parser.add_argument("--output-dir", type=str, default="models/section_7_1")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        raise FileNotFoundError(f"找不到数据文件 {json_path}，请先运行生成脚本。")

    results = json.loads(json_path.read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("正在绘制 Chunk与Session对比柱状图 (7_1_dual_level_performance.png)...")
    plot_dual_level_performance(results, out_dir)
    
    print("正在绘制 4宫格混淆矩阵图 (7_1_dual_confusion_matrices.png)...")
    plot_dual_confusion_matrices(results, out_dir)
    
    print("绘图完成！图表已保存至", out_dir)

if __name__ == "__main__":
    main()