"""
Generate visualizations for ablation study.

Produces per-software:
  - heatmap.png    : classifier × tier heatmap of chunk-level accuracy
  - lineplot.png   : line chart (one line per classifier, x=tier, y=accuracy)
  - delta_bars.png : side-by-side bars of Δ_features and Δ_classifiers

Usage:
  python linux/scripts/ablation_visualize.py \
      --results-csv models/ablation/freecad/ablation_results.csv \
      --output-dir  models/ablation/freecad
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIER_ORDER = ["F1", "F2", "F3", "F4", "F5"]
CLASSIFIER_ORDER = ["LogReg", "DecisionTree", "RandomForest", "GradientBoosting"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ablation study results.")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", default="chunk_accuracy",
                        help="Column name in results.csv to visualize (default: chunk_accuracy)")
    return parser.parse_args()


def plot_heatmap(matrix: pd.DataFrame, software: str, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Feature Tier")
    ax.set_ylabel("Classifier")
    ax.set_title(f"{software.upper()} — {metric} (chunk-level)")

    # Annotate each cell
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.values[i, j]
            text_color = "white" if val < 0.7 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10)

    fig.colorbar(im, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lineplot(matrix: pd.DataFrame, software: str, metric: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(matrix.columns))
    colors = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
    markers = ["o", "s", "^", "D"]

    for i, clf_name in enumerate(matrix.index):
        ax.plot(
            x, matrix.loc[clf_name].values,
            marker=markers[i % len(markers)], linewidth=2, markersize=8,
            color=colors[i % len(colors)], label=clf_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(matrix.columns)
    ax.set_xlabel("Feature Tier (F1 = minimal → F5 = full + preprocessing)")
    ax.set_ylabel(f"{metric}")
    ax.set_title(f"{software.upper()} — Classifier accuracy vs feature tier")
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta_bars(matrix: pd.DataFrame, software: str, out_path: Path) -> None:
    """Compare Δ_features (per classifier) vs Δ_classifiers (per tier)."""
    dfeat = (matrix.max(axis=1) - matrix.min(axis=1))  # per classifier
    dclf = (matrix.max(axis=0) - matrix.min(axis=0))   # per tier

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Δ_features (varying tier for each classifier)
    axes[0].bar(dfeat.index, dfeat.values, color="steelblue")
    axes[0].set_title("Δ_features\n(range of accuracy across tiers, fixed classifier)")
    axes[0].set_ylabel("Accuracy range")
    axes[0].set_ylim(0, max(dfeat.max(), dclf.max()) * 1.15)
    for i, v in enumerate(dfeat.values):
        axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    axes[0].tick_params(axis='x', rotation=15)

    # Right: Δ_classifiers (varying classifier for each tier)
    axes[1].bar(dclf.index, dclf.values, color="coral")
    axes[1].set_title("Δ_classifiers\n(range of accuracy across classifiers, fixed tier)")
    axes[1].set_ylabel("Accuracy range")
    axes[1].set_ylim(0, max(dfeat.max(), dclf.max()) * 1.15)
    for i, v in enumerate(dclf.values):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle(f"{software.upper()} — Feature vs Classifier impact on accuracy", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results_csv)
    software = str(df["software"].iloc[0])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build pivot matrix with ordered rows/columns
    matrix = df.pivot(index="classifier", columns="tier", values=args.metric)
    matrix = matrix.reindex(index=CLASSIFIER_ORDER, columns=TIER_ORDER)

    plot_heatmap(matrix, software, args.metric, output_dir / "heatmap.png")
    plot_lineplot(matrix, software, args.metric, output_dir / "lineplot.png")
    plot_delta_bars(matrix, software, output_dir / "delta_bars.png")

    print(f"Saved visualizations to {output_dir}")
    print(f"\nMatrix:")
    print(matrix.to_string(float_format="%.4f"))

    dfeat = (matrix.max(axis=1) - matrix.min(axis=1))
    dclf = (matrix.max(axis=0) - matrix.min(axis=0))
    print(f"\nDelta_features (per classifier, varying tier):")
    for k, v in dfeat.items():
        print(f"  {k:<18s}: {v:.4f}")
    print(f"\nDelta_classifiers (per tier, varying classifier):")
    for k, v in dclf.items():
        print(f"  {k:<4s}: {v:.4f}")
    print(f"\nmax_delta_features    = {dfeat.max():.4f}")
    print(f"max_delta_classifiers = {dclf.max():.4f}")
    print(f"Hypothesis confirmed  = {dfeat.max() > dclf.max()}")


if __name__ == "__main__":
    main()
