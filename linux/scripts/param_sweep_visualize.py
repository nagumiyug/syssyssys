#!/usr/bin/env python3
"""
Visualisation for parameter sweep results.

Generates two heatmaps (chunk_size × threshold):
  1. param_sweep_accuracy.png  – session-level accuracy
  2. param_sweep_fpr.png       – session-level FPR

The production default (chunk_size=1000, threshold=0.5) is highlighted
with a red border on each heatmap.

Usage:
  python param_sweep_visualize.py \
      --results-csv models/param_sweep/freecad/param_sweep_results.csv \
      --output-dir models/param_sweep/freecad \
      --software FreeCAD
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_THRESHOLD  = 0.5


def load_results(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _draw_heatmap(
    ax,
    matrix: np.ndarray,
    row_labels: list,
    col_labels: list,
    title: str,
    fmt: str,
    cmap: str,
    vmin: float,
    vmax: float,
    default_row: int,
    default_col: int,
):
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels([f"{th:.1f}" for th in col_labels], fontsize=10)
    ax.set_yticklabels([str(cs) for cs in row_labels], fontsize=10)
    ax.set_xlabel("Voting Threshold", fontsize=11)
    ax.set_ylabel("Chunk Size", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # annotate each cell; dark cells (normalised value > 0.6) get white text
    span = vmax - vmin if vmax > vmin else 1.0
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            norm_val = (val - vmin) / span
            text_color = "white" if norm_val > 0.6 else "black"
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    # highlight production default with red border
    rect = mpatches.FancyBboxPatch(
        (default_col - 0.5, default_row - 0.5), 1.0, 1.0,
        boxstyle="square,pad=0",
        linewidth=2.5, edgecolor="red", facecolor="none",
    )
    ax.add_patch(rect)

    return im


def plot_heatmap(df: pd.DataFrame, output_dir: str, software: str, metric: str):
    chunk_sizes = sorted(df["chunk_size"].unique())
    thresholds  = sorted(df["threshold"].unique())

    pivot = df.pivot(index="chunk_size", columns="threshold", values=metric)
    matrix = pivot.loc[chunk_sizes, thresholds].values

    default_row = chunk_sizes.index(DEFAULT_CHUNK_SIZE) if DEFAULT_CHUNK_SIZE in chunk_sizes else 0
    default_col = thresholds.index(DEFAULT_THRESHOLD)   if DEFAULT_THRESHOLD  in thresholds  else 0

    if metric == "session_accuracy":
        title  = f"{software} — Session-level Accuracy\n(chunk_size × voting_threshold)"
        fmt    = "{:.1%}"
        cmap   = "Blues"
        vmin, vmax = max(0.5, matrix.min() - 0.05), 1.0
        fname  = "param_sweep_accuracy.png"
    else:  # session_fpr
        title  = f"{software} — Session-level FPR\n(chunk_size × voting_threshold)"
        fmt    = "{:.1%}"
        cmap   = "Reds"
        vmin, vmax = 0.0, min(0.5, matrix.max() + 0.05)
        fname  = "param_sweep_fpr.png"

    fig, ax = plt.subplots(figsize=(7, 5))
    im = _draw_heatmap(
        ax, matrix, chunk_sizes, thresholds,
        title, fmt, cmap, vmin, vmax, default_row, default_col,
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # legend for the red box
    red_patch = mpatches.Patch(edgecolor="red", facecolor="none",
                                linewidth=2, label=f"Production default\n(size={DEFAULT_CHUNK_SIZE}, thr={DEFAULT_THRESHOLD})")
    ax.legend(handles=[red_patch], loc="upper left", fontsize=8,
              bbox_to_anchor=(0.0, -0.15), frameon=False)

    fig.tight_layout()
    out = os.path.join(output_dir, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


def main():
    parser = argparse.ArgumentParser(description="Visualise parameter sweep results")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--software",    default="Software")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_results(args.results_csv)

    plot_heatmap(df, args.output_dir, args.software, "session_accuracy")
    plot_heatmap(df, args.output_dir, args.software, "session_fpr")


if __name__ == "__main__":
    main()
