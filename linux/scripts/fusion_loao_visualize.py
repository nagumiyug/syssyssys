#!/usr/bin/env python3
"""
Visualisation for the Leave-One-Anomaly-Out (LOAO) experiment.

Produces three figures:
  1. loao_recall_heatmap.png   — heatmap of session-level recall per
                                 (mode × held_out_type), colour-coded by value
  2. loao_recall_bar.png       — grouped bar chart: recall per held-out type,
                                 fusion highlighted
  3. loao_combined.png         — 1×2 panel: recall heatmap + mean-recall bar

Usage:
  python fusion_loao_visualize.py \
      --loao-csv  models/fusion_advantage/loao/freecad/loao_results.csv \
      --output-dir models/fusion_advantage/loao/freecad \
      --software  FreeCAD
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── style constants ─────────────────────────────────────────────────────────────

MODES = ["scalars", "freq", "seq", "trans", "ctx", "fusion"]

MODE_COLORS = {
    "scalars": "#9467bd",
    "freq":    "#8c564b",
    "seq":     "#2ca02c",
    "trans":   "#ff7f0e",
    "ctx":     "#17becf",
    "fusion":  "#d62728",
}


# ── Figure 1: Recall heatmap ────────────────────────────────────────────────────

def plot_recall_heatmap(df: pd.DataFrame, output_dir: str, software: str):
    # Build pivot: rows = mode (ordered), cols = held_out_type
    held_out_types = sorted(df["held_out_type"].unique().tolist())
    pivot = df.pivot(index="mode", columns="held_out_type",
                     values="session_session_recall")
    # Reindex to desired mode order
    pivot = pivot.reindex([m for m in MODES if m in pivot.index])
    pivot["mean"] = pivot[held_out_types].mean(axis=1)
    sorted_pivot  = pivot.sort_values("mean", ascending=False)
    display_cols  = held_out_types + ["mean"]
    data = sorted_pivot[display_cols].values
    row_labels = sorted_pivot.index.tolist()
    col_labels = [c.replace("_", "\n") for c in held_out_types] + ["mean\nrecall"]

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(display_cols)), 3.5))
    im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data[i, j]
            text_color = "black" if 0.35 <= val <= 0.75 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")
    # Colour bar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Session Recall", fontsize=9)

    # Highlight fusion row
    fusion_row = row_labels.index("fusion") if "fusion" in row_labels else None
    if fusion_row is not None:
        for j in range(len(col_labels)):
            ax.add_patch(plt.Rectangle((j - 0.5, fusion_row - 0.5), 1, 1,
                                       fill=False, edgecolor="#d62728",
                                       linewidth=2.0))

    ax.set_title(f"{software} — LOAO: Recall on Held-out Anomaly Type\n"
                 f"(zero-shot detection — model never trained on that type)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(output_dir, "loao_recall_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Figure 2: Grouped bar chart ─────────────────────────────────────────────────

def plot_recall_bars(df: pd.DataFrame, output_dir: str, software: str):
    held_out_types = sorted(df["held_out_type"].unique().tolist())
    modes_present  = [m for m in MODES if m in df["mode"].unique()]
    n_modes        = len(modes_present)
    n_types        = len(held_out_types)
    bar_width      = 0.8 / n_modes
    x              = np.arange(n_types)

    fig, ax = plt.subplots(figsize=(max(7, 2.5 * n_types), 4.5))
    for i, mode in enumerate(modes_present):
        sub    = df[df["mode"] == mode].set_index("held_out_type")
        vals   = [sub.loc[t, "session_session_recall"]
                  if t in sub.index else 0.0
                  for t in held_out_types]
        offset = (i - n_modes / 2 + 0.5) * bar_width
        bars   = ax.bar(x + offset, vals, width=bar_width,
                        color=MODE_COLORS.get(mode, "gray"),
                        label=mode,
                        linewidth=1.5 if mode == "fusion" else 0.6,
                        edgecolor="black" if mode == "fusion" else "none",
                        zorder=3 if mode == "fusion" else 2)
        # Label fusion bars
        if mode == "fusion":
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8,
                        color=MODE_COLORS["fusion"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in held_out_types], fontsize=10)
    ax.set_ylabel("Session-level Recall", fontsize=11)
    ax.set_ylim(0.0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, ncol=n_modes, loc="upper right")
    ax.set_title(f"{software} — LOAO: Recall on Held-out Anomaly Type\n"
                 f"(each bar = session recall when that type was never in training)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(output_dir, "loao_recall_bar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── Figure 3: Mean recall comparison (single bar per mode) ──────────────────────

def plot_mean_recall_bar(df: pd.DataFrame, output_dir: str, software: str):
    modes_present = [m for m in MODES if m in df["mode"].unique()]
    mean_recall   = (
        df.groupby("mode")["session_session_recall"]
        .mean()
        .reindex(modes_present)
    )
    colors = [MODE_COLORS.get(m, "gray") for m in modes_present]
    edge   = ["black" if m == "fusion" else "none" for m in modes_present]
    lw     = [1.8 if m == "fusion" else 0.6 for m in modes_present]

    fig, ax = plt.subplots(figsize=(7, 4.0))
    bars = ax.bar(modes_present, mean_recall.values, color=colors,
                  edgecolor=edge, linewidth=lw, zorder=2)
    for bar, v in zip(bars, mean_recall.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Mean Session Recall (across all held-out types)", fontsize=10)
    ax.set_ylim(0.0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{software} — LOAO: Mean Zero-Shot Recall per Mode",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(output_dir, "loao_mean_recall_bar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LOAO visualisation")
    parser.add_argument("--loao-csv",    required=True)
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--software",    default="Software")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.loao_csv)

    plot_recall_heatmap(df, args.output_dir, args.software)
    plot_recall_bars(df,    args.output_dir, args.software)
    plot_mean_recall_bar(df, args.output_dir, args.software)


if __name__ == "__main__":
    main()
