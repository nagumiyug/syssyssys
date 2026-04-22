#!/usr/bin/env python3
"""
Visualize results for the three multi-class fusion advantage experiments:
  1. Confidence study       — bar chart of mean_correct_proba / macro_auc_roc
  2. Multi-class efficiency — accuracy vs training fraction (line chart)
  3. Multi-class noise      — accuracy vs noise level (line chart)

Inputs (one directory per software):
  confidence/freecad/confidence_results.csv
  confidence/kicad/confidence_results.csv
  multiclass_efficiency/freecad/multiclass_efficiency_summary.csv
  multiclass_efficiency/kicad/multiclass_efficiency_summary.csv
  multiclass_noise/freecad/multiclass_noise_summary.csv
  multiclass_noise/kicad/multiclass_noise_summary.csv

Outputs (saved to --output-dir):
  mc_confidence_bar.png
  mc_efficiency_line.png
  mc_noise_line.png
  mc_combined_summary.png   — 4-panel composite

Usage:
  python fusion_multiclass_visualize.py \
      --results-root models/fusion_advantage \
      --output-dir   models/fusion_advantage/multiclass_figures
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── visual style ──────────────────────────────────────────────────────────────

MODES        = ["scalars", "freq", "seq", "trans", "ctx", "fusion"]
SOFTWARES    = ["freecad", "kicad"]
SW_LABELS    = {"freecad": "FreeCAD", "kicad": "KiCad"}

# fusion = red solid; single modes = distinct muted colours, dashed
PALETTE = {
    "scalars": "#4e79a7",
    "freq":    "#f28e2b",
    "seq":     "#59a14f",
    "trans":   "#76b7b2",
    "ctx":     "#b07aa1",
    "fusion":  "#e15759",
}
LW     = {"fusion": 2.6, "default": 1.5}
DASHES = {"fusion": (None, None), "default": (5, 3)}

plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi":    150,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"Missing results file: {path}")
    return pd.read_csv(path)


def mode_line_kwargs(mode: str) -> dict:
    color = PALETTE[mode]
    lw    = LW["fusion"] if mode == "fusion" else LW["default"]
    dash  = DASHES["fusion"] if mode == "fusion" else DASHES["default"]
    kw    = dict(color=color, linewidth=lw, marker="o", markersize=4,
                 label=mode, zorder=5 if mode == "fusion" else 2)
    if dash[0] is not None:
        kw["dashes"] = dash
    return kw


def finish_ax(ax, title, xlabel, ylabel, ylim=(0, 1.05)):
    ax.set_title(title, pad=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.35, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)


# ── 1. confidence bar chart ───────────────────────────────────────────────────

def plot_confidence(results_root: str, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    fig.suptitle("Chunk-level Prediction Confidence — Fusion vs Single Modes",
                 fontsize=12, y=1.01)

    for ax, sw in zip(axes, SOFTWARES):
        csv = os.path.join(results_root, "confidence", sw, "confidence_results.csv")
        df  = load(csv)
        df  = df[df["mode"].isin(MODES)].copy()
        df["mode"] = pd.Categorical(df["mode"], categories=MODES, ordered=True)
        df = df.sort_values("mode")

        x       = np.arange(len(MODES))
        width   = 0.35
        colors  = [PALETTE[m] for m in df["mode"].tolist()]
        edgecol = ["#c0392b" if m == "fusion" else "none" for m in df["mode"].tolist()]

        b1 = ax.bar(x - width/2, df["mean_correct_proba"], width,
                    color=colors, edgecolor=edgecol, linewidth=1.2,
                    label="mean correct proba", alpha=0.85)
        b2 = ax.bar(x + width/2, df["macro_auc_roc"], width,
                    color=colors, edgecolor=edgecol, linewidth=1.2,
                    alpha=0.50, hatch="//", label="macro AUC-ROC")

        ax.set_xticks(x)
        ax.set_xticklabels(df["mode"].tolist(), rotation=30, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_title(SW_LABELS[sw])
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.35, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

        # annotate fusion bar
        fusion_row = df[df["mode"] == "fusion"]
        if not fusion_row.empty:
            fi = df["mode"].tolist().index("fusion")
            ax.annotate("fusion", xy=(x[fi], 1.02), ha="center",
                        fontsize=8, color="#c0392b", fontweight="bold")

    # legend once
    legend_patches = [
        mpatches.Patch(facecolor="grey", alpha=0.85, label="mean correct proba"),
        mpatches.Patch(facecolor="grey", alpha=0.50, hatch="//", label="macro AUC-ROC"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06), frameon=False)

    fig.tight_layout()
    out = os.path.join(out_dir, "mc_confidence_bar.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── 2. efficiency line chart ──────────────────────────────────────────────────

def plot_efficiency(results_root: str, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    fig.suptitle("4-class Session Accuracy vs Training Fraction",
                 fontsize=12, y=1.01)

    for ax, sw in zip(axes, SOFTWARES):
        csv = os.path.join(results_root, "multiclass_efficiency", sw,
                           "multiclass_efficiency_summary.csv")
        df  = load(csv)

        for mode in MODES:
            sub = df[df["mode"] == mode].sort_values("train_fraction")
            if sub.empty:
                continue
            x   = sub["train_fraction"].values * 100
            y   = sub["acc_mean"].values
            err = sub["acc_std"].values
            kw  = mode_line_kwargs(mode)
            ax.plot(x, y, **kw)
            ax.fill_between(x, y - err, y + err,
                            color=PALETTE[mode], alpha=0.10)

        ax.set_xticks([10, 20, 30, 50, 70, 100])
        ax.set_xticklabels(["10%", "20%", "30%", "50%", "70%", "100%"])
        finish_ax(ax, SW_LABELS[sw], "Training fraction", "Session accuracy")

    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out = os.path.join(out_dir, "mc_efficiency_line.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── 3. noise robustness line chart ────────────────────────────────────────────

def plot_noise(results_root: str, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    fig.suptitle("4-class Session Accuracy vs Syscall Noise Level",
                 fontsize=12, y=1.01)

    for ax, sw in zip(axes, SOFTWARES):
        csv = os.path.join(results_root, "multiclass_noise", sw,
                           "multiclass_noise_summary.csv")
        df  = load(csv)

        for mode in MODES:
            sub = df[df["mode"] == mode].sort_values("noise_level")
            if sub.empty:
                continue
            x   = sub["noise_level"].values * 100
            y   = sub["acc_mean"].values
            err = sub["acc_std"].values
            kw  = mode_line_kwargs(mode)
            ax.plot(x, y, **kw)
            ax.fill_between(x, y - err, y + err,
                            color=PALETTE[mode], alpha=0.10)

        ax.set_xticks([0, 5, 10, 20, 30, 50])
        ax.set_xticklabels(["0%", "5%", "10%", "20%", "30%", "50%"])
        finish_ax(ax, SW_LABELS[sw], "Noise level (% syscalls replaced)", "Session accuracy")

    axes[0].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    out = os.path.join(out_dir, "mc_noise_line.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── 4. combined 4-panel summary ───────────────────────────────────────────────

def plot_combined(results_root: str, out_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Multi-class Fusion Advantage — Combined Summary",
                 fontsize=13, y=1.01)

    # top row: efficiency (freecad left, kicad right)
    for col, sw in enumerate(SOFTWARES):
        ax  = axes[0, col]
        csv = os.path.join(results_root, "multiclass_efficiency", sw,
                           "multiclass_efficiency_summary.csv")
        df  = load(csv)
        for mode in MODES:
            sub = df[df["mode"] == mode].sort_values("train_fraction")
            if sub.empty:
                continue
            x = sub["train_fraction"].values * 100
            y = sub["acc_mean"].values
            ax.plot(x, y, **mode_line_kwargs(mode))
        ax.set_xticks([10, 20, 30, 50, 70, 100])
        ax.set_xticklabels(["10%","20%","30%","50%","70%","100%"], fontsize=8)
        finish_ax(ax, f"{SW_LABELS[sw]} — Efficiency",
                  "Training fraction", "4-class session acc")
        if col == 0:
            ax.legend(loc="lower right", frameon=False, fontsize=7)

    # bottom row: noise (freecad left, kicad right)
    for col, sw in enumerate(SOFTWARES):
        ax  = axes[1, col]
        csv = os.path.join(results_root, "multiclass_noise", sw,
                           "multiclass_noise_summary.csv")
        df  = load(csv)
        for mode in MODES:
            sub = df[df["mode"] == mode].sort_values("noise_level")
            if sub.empty:
                continue
            x = sub["noise_level"].values * 100
            y = sub["acc_mean"].values
            ax.plot(x, y, **mode_line_kwargs(mode))
        ax.set_xticks([0, 5, 10, 20, 30, 50])
        ax.set_xticklabels(["0%","5%","10%","20%","30%","50%"], fontsize=8)
        finish_ax(ax, f"{SW_LABELS[sw]} — Noise Robustness",
                  "Noise level", "4-class session acc")
        if col == 0:
            ax.legend(loc="lower left", frameon=False, fontsize=7)

    fig.tight_layout()
    out = os.path.join(out_dir, "mc_combined_summary.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-class fusion advantage visualizer")
    parser.add_argument("--results-root", required=True,
                        help="Root dir containing confidence/, multiclass_efficiency/, "
                             "multiclass_noise/ subdirectories")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_confidence(args.results_root, args.output_dir)
    plot_efficiency(args.results_root, args.output_dir)
    plot_noise(args.results_root, args.output_dir)
    plot_combined(args.results_root, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
