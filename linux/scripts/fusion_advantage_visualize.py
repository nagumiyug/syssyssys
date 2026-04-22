#!/usr/bin/env python3
"""
Visualisation for the three fusion advantage experiments:
  1. Early detection  — accuracy vs K (first K chunks)
  2. Data efficiency  — accuracy vs training fraction (with error bands)
  3. Noise robustness — accuracy vs noise level (with error bands)

Each experiment produces one figure per software (stored in the output dir
passed on the command line).

Usage:
  python fusion_advantage_visualize.py \
      --early-csv   models/fusion_advantage/early_detection/freecad/early_detection_results.csv \
      --efficiency-csv models/fusion_advantage/data_efficiency/freecad/data_efficiency_summary.csv \
      --noise-csv   models/fusion_advantage/noise_robustness/freecad/noise_robustness_summary.csv \
      --output-dir  models/fusion_advantage/freecad \
      --software    FreeCAD
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── style constants ────────────────────────────────────────────────────────────

MODES = ["scalars", "freq", "seq", "trans", "ctx", "fusion"]

MODE_COLORS = {
    "scalars": "#9467bd",
    "freq":    "#8c564b",
    "seq":     "#2ca02c",
    "trans":   "#ff7f0e",
    "ctx":     "#17becf",
    "fusion":  "#d62728",    # fusion in red — stands out
}
MODE_STYLES = {
    "scalars": (1.2, "--"),
    "freq":    (1.2, "--"),
    "seq":     (1.2, "--"),
    "trans":   (1.2, "--"),
    "ctx":     (1.2, "--"),
    "fusion":  (2.8, "-"),   # fusion thicker and solid
}
CHUNK_SIZES = [200, 500, 1000]


def _mode_line_kwargs(mode: str) -> dict:
    lw, ls = MODE_STYLES[mode]
    return dict(color=MODE_COLORS[mode], linewidth=lw, linestyle=ls,
                marker="o", markersize=4 if mode == "fusion" else 3,
                label=mode, zorder=3 if mode == "fusion" else 2)


# ── Figure 1: Early detection ──────────────────────────────────────────────────

def plot_early_detection(csv_path: str, output_dir: str, software: str):
    df = pd.read_csv(csv_path)

    chunk_sizes_present = sorted(df["chunk_size"].unique())
    n_cols = len(chunk_sizes_present)

    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, cs in zip(axes, chunk_sizes_present):
        sub = df[df["chunk_size"] == cs]
        for mode in MODES:
            msub = sub[sub["mode"] == mode].sort_values("K")
            if msub.empty:
                continue
            ax.plot(msub["K"], msub["session_accuracy"], **_mode_line_kwargs(mode))

        ax.set_xlabel("Number of chunks observed (K)", fontsize=10)
        ax.set_title(f"chunk_size = {cs}", fontsize=11, fontweight="bold")
        ax.set_ylim(0.4, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Session-level Accuracy", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODES),
               fontsize=9, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle(f"{software} — Early Detection: accuracy using first K chunks",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(output_dir, "early_detection.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

    # Supplementary: detection_rate (anomalous sessions correctly flagged)
    if "detection_rate" in df.columns:
        fig2, axes2 = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.5), sharey=True)
        if n_cols == 1:
            axes2 = [axes2]
        for ax, cs in zip(axes2, chunk_sizes_present):
            sub = df[df["chunk_size"] == cs]
            for mode in MODES:
                msub = sub[sub["mode"] == mode].sort_values("K")
                if msub.empty:
                    continue
                ax.plot(msub["K"], msub["detection_rate"], **_mode_line_kwargs(mode))
            ax.set_xlabel("Number of chunks observed (K)", fontsize=10)
            ax.set_title(f"chunk_size = {cs}", fontsize=11, fontweight="bold")
            ax.set_ylim(0.0, 1.05)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8)
            ax.grid(axis="y", alpha=0.3)
        axes2[0].set_ylabel("Detection Rate (anomalous sessions)", fontsize=10)
        handles, labels = axes2[0].get_legend_handles_labels()
        fig2.legend(handles, labels, loc="lower center", ncol=len(MODES),
                    fontsize=9, bbox_to_anchor=(0.5, -0.08), frameon=False)
        fig2.suptitle(f"{software} — Early Detection: anomaly detection rate",
                      fontsize=12, fontweight="bold", y=1.01)
        fig2.tight_layout()
        out2 = os.path.join(output_dir, "early_detection_rate.png")
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved → {out2}")


# ── Figure 2: Data efficiency ──────────────────────────────────────────────────

def plot_data_efficiency(csv_path: str, output_dir: str, software: str):
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in MODES:
        sub = df[df["mode"] == mode].sort_values("train_fraction")
        if sub.empty:
            continue
        kw  = _mode_line_kwargs(mode)
        ax.plot(sub["train_fraction"] * 100, sub["acc_mean"], **kw)
        err = sub["acc_std"].fillna(0.0)
        if err.max() > 0:
            ax.fill_between(
                sub["train_fraction"] * 100,
                sub["acc_mean"] - err,
                sub["acc_mean"] + err,
                color=MODE_COLORS[mode], alpha=0.12,
            )

    ax.set_xlabel("Training data fraction (%)", fontsize=11)
    ax.set_ylabel("Session-level Accuracy", fontsize=11)
    ax.set_title(f"{software} — Data Efficiency: accuracy vs training set size",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_ylim(0.4, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    out = os.path.join(output_dir, "data_efficiency.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


# ── Figure 3: Noise robustness ─────────────────────────────────────────────────

def plot_noise_robustness(csv_path: str, output_dir: str, software: str):
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in MODES:
        sub = df[df["mode"] == mode].sort_values("noise_level")
        if sub.empty:
            continue
        kw  = _mode_line_kwargs(mode)
        ax.plot(sub["noise_level"] * 100, sub["acc_mean"], **kw)
        err = sub["acc_std"].fillna(0.0)
        if err.max() > 0:
            ax.fill_between(
                sub["noise_level"] * 100,
                sub["acc_mean"] - err,
                sub["acc_mean"] + err,
                color=MODE_COLORS[mode], alpha=0.12,
            )

    ax.set_xlabel("Noise level (% syscalls replaced)", fontsize=11)
    ax.set_ylabel("Session-level Accuracy", fontsize=11)
    ax.set_title(f"{software} — Noise Robustness: accuracy vs noise injection",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-1, 53)
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    out = os.path.join(output_dir, "noise_robustness.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


# ── Figure 4: Combined 3-panel summary ────────────────────────────────────────

def plot_combined_summary(early_csv: str, efficiency_csv: str, noise_csv: str,
                          output_dir: str, software: str):
    """
    Single figure with 3 sub-panels at chunk_size=1000 (or first available).
    Suitable for paper/report inclusion.
    """
    df_e = pd.read_csv(early_csv)
    df_d = pd.read_csv(efficiency_csv)
    df_n = pd.read_csv(noise_csv)

    # Pick chunk_size=1000 for early detection panel (or largest available)
    cs = 1000 if 1000 in df_e["chunk_size"].values else df_e["chunk_size"].max()
    df_e = df_e[df_e["chunk_size"] == cs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel A: early detection
    ax = axes[0]
    for mode in MODES:
        sub = df_e[df_e["mode"] == mode].sort_values("K")
        if sub.empty:
            continue
        ax.plot(sub["K"], sub["session_accuracy"], **_mode_line_kwargs(mode))
    ax.set_xlabel("Chunks observed (K)", fontsize=10)
    ax.set_ylabel("Session Accuracy", fontsize=10)
    ax.set_title(f"(A) Early Detection\nchunk_size={cs}", fontsize=10, fontweight="bold")
    ax.set_ylim(0.4, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # Panel B: data efficiency
    ax = axes[1]
    for mode in MODES:
        sub = df_d[df_d["mode"] == mode].sort_values("train_fraction")
        if sub.empty:
            continue
        kw = _mode_line_kwargs(mode)
        ax.plot(sub["train_fraction"] * 100, sub["acc_mean"], **kw)
        err = sub["acc_std"].fillna(0.0)
        if err.max() > 0:
            ax.fill_between(sub["train_fraction"] * 100,
                            sub["acc_mean"] - err, sub["acc_mean"] + err,
                            color=MODE_COLORS[mode], alpha=0.12)
    ax.set_xlabel("Training fraction (%)", fontsize=10)
    ax.set_title("(B) Data Efficiency", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_ylim(0.4, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # Panel C: noise robustness
    ax = axes[2]
    for mode in MODES:
        sub = df_n[df_n["mode"] == mode].sort_values("noise_level")
        if sub.empty:
            continue
        kw = _mode_line_kwargs(mode)
        ax.plot(sub["noise_level"] * 100, sub["acc_mean"], **kw)
        err = sub["acc_std"].fillna(0.0)
        if err.max() > 0:
            ax.fill_between(sub["noise_level"] * 100,
                            sub["acc_mean"] - err, sub["acc_mean"] + err,
                            color=MODE_COLORS[mode], alpha=0.12)
    ax.set_xlabel("Noise level (%)", fontsize=10)
    ax.set_title("(C) Noise Robustness", fontsize=10, fontweight="bold")
    ax.set_xlim(-1, 53)
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(MODES),
               fontsize=10, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle(f"{software} — Fusion Model Advantage: Three Perspectives",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(output_dir, "fusion_advantage_summary.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fusion advantage visualisation")
    parser.add_argument("--early-csv",      required=True)
    parser.add_argument("--efficiency-csv", required=True)
    parser.add_argument("--noise-csv",      required=True)
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--software",       default="Software")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    plot_early_detection(args.early_csv,      args.output_dir, args.software)
    plot_data_efficiency(args.efficiency_csv,  args.output_dir, args.software)
    plot_noise_robustness(args.noise_csv,      args.output_dir, args.software)
    plot_combined_summary(
        args.early_csv, args.efficiency_csv, args.noise_csv,
        args.output_dir, args.software,
    )


if __name__ == "__main__":
    main()
