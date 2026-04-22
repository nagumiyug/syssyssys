#!/usr/bin/env python3
"""
Visualisation for feature isolation study results.

Generates three figures per run:
  1. isolation_bar.png        – accuracy per isolation mode (chunk + session)
  2. anomaly_recall_bar.png   – per-anomaly-type recall per mode
  3. feature_importance.png   – RF feature importance by group (fusion mode)

Usage:
  python isolation_visualize.py \
      --results-csv models/isolation/freecad/isolation_results.csv \
      --summary-json models/isolation/freecad/isolation_summary.json \
      --output-dir models/isolation/freecad \
      --software FreeCAD
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

MODES = ["scalars", "freq", "seq", "trans", "ctx", "fusion"]
MODE_LABELS = {
    "scalars": "Scalars\n(entropy)",
    "freq":    "Freq\n(count)",
    "seq":     "Sequence\n(bi/trigram)",
    "trans":   "Transition\nProb",
    "ctx":     "Context\n(path type)",
    "fusion":  "Fusion\n(all)",
}

# color scheme
COLOR_CHUNK   = "#4C72B0"
COLOR_SESSION = "#DD8452"
ANOMALY_COLORS = ["#55A868", "#C44E52", "#8172B2", "#937860"]


def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["mode"] = pd.Categorical(df["mode"], categories=MODES, ordered=True)
    return df.sort_values("mode")


def load_summary(json_path: str) -> dict:
    with open(json_path) as f:
        return json.load(f)


# ── Figure 1: isolation bar chart ─────────────────────────────────────────────

def plot_isolation_bar(df: pd.DataFrame, output_dir: str, software: str):
    modes_present = [m for m in MODES if m in df["mode"].values]
    x = np.arange(len(modes_present))
    w = 0.35

    chunk_acc   = [df.loc[df["mode"] == m, "chunk_accuracy"].values[0]   for m in modes_present]
    session_acc = [df.loc[df["mode"] == m, "session_accuracy"].values[0] for m in modes_present]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w / 2, chunk_acc,   w, label="Chunk-level",   color=COLOR_CHUNK,   alpha=0.85)
    bars2 = ax.bar(x + w / 2, session_acc, w, label="Session-level", color=COLOR_SESSION, alpha=0.85)

    ax.set_xlabel("Feature Group", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(f"{software} — Accuracy per Isolated Feature Group", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes_present], fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="90% threshold")
    ax.legend(fontsize=10)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.1%}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.1%}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = os.path.join(output_dir, "isolation_bar.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


# ── Figure 2: per-anomaly recall ──────────────────────────────────────────────

def plot_anomaly_recall(df: pd.DataFrame, output_dir: str, software: str):
    recall_cols = [c for c in df.columns if c.startswith("recall_")]
    if not recall_cols:
        print("  No anomaly recall columns found — skipping anomaly_recall_bar.png")
        return

    subtypes = [c[len("recall_"):] for c in recall_cols]
    modes_present = [m for m in MODES if m in df["mode"].values]
    n_modes = len(modes_present)
    n_types = len(subtypes)
    x = np.arange(n_modes)
    w = 0.8 / n_types

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, st in enumerate(subtypes):
        col = f"recall_{st}"
        vals = [
            df.loc[df["mode"] == m, col].values[0]
            if (col in df.columns and (df["mode"] == m).any())
            else float("nan")
            for m in modes_present
        ]
        offset = (i - (n_types - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w,
                      label=st, color=ANOMALY_COLORS[i % len(ANOMALY_COLORS)], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Feature Group", fontsize=11)
    ax.set_ylabel("Recall (chunk-level)", fontsize=11)
    ax.set_title(f"{software} — Per-Anomaly-Type Recall per Feature Group", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes_present], fontsize=9)
    ax.set_ylim(0.0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(title="Anomaly Type", fontsize=9, title_fontsize=9)

    fig.tight_layout()
    out = os.path.join(output_dir, "anomaly_recall_bar.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


# ── Figure 3: feature importance by group ─────────────────────────────────────

def plot_feature_importance(summary: dict, output_dir: str, software: str):
    fi = summary.get("feature_importances", {})
    by_group = fi.get("by_group", {})
    if not by_group:
        print("  No feature importance data — skipping feature_importance.png")
        return

    groups = list(by_group.keys())
    values = [by_group[g] for g in groups]
    total  = sum(values)
    pcts   = [v / total if total > 0 else 0.0 for v in values]

    group_labels = {
        "scalars": "Scalars\n(entropy)",
        "freq":    "Freq\n(count)",
        "seq":     "Sequence\n(bi/trigram)",
        "trans":   "Transition\nProb",
        "ctx":     "Context\n(path type)",
    }
    colors = [COLOR_CHUNK, COLOR_SESSION, "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(groups))
    bars = ax.bar(x, pcts, color=colors[: len(groups)], alpha=0.85)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{pct:.1%}", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Feature Group", fontsize=11)
    ax.set_ylabel("Importance Share", fontsize=11)
    ax.set_title(f"{software} — RF Feature Importance by Group (Fusion Model)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([group_labels.get(g, g) for g in groups], fontsize=10)
    ax.set_ylim(0, max(pcts) * 1.25)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    fig.tight_layout()
    out = os.path.join(output_dir, "feature_importance.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise isolation study results")
    parser.add_argument("--results-csv",  required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-dir",   required=True)
    parser.add_argument("--software",     default="Software")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df      = load_results(args.results_csv)
    summary = load_summary(args.summary_json)

    plot_isolation_bar(df, args.output_dir, args.software)
    plot_anomaly_recall(df, args.output_dir, args.software)
    plot_feature_importance(summary, args.output_dir, args.software)


if __name__ == "__main__":
    main()
