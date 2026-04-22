#!/usr/bin/env python3
"""
Visualization for Section 十三 — Feature Necessity Analysis.

Generates:
  gini_importance_bar.png          — RF Gini importance grouped bar chart
                                     (FreeCAD + KiCad side-by-side)
  permutation_importance_heatmap.png — per-class permutation importance heatmap
                                       (FreeCAD + KiCad panels)
  loo_noise_comparison.png         — Leave-one-out noise robustness line chart
                                     (FreeCAD + KiCad panels)
  ensemble_loo_blend.png           — Ensemble LOO under blend attack
                                     (FreeCAD project_scan + KiCad project_scan panels)

Usage:
  python visualize_feature_necessity.py \\
      --results-dir /path/to/tmp/s410_results/fusion_advantage \\
      --output-dir  /path/to/client_docs/plans/ablation_figures
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

GROUP_ORDER  = ["seq", "trans", "ctx", "freq", "scalars"]
GROUP_COLORS = {
    "scalars": "#e74c3c",
    "freq":    "#3498db",
    "seq":     "#2ecc71",
    "trans":   "#f39c12",
    "ctx":     "#9b59b6",
}

CONFIG_ORDER  = ["fusion_full", "fusion_minus_scalars", "fusion_minus_freq",
                 "fusion_minus_seq", "fusion_minus_trans", "fusion_minus_ctx"]
CONFIG_COLORS = {
    "fusion_full":          "#2c3e50",
    "fusion_minus_scalars": "#e74c3c",
    "fusion_minus_freq":    "#3498db",
    "fusion_minus_seq":     "#2ecc71",
    "fusion_minus_trans":   "#f39c12",
    "fusion_minus_ctx":     "#9b59b6",
}
CONFIG_STYLES = {
    "fusion_full":          ("-",  "o", 2.5, 5.0),
    "fusion_minus_scalars": ("--", "s", 1.6, 4.0),
    "fusion_minus_freq":    ("--", "D", 1.6, 4.0),
    "fusion_minus_seq":     ("--", "v", 1.6, 4.0),
    "fusion_minus_trans":   ("--", "^", 1.6, 4.0),
    "fusion_minus_ctx":     ("--", "p", 1.6, 4.0),
}

ENS_ORDER  = ["ens_full", "ens_minus_scalars", "ens_minus_freq",
              "ens_minus_seq", "ens_minus_trans", "ens_minus_ctx"]
ENS_COLORS = {
    "ens_full":          "#2c3e50",
    "ens_minus_scalars": "#e74c3c",
    "ens_minus_freq":    "#3498db",
    "ens_minus_seq":     "#2ecc71",
    "ens_minus_trans":   "#f39c12",
    "ens_minus_ctx":     "#9b59b6",
}
ENS_STYLES = {
    "ens_full":          ("-",  "o", 2.8, 6.0),
    "ens_minus_scalars": ("--", "s", 1.8, 4.5),
    "ens_minus_freq":    ("--", "D", 1.8, 4.5),
    "ens_minus_seq":     ("--", "v", 1.4, 3.5),
    "ens_minus_trans":   ("--", "^", 1.4, 3.5),
    "ens_minus_ctx":     ("--", "p", 1.4, 3.5),
}

CLASS_ORDER = ["bulk_export", "normal", "project_scan", "source_copy"]

BLEND_RATIOS_VAL    = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
BLEND_RATIOS_LABELS = ["0%", "5%", "10%", "20%", "30%", "50%", "70%", "100%"]


# ── Figure 1: Gini importance grouped bar ─────────────────────────────────────

def plot_gini_importance(fc_csv: str, kc_csv: str, out_path: str):
    fc = pd.read_csv(fc_csv).set_index("group")
    kc = pd.read_csv(kc_csv).set_index("group")
    fc = fc.reindex(GROUP_ORDER)
    kc = kc.reindex(GROUP_ORDER)

    x = np.arange(len(GROUP_ORDER))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    fc_pct = fc["pct_of_total"].values * 100
    kc_pct = kc["pct_of_total"].values * 100
    colors = [GROUP_COLORS[g] for g in GROUP_ORDER]

    bars_fc = ax.bar(x - width / 2, fc_pct, width,
                     color=colors, alpha=0.95, edgecolor="white",
                     label="FreeCAD")
    bars_kc = ax.bar(x + width / 2, kc_pct, width,
                     color=colors, alpha=0.55, edgecolor="white",
                     hatch="//", label="KiCad")

    for bar, val in zip(bars_fc, fc_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0,
                f"{val:.1f}%", ha="center", fontsize=8)
    for bar, val in zip(bars_kc, kc_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.0,
                f"{val:.1f}%", ha="center", fontsize=8)

    ax.set_title("RF Gini 特征重要性分组占比（FreeCAD vs KiCad）", fontsize=12)
    ax.set_xlabel("特征组")
    ax.set_ylabel("总 Gini 重要性占比 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, fontsize=10)
    ax.set_ylim(0, max(max(fc_pct), max(kc_pct)) + 10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#888", alpha=0.95, label="FreeCAD"),
        Patch(facecolor="#888", alpha=0.55, hatch="//", label="KiCad"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 2: Permutation importance heatmap (FreeCAD + KiCad) ────────────────

def plot_permutation_heatmap(fc_csv: str, kc_csv: str, out_path: str):
    fc = pd.read_csv(fc_csv)
    kc = pd.read_csv(kc_csv)

    def to_matrix(df):
        pivot = df.pivot(index="group", columns="class", values="mean_acc_drop")
        pivot = pivot.reindex(GROUP_ORDER)
        pivot = pivot[CLASS_ORDER]
        return pivot

    fc_mat = to_matrix(fc)
    kc_mat = to_matrix(kc)

    vmax = max(fc_mat.abs().values.max(), kc_mat.abs().values.max())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, mat, title in [(axes[0], fc_mat, "FreeCAD"),
                            (axes[1], kc_mat, "KiCad")]:
        im = ax.imshow(mat.values, cmap="RdBu_r", aspect="auto",
                       vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(CLASS_ORDER)))
        ax.set_xticklabels([c.replace("_", " ") for c in CLASS_ORDER],
                           rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(GROUP_ORDER)))
        ax.set_yticklabels(GROUP_ORDER, fontsize=10)
        ax.set_title(f"{title} — Permutation 准确率下降", fontsize=11)
        # annotate
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.values[i, j]
                color = "white" if abs(val) > vmax * 0.45 else "black"
                ax.text(j, i, f"{val:+.3f}",
                        ha="center", va="center",
                        color=color, fontsize=8)
        ax.set_xlabel("类别")
        if title == "FreeCAD":
            ax.set_ylabel("特征组")

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04,
                        label="准确率下降（越大 = 越必要）")
    fig.suptitle("Per-class Permutation 重要性：移除某组特征后各类识别率下降",
                 fontsize=12, y=1.02)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 3: LOO noise robustness comparison ────────────────────────────────

def plot_loo_noise(summary_csv: str, out_path: str):
    df = pd.read_csv(summary_csv)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)

    for ax, sw in zip(axes, ["freecad", "kicad"]):
        sub = df[df["software"] == sw]
        for config in CONFIG_ORDER:
            row = sub[sub["config"] == config].sort_values("noise_level")
            if row.empty:
                continue
            ls, mk, lw, ms = CONFIG_STYLES[config]
            label = config
            if config == "fusion_minus_ctx":
                label = "fusion − ctx（关键）"
            elif config == "fusion_full":
                label = "fusion_full（基准）"
            ax.plot(row["noise_level"].values * 100,
                    row["acc_mean"].values,
                    color=CONFIG_COLORS[config],
                    linestyle=ls, marker=mk,
                    linewidth=lw, markersize=ms,
                    label=label,
                    zorder=4 if config in ("fusion_full", "fusion_minus_ctx")
                    else 2)
        ax.set_title(f"{sw.upper()} 噪声鲁棒性", fontsize=11)
        ax.set_xlabel("噪声注入比例 (%)")
        ax.set_xticks([0, 5, 10, 30])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    axes[0].set_ylabel("session 准确率")
    fig.suptitle("Leave-one-out 消融：移除一组特征后的噪声鲁棒性",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Figure 4: Ensemble LOO under blend attack ────────────────────────────────

def plot_ensemble_loo_blend(fc_csv: str, kc_csv: str, out_path: str):
    fc = pd.read_csv(fc_csv)
    kc = pd.read_csv(kc_csv)

    # FreeCAD panel: project_scan (strong signal for scalars necessity)
    # KiCad panel:  project_scan (strong signal for freq necessity)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)

    panels = [
        (axes[0], fc, "project_scan", "FreeCAD — project_scan",
         "移除 scalars 后崩溃"),
        (axes[1], kc, "project_scan", "KiCad — project_scan",
         "移除 freq 后崩溃"),
    ]

    for ax, df, attack_type, title, annot in panels:
        sub = df[df["attack_type"] == attack_type]
        for variant in ENS_ORDER:
            row = sub[sub["variant"] == variant].sort_values("blend_ratio")
            if row.empty:
                continue
            ls, mk, lw, ms = ENS_STYLES[variant]
            label = variant
            if variant == "ens_full":
                label = "ens_full（基准）"
            elif variant == "ens_minus_scalars" and "FreeCAD" in title:
                label = "ens − scalars（关键）"
            elif variant == "ens_minus_freq" and "KiCad" in title:
                label = "ens − freq（关键）"
            ax.plot(row["blend_ratio"].values,
                    row["dr_mean"].values,
                    color=ENS_COLORS[variant],
                    linestyle=ls, marker=mk,
                    linewidth=lw, markersize=ms,
                    label=label,
                    zorder=4 if variant in ("ens_full",
                                            "ens_minus_scalars",
                                            "ens_minus_freq") else 2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("攻击浓度 (blend ratio)")
        ax.set_xticks(BLEND_RATIOS_VAL)
        ax.set_xticklabels(BLEND_RATIOS_LABELS, rotation=45, ha="right",
                           fontsize=8)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(-0.05, 1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=1)
        ax.annotate(annot, xy=(0.98, 0.05), xycoords="axes fraction",
                    ha="right", va="bottom",
                    fontsize=8.5, color="#c0392b",
                    fontweight="bold")

    axes[0].set_ylabel("检测率 (detection rate)")
    fig.suptitle("集成 LOO 混合攻击：移除某模式后检测率变化", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                        help="Root of fusion_advantage results directory")
    parser.add_argument("--output-dir",  required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    r = args.results_dir
    o = args.output_dir

    print("Generating Section 十三 figures …")

    # 1 — Gini
    plot_gini_importance(
        f"{r}/importance/freecad/gini_importance_grouped.csv",
        f"{r}/importance/kicad/gini_importance_grouped.csv",
        f"{o}/gini_importance_bar.png",
    )

    # 2 — Permutation heatmap
    plot_permutation_heatmap(
        f"{r}/importance/freecad/permutation_importance_per_class.csv",
        f"{r}/importance/kicad/permutation_importance_per_class.csv",
        f"{o}/permutation_importance_heatmap.png",
    )

    # 3 — LOO noise
    plot_loo_noise(
        f"{r}/leave_one_out/leave_one_out_noise_summary.csv",
        f"{o}/loo_noise_comparison.png",
    )

    # 4 — Ensemble LOO blend
    plot_ensemble_loo_blend(
        f"{r}/ensemble_loo/freecad/ensemble_loo_blend_summary.csv",
        f"{r}/ensemble_loo/kicad/ensemble_loo_blend_summary.csv",
        f"{o}/ensemble_loo_blend.png",
    )

    print("Done.")


if __name__ == "__main__":
    main()
