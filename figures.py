"""
Figure Generator
----------------
Produces all publication-quality figures for the README and notebook.
Run after experiment.py: python src/figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# --- Style ---
DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
ACCENT     = "#58a6ff"
ACCENT2    = "#f78166"
ACCENT3    = "#3fb950"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"
MUTED      = "#6e7681"

def apply_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)


def plot_roc_pr():
    """ROC and PR curves side by side."""
    roc = json.load(open(RESULTS_DIR / "roc_curve.json"))
    pr  = json.load(open(RESULTS_DIR / "pr_curve.json"))
    sweep = json.load(open(RESULTS_DIR / "threshold_sweep.json"))
    summary = json.load(open(RESULTS_DIR / "summary.json"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)

    # ROC
    ax = axes[0]
    ax.plot(roc["fpr"], roc["tpr"], color=ACCENT, linewidth=2.5, label=f'AUC-ROC = {summary["best_auc_roc"]:.4f}')
    ax.plot([0,1],[0,1], color=MUTED, linewidth=1, linestyle="--", label="Random baseline")
    ax.fill_between(roc["fpr"], roc["tpr"], alpha=0.08, color=ACCENT)
    apply_style(ax, "ROC Curve", "False Positive Rate", "True Positive Rate")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # PR
    ax = axes[1]
    ax.plot(pr["recall"], pr["precision"], color=ACCENT2, linewidth=2.5,
            label=f'AP = {summary["best_f1_injection"]:.4f}')
    ax.fill_between(pr["recall"], pr["precision"], alpha=0.08, color=ACCENT2)
    apply_style(ax, "Precision–Recall Curve", "Recall", "Precision")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout(pad=2)
    plt.savefig(FIGURES_DIR / "roc_pr_curves.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] roc_pr_curves.png saved")


def plot_threshold_sweep():
    """F1, Precision, Recall vs threshold."""
    sweep = json.load(open(RESULTS_DIR / "threshold_sweep.json"))

    thresholds = [r["threshold"] for r in sweep]
    f1s        = [r["f1_injection"] for r in sweep]
    precisions = [r["precision_injection"] for r in sweep]
    recalls    = [r["recall_injection"] for r in sweep]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)

    ax.plot(thresholds, f1s,        color=ACCENT,  marker="o", linewidth=2, label="F1 (injection)")
    ax.plot(thresholds, precisions, color=ACCENT3, marker="s", linewidth=2, label="Precision")
    ax.plot(thresholds, recalls,    color=ACCENT2, marker="^", linewidth=2, label="Recall")

    best_t = max(sweep, key=lambda x: x["f1_injection"])["threshold"]
    ax.axvline(best_t, color=MUTED, linewidth=1, linestyle="--", label=f"Best threshold ({best_t})")

    apply_style(ax, "Detection Performance vs. Decision Threshold", "Threshold", "Score")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_sweep.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] threshold_sweep.png saved")


def plot_evasion_curve():
    """Security Decay / evasion curve."""
    evasion = json.load(open(RESULTS_DIR / "evasion_curve.json"))

    dilutions = [r["dilution_ratio"] for r in evasion]
    det_rates = [r["detection_rate"] * 100 for r in evasion]
    scores    = [r["mean_anomaly_score"] * 100 for r in evasion]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)

    ax1.plot(dilutions, det_rates, color=ACCENT2, marker="o", linewidth=2.5,
             label="Detection Rate (%)")
    ax1.fill_between(dilutions, det_rates, alpha=0.08, color=ACCENT2)

    ax2 = ax1.twinx()
    ax2.plot(dilutions, scores, color=ACCENT, marker="s", linewidth=2,
             linestyle="--", label="Mean Anomaly Score (×100)")
    ax2.set_facecolor(PANEL_BG)
    ax2.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax2.yaxis.label.set_color(TEXT_COLOR)
    ax2.set_ylabel("Mean Anomaly Score (scaled)", fontsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID_COLOR)

    # Mark goldilocks zone
    ax1.axvspan(0.3, 0.6, alpha=0.07, color=ACCENT3,
                label='"Goldilocks Zone" (moderate dilution)')

    apply_style(ax1, "Security Decay Curve: Detection Rate vs. Attack Dilution",
                "Benign Content Dilution Ratio", "Detection Rate (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "evasion_curve.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] evasion_curve.png saved")


def plot_feature_importance():
    """Top discriminative features bar chart."""
    importance = json.load(open(RESULTS_DIR / "feature_importance.json"))[:12]

    features  = [r["feature"].replace("_", " ") for r in importance]
    benign    = [r["benign_mean"] for r in importance]
    injection = [r["injection_mean"] for r in importance]

    x = np.arange(len(features))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK_BG)

    ax.barh(x + width/2, injection, width, label="Injection", color=ACCENT2, alpha=0.85)
    ax.barh(x - width/2, benign,    width, label="Benign",    color=ACCENT3, alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(features, fontsize=8)
    apply_style(ax, "Feature Distributions: Injection vs. Benign Prompts",
                "Mean Feature Value", "")
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] feature_importance.png saved")


def plot_category_breakdown():
    """Detection rate by injection category."""
    cats = json.load(open(RESULTS_DIR / "category_breakdown.json"))

    labels = [r["category"].replace("_", " ").title() for r in cats]
    rates  = [r["detection_rate"] * 100 for r in cats]

    colors = [ACCENT2 if r < 70 else ACCENT3 if r >= 85 else ACCENT for r in rates]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(DARK_BG)

    bars = ax.bar(labels, rates, color=colors, edgecolor=DARK_BG, linewidth=0.5)
    ax.axhline(np.mean(rates), color=MUTED, linewidth=1.5, linestyle="--",
               label=f"Mean: {np.mean(rates):.1f}%")

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.0f}%", ha="center", va="bottom",
                fontsize=9, color=TEXT_COLOR)

    apply_style(ax, "Detection Rate by Injection Category", "", "Detection Rate (%)")
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "category_breakdown.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print("[Figure] category_breakdown.png saved")


if __name__ == "__main__":
    plot_roc_pr()
    plot_threshold_sweep()
    plot_evasion_curve()
    plot_feature_importance()
    plot_category_breakdown()
    print("\n[Done] All figures saved to /figures")
