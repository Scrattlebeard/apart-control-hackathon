"""Generate usefulness chart with error bars from RESULTS.md data."""

import numpy as np
import matplotlib.pyplot as plt

# Data from RESULTS.md
data = {
    "Haiku 4.5": {
        "Baseline": [0.875, 0.925, 0.925],
        "GOAP": [0.700, 0.700, 0.700],
    },
    "Gemini 3\nFlash": {
        "Baseline": [0.975, 0.975, 0.950],
        "GOAP": [0.875, 0.900, 0.875],
    },
    "Gemini 3.1\nFlash Lite": {
        "Baseline": [0.925, 0.900, 0.875],
        "GOAP": [0.825, 0.800, 0.800],
    },
}

models = list(data.keys())
n_models = len(models)

fig, ax = plt.subplots(figsize=(8, 5))

bar_width = 0.32
x = np.arange(n_models)

# Baseline bars
baseline_means = []
baseline_errs = []
for m in models:
    vals = data[m]["Baseline"]
    if vals:
        baseline_means.append(np.mean(vals))
        baseline_errs.append(np.std(vals, ddof=1))
    else:
        baseline_means.append(0)
        baseline_errs.append(0)

# GOAP bars
goap_means = []
goap_errs = []
goap_mask = []
for m in models:
    vals = data[m]["GOAP"]
    if vals:
        goap_means.append(np.mean(vals))
        goap_errs.append(np.std(vals, ddof=1))
        goap_mask.append(True)
    else:
        goap_means.append(0)
        goap_errs.append(0)
        goap_mask.append(False)

bars1 = ax.bar(
    x - bar_width / 2, baseline_means, bar_width,
    yerr=baseline_errs, capsize=4,
    label="Baseline (unconstrained)", color="#4A90D9", edgecolor="white",
)

# Only plot GOAP bars where data exists
goap_x = [xi for xi, has in zip(x, goap_mask) if has]
goap_m = [m for m, has in zip(goap_means, goap_mask) if has]
goap_e = [e for e, has in zip(goap_errs, goap_mask) if has]

bars2 = ax.bar(
    np.array(goap_x) + bar_width / 2, goap_m, bar_width,
    yerr=goap_e, capsize=4,
    label="GOAP (control-by-construction)", color="#E8834A", edgecolor="white",
)

# Formatting
ax.set_ylabel("Main Task Success Rate", fontsize=12)
ax.set_title("Usefulness: Baseline vs GOAP Protocol", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(0, 1.18)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(loc="upper center", ncol=2, fontsize=10, framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)

# Value labels on bars — positioned above error bars
for bar, err in zip(bars1, baseline_errs):
    if bar.get_height() > 0:
        label_y = bar.get_height() + err + 0.015
        ax.text(
            bar.get_x() + bar.get_width() / 2, label_y,
            f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9,
        )
for bar, err in zip(bars2, goap_e):
    if bar.get_height() > 0:
        label_y = bar.get_height() + err + 0.015
        ax.text(
            bar.get_x() + bar.get_width() / 2, label_y,
            f"{bar.get_height():.1%}", ha="center", va="bottom", fontsize=9,
        )

plt.tight_layout()
plt.savefig("figures/usefulness.png", dpi=150)
plt.savefig("figures/usefulness.pdf")
print("Saved to figures/usefulness.png and figures/usefulness.pdf")
