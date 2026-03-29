import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
CSV_PATH    = os.path.join(RESULTS_DIR, "baseline_comparison.csv")

# IEEE-style rcParams
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 11,
    "axes.titlesize"   : 12,
    "axes.labelsize"   : 11,
    "xtick.labelsize"  : 10,
    "ytick.labelsize"  : 10,
    "legend.fontsize"  : 10,
    "figure.dpi"       : 300,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
})

BAR_COLOR  = "#2166ac"
BAR_EDGE   = "#1a1a2e"
BAR_WIDTH  = 0.5
FIGSIZE    = (6, 4)


# =============================================================================
# Load CSV
# =============================================================================

def load_csv(path):
    """
    Returns list of dicts with keys: baseline, mean_latency_s,
    mean_energy_J, mean_sla_violation_s.  Rows with N/A values are kept
    with None so callers can skip them gracefully.
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def parse(v):
                v = v.strip()
                return float(v) if v not in ("", "N/A") else None
            rows.append({
                "baseline"     : row["baseline"].strip(),
                "latency"      : parse(row["mean_latency_s"]),
                "energy"       : parse(row["mean_energy_J"]),
                "sla"          : parse(row["mean_sla_violation_s"]),
            })
    return rows


# =============================================================================
# Shared bar-chart helper
# =============================================================================

def _bar_chart(names, values, ylabel, title, out_path, unit_fmt="{:.4f}"):
    """
    Draw a single grouped bar chart (one bar per baseline).

    Bars for N/A (None) values are omitted; remaining bars keep their
    original x-positions so the axis labels stay aligned.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x      = np.arange(len(names))
    colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(names)))

    for xi, (name, val, color) in enumerate(zip(names, values, colors)):
        if val is None:
            continue
        bar = ax.bar(xi, val, width=BAR_WIDTH, color=color,
                     edgecolor=BAR_EDGE, linewidth=0.8, zorder=3)
        # Value label on top of bar
        label = unit_fmt.format(val)
        ax.text(xi, val + 0.01 * val, label,
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# =============================================================================
# Three figures
# =============================================================================

def plot_latency(rows):
    names  = [r["baseline"] for r in rows]
    values = [r["latency"]  for r in rows]
    _bar_chart(
        names, values,
        ylabel   = "Mean Latency (s)",
        title    = "Mean Task Completion Latency — Baseline Comparison",
        out_path = os.path.join(RESULTS_DIR, "latency_comparison.png"),
        unit_fmt = "{:.4f}",
    )


def plot_energy(rows):
    names  = [r["baseline"] for r in rows]
    values = [r["energy"]   for r in rows]
    _bar_chart(
        names, values,
        ylabel   = "Mean Energy Consumption (J)",
        title    = "Mean Energy Consumption — Baseline Comparison",
        out_path = os.path.join(RESULTS_DIR, "energy_comparison.png"),
        unit_fmt = "{:.2e}",
    )


def plot_sla(rows):
    names  = [r["baseline"] for r in rows]
    values = [r["sla"]      for r in rows]
    _bar_chart(
        names, values,
        ylabel   = "Mean SLA Violation (s)",
        title    = "Mean SLA Violation — Baseline Comparison",
        out_path = os.path.join(RESULTS_DIR, "sla_comparison.png"),
        unit_fmt = "{:.4f}",
    )


# =============================================================================
# Entry point
# =============================================================================

def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        print("Run evaluate.py first to generate baseline_comparison.csv.")
        return

    rows = load_csv(CSV_PATH)
    print(f"Loaded {len(rows)} baselines from {CSV_PATH}")

    plot_latency(rows)
    plot_energy(rows)
    plot_sla(rows)

    print("Done — all figures saved to results/")


if __name__ == "__main__":
    main()
