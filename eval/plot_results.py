"""
eval/plot_results.py
--------------------
Produces visualizations from the benchmark results.

Requires: results/human_rankings.csv, faithfulness_scores.csv,
          quality_scores.csv  (run eval/analyze_results.py first)

Outputs (saved to results/plots/):
  human_overall.png        -- overall human vote distribution
  human_by_category.png    -- decisive win rate per category
  faithfulness_by_cat.png  -- VLM faithfulness scores per category
  summary.png              -- 4-panel combined figure

Usage:
    python -m eval.plot_results
    python -m eval.plot_results --annotator "Alice"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from providers.registry import label as provider_label

RANKINGS_CSV = Path("results/human_rankings.csv")
FAITH_CSV    = Path("results/faithfulness_scores.csv")
QUALITY_CSV  = Path("results/quality_scores.csv")
PLOTS_DIR    = Path("results/plots")

# Color palette assigned by sorted provider index; special colors for tie/neither
PROVIDER_PALETTE = ["#4285F4", "#10A37F", "#FF5722", "#9C27B0"]
TIE_COLOR        = "#9E9E9E"
NEITHER_COLOR    = "#E53935"

CATEGORIES = ["binding", "count", "spatial", "typography"]
CAT_LABELS  = {"binding": "Binding", "count": "Count",
               "spatial": "Spatial", "typography": "Typography"}


def provider_colors(providers: list[str]) -> dict[str, str]:
    """Map provider names to palette colors (consistent by sorted index)."""
    return {p: PROVIDER_PALETTE[i % len(PROVIDER_PALETTE)] for i, p in enumerate(sorted(providers))}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(RANKINGS_CSV),
        pd.read_csv(FAITH_CSV),
        pd.read_csv(QUALITY_CSV),
    )


def save(fig: plt.Figure, name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Plot 1 – overall human vote distribution (stacked bar)
# ---------------------------------------------------------------------------

def plot_human_overall(h: pd.DataFrame) -> plt.Figure:
    n          = len(h)
    _providers = sorted(w for w in h["winner"].unique() if w not in {"tie", "neither"})
    _pcolors   = provider_colors(_providers)

    counts: dict = {}
    for p in _providers:
        counts[f"{provider_label(p)} wins"] = (h["winner"] == p).sum()
    counts["Tie"]       = (h["winner"] == "tie").sum()
    counts["Both fail"] = (h["winner"] == "neither").sum()
    colors = [_pcolors[p] for p in _providers] + [TIE_COLOR, NEITHER_COLOR]

    fig, ax = plt.subplots(figsize=(7, 4))
    left = 0
    for (label, val), color in zip(counts.items(), colors):
        pct = val / n * 100
        ax.barh(0, pct, left=left, color=color, height=0.5, label=f"{label} ({val}, {pct:.1f}%)")
        if pct > 5:
            ax.text(left + pct / 2, 0, f"{pct:.1f}%", va="center", ha="center",
                    fontsize=10, fontweight="bold", color="white")
        left += pct

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Share of votes (%)")
    ax.set_title(f"Overall Human Preference  (n={n} votes)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2 – decisive win rate by category (grouped bar)
# ---------------------------------------------------------------------------

def plot_human_by_category(h: pd.DataFrame) -> plt.Figure:
    cats       = CATEGORIES
    _providers = sorted(w for w in h["winner"].unique() if w not in {"tie", "neither"})
    _pcolors   = provider_colors(_providers)
    rates      = {p: [] for p in _providers}
    n_decisive = []

    for cat in cats:
        sub = h[h["category"] == cat]
        dec = sub[sub["winner"].isin(_providers)]
        nd  = len(dec)
        n_decisive.append(nd)
        for p in _providers:
            rates[p].append((dec["winner"] == p).sum() / nd * 100 if nd > 0 else 0)

    x   = np.arange(len(cats))
    w   = 0.8 / max(len(_providers), 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, p in enumerate(_providers):
        offset = (i - (len(_providers) - 1) / 2) * w
        bars = ax.bar(x + offset, rates[p], w, label=provider_label(p), color=_pcolors[p], alpha=0.9)
        for bar in bars:
            h_ = bar.get_height()
            if h_ > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h_ + 1, f"{h_:.0f}%",
                        ha="center", va="bottom", fontsize=9)

    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{CAT_LABELS[c]}\n(n={n_decisive[i]} decisive)" for i, c in enumerate(cats)],
        fontsize=10,
    )
    ax.set_ylabel("Win rate among decisive votes (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Decisive Human Win Rate by Category\n(ties & both-fail excluded)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3 – faithfulness scores by category (grouped bar)
# ---------------------------------------------------------------------------

def plot_faithfulness_by_cat(f: pd.DataFrame) -> plt.Figure:
    cats       = CATEGORIES
    _providers = sorted(f["provider"].unique())
    _pcolors   = provider_colors(_providers)
    scores     = {p: [] for p in _providers}

    for cat in cats:
        sub = f[f["category"] == cat]
        for p in _providers:
            scores[p].append(sub[sub["provider"] == p]["faithfulness_score"].mean() * 100)

    x  = np.arange(len(cats))
    w  = 0.8 / max(len(_providers), 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    all_bars = []
    for i, p in enumerate(_providers):
        offset = (i - (len(_providers) - 1) / 2) * w
        bars = ax.bar(x + offset, scores[p], w, label=provider_label(p), color=_pcolors[p], alpha=0.9)
        all_bars.extend(bars)

    for bar in all_bars:
        h_ = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, min(h_ + 0.5, 99),
                f"{h_:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABELS[c] for c in cats], fontsize=11)
    ax.set_ylabel("Mean faithfulness score (%)")
    ax.set_ylim(75, 105)
    ax.set_title("VLM Faithfulness Scores by Category\n(Qwen2-VL-2B judge, avg over 3 samples)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 4 – 4-panel summary figure
# ---------------------------------------------------------------------------

def plot_summary(h: pd.DataFrame, f: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Text-to-Image Benchmark Summary", fontsize=15, fontweight="bold", y=1.01)

    _providers_h = sorted(w for w in h["winner"].unique() if w not in {"tie", "neither"})
    _pcolors_h   = provider_colors(_providers_h)
    _providers_f = sorted(f["provider"].unique())
    _pcolors_f   = provider_colors(_providers_f)
    cats = CATEGORIES

    # Panel A – overall vote distribution
    ax = axes[0, 0]
    n      = len(h)
    labels = [f"{provider_label(p)}\nwins" for p in _providers_h] + ["Tie", "Both\nfail"]
    vals   = [(h["winner"] == p).sum() for p in _providers_h] + [
        (h["winner"] == "tie").sum(),
        (h["winner"] == "neither").sum(),
    ]
    colors = [_pcolors_h[p] for p in _providers_h] + [TIE_COLOR, NEITHER_COLOR]
    bars = ax.bar(labels, vals, color=colors, alpha=0.9, width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val}\n({val/n*100:.0f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title("A. Overall Human Preferences", fontweight="bold")
    ax.set_ylabel("Votes")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel B – decisive win rate by category
    ax = axes[0, 1]
    rates_b = {p: [] for p in _providers_h}
    for cat in cats:
        sub = h[h["category"] == cat]
        dec = sub[sub["winner"].isin(_providers_h)]
        nd  = len(dec)
        for p in _providers_h:
            rates_b[p].append((dec["winner"] == p).sum() / nd * 100 if nd > 0 else 0)

    x = np.arange(len(cats))
    w = 0.8 / max(len(_providers_h), 1)
    for i, p in enumerate(_providers_h):
        offset = (i - (len(_providers_h) - 1) / 2) * w
        ax.bar(x + offset, rates_b[p], w, color=_pcolors_h[p], alpha=0.9, label=provider_label(p))
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABELS[c] for c in cats], fontsize=9)
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("B. Decisive Win Rate by Category", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel C – faithfulness by category (reuses x from Panel B)
    ax = axes[1, 0]
    scores_c = {p: [] for p in _providers_f}
    for cat in cats:
        sub = f[f["category"] == cat]
        for p in _providers_f:
            scores_c[p].append(sub[sub["provider"] == p]["faithfulness_score"].mean() * 100)
    w_c = 0.8 / max(len(_providers_f), 1)
    for i, p in enumerate(_providers_f):
        offset = (i - (len(_providers_f) - 1) / 2) * w_c
        ax.bar(x + offset, scores_c[p], w_c, color=_pcolors_f[p], alpha=0.9, label=provider_label(p))
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABELS[c] for c in cats], fontsize=9)
    ax.set_ylabel("Mean faithfulness (%)")
    ax.set_ylim(75, 105)
    ax.set_title("C. VLM Faithfulness by Category", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel D – tie rate by category
    ax = axes[1, 1]
    tie_rates = []
    for cat in cats:
        sub = h[h["category"] == cat]
        tie_rates.append((sub["winner"] == "tie").sum() / len(sub) * 100)
    ax.bar([CAT_LABELS[c] for c in cats], tie_rates, color=TIE_COLOR, alpha=0.9, width=0.5)
    for i, v in enumerate(tie_rates):
        ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Tie rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("D. Tie Rate by Category", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", default="", help="Filter human rankings to this annotator only")
    parser.add_argument("--plots-dir", default="", help="Override output directory for plots")
    args = parser.parse_args()

    global PLOTS_DIR
    if args.plots_dir:
        PLOTS_DIR = Path(args.plots_dir)

    for p in (RANKINGS_CSV, FAITH_CSV, QUALITY_CSV):
        if not p.exists():
            raise SystemExit(f"Required file not found: {p}  — run eval/analyze_results.py first.")

    print("Loading data...")
    h, f, q = load()

    if args.annotator and "annotator" in h.columns:
        h = h[h["annotator"] == args.annotator].copy()
        print(f"  Filtered to annotator '{args.annotator}': {len(h)} votes")

    print("Generating plots...")
    save(plot_human_overall(h),       "human_overall.png")
    save(plot_human_by_category(h),   "human_by_category.png")
    save(plot_faithfulness_by_cat(f), "faithfulness_by_cat.png")
    save(plot_summary(h, f),          "summary.png")

    print("Done. All plots saved to results/plots/")


if __name__ == "__main__":
    main()
