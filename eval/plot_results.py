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
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

RANKINGS_CSV = Path("results/human_rankings.csv")
FAITH_CSV    = Path("results/faithfulness_scores.csv")
QUALITY_CSV  = Path("results/quality_scores.csv")
PLOTS_DIR    = Path("results/plots")

GEMINI_COLOR  = "#4285F4"   # Google blue
CHATGPT_COLOR = "#10A37F"   # OpenAI green
TIE_COLOR     = "#9E9E9E"
NEITHER_COLOR = "#E53935"

CATEGORIES = ["binding", "count", "spatial", "typography"]
CAT_LABELS  = {"binding": "Binding", "count": "Count",
               "spatial": "Spatial", "typography": "Typography"}

PROVIDER_LABELS = {"gemini": "Gemini (Imagen)", "chatgpt": "ChatGPT (GPT-Image-1-Mini)"}


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
    n = len(h)
    counts = {
        "Gemini wins":  (h["winner"] == "gemini").sum(),
        "ChatGPT wins": (h["winner"] == "chatgpt").sum(),
        "Tie":          (h["winner"] == "tie").sum(),
        "Both fail":    (h["winner"] == "neither").sum(),
    }
    colors = [GEMINI_COLOR, CHATGPT_COLOR, TIE_COLOR, NEITHER_COLOR]

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
    ax.set_title("Overall Human Preference  (n=120 votes)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 2 – decisive win rate by category (grouped bar)
# ---------------------------------------------------------------------------

def plot_human_by_category(h: pd.DataFrame) -> plt.Figure:
    cats = CATEGORIES
    gemini_rates, chatgpt_rates, n_decisive = [], [], []

    for cat in cats:
        sub = h[h["category"] == cat]
        dec = sub[sub["winner"].isin(["gemini", "chatgpt"])]
        nd  = len(dec)
        n_decisive.append(nd)
        if nd > 0:
            gemini_rates.append(  (dec["winner"] == "gemini").sum()  / nd * 100)
            chatgpt_rates.append( (dec["winner"] == "chatgpt").sum() / nd * 100)
        else:
            gemini_rates.append(0)
            chatgpt_rates.append(0)

    x   = np.arange(len(cats))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    bars_g = ax.bar(x - w/2, gemini_rates,  w, label="Gemini (Imagen)",          color=GEMINI_COLOR,  alpha=0.9)
    bars_c = ax.bar(x + w/2, chatgpt_rates, w, label="ChatGPT (GPT-Image-1-Mini)", color=CHATGPT_COLOR, alpha=0.9)

    for bar in bars_g:
        h_ = bar.get_height()
        if h_ > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h_ + 1, f"{h_:.0f}%",
                    ha="center", va="bottom", fontsize=9)
    for bar in bars_c:
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
    cats = CATEGORIES
    gem_scores, gpt_scores = [], []

    for cat in cats:
        sub = f[f["category"] == cat]
        gem_scores.append(sub[sub["provider"] == "gemini" ]["faithfulness_score"].mean() * 100)
        gpt_scores.append(sub[sub["provider"] == "chatgpt"]["faithfulness_score"].mean() * 100)

    x = np.arange(len(cats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    bars_g = ax.bar(x - w/2, gem_scores, w, label="Gemini (Imagen)",          color=GEMINI_COLOR,  alpha=0.9)
    bars_c = ax.bar(x + w/2, gpt_scores, w, label="ChatGPT (GPT-Image-1-Mini)", color=CHATGPT_COLOR, alpha=0.9)

    for bar in bars_g + bars_c:
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
    fig.suptitle("Text-to-Image Benchmark: Gemini vs ChatGPT", fontsize=15, fontweight="bold", y=1.01)

    # Panel A – overall vote distribution
    ax = axes[0, 0]
    n = len(h)
    labels = ["Gemini\nwins", "ChatGPT\nwins", "Tie", "Both\nfail"]
    vals   = [
        (h["winner"] == "gemini").sum(),
        (h["winner"] == "chatgpt").sum(),
        (h["winner"] == "tie").sum(),
        (h["winner"] == "neither").sum(),
    ]
    colors = [GEMINI_COLOR, CHATGPT_COLOR, TIE_COLOR, NEITHER_COLOR]
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
    cats = CATEGORIES
    gemini_rates, chatgpt_rates = [], []
    for cat in cats:
        sub = h[h["category"] == cat]
        dec = sub[sub["winner"].isin(["gemini", "chatgpt"])]
        nd  = len(dec)
        if nd > 0:
            gemini_rates.append( (dec["winner"] == "gemini").sum()  / nd * 100)
            chatgpt_rates.append((dec["winner"] == "chatgpt").sum() / nd * 100)
        else:
            gemini_rates.append(0); chatgpt_rates.append(0)

    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w/2, gemini_rates,  w, color=GEMINI_COLOR,  alpha=0.9, label="Gemini")
    ax.bar(x + w/2, chatgpt_rates, w, color=CHATGPT_COLOR, alpha=0.9, label="ChatGPT")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABELS[c] for c in cats], fontsize=9)
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("B. Decisive Win Rate by Category", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel C – faithfulness by category
    ax = axes[1, 0]
    gem_f, gpt_f = [], []
    for cat in cats:
        sub = f[f["category"] == cat]
        gem_f.append(sub[sub["provider"] == "gemini" ]["faithfulness_score"].mean() * 100)
        gpt_f.append(sub[sub["provider"] == "chatgpt"]["faithfulness_score"].mean() * 100)
    ax.bar(x - w/2, gem_f, w, color=GEMINI_COLOR,  alpha=0.9, label="Gemini")
    ax.bar(x + w/2, gpt_f, w, color=CHATGPT_COLOR, alpha=0.9, label="ChatGPT")
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
    args = parser.parse_args()

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
