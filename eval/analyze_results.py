"""
eval/analyze_results.py
-----------------------
Merges human rankings, VLM faithfulness, and quality scores into a
unified analysis report.

Outputs:
  results/human_win_rates.csv          -- human vote totals (overall)
  results/human_win_rates_by_cat.csv   -- human votes broken down by category
  results/combined_head_to_head.csv    -- per-prompt merged view of all signals
  results/analysis_report.txt          -- plain-text summary

Usage:
    python -m eval.analyze_results
    python -m eval.analyze_results --annotator "Alice"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RANKINGS_CSV = Path("results/human_rankings.csv")
FAITH_CSV    = Path("results/faithfulness_scores.csv")
QUALITY_CSV  = Path("results/quality_scores.csv")
OUT_DIR      = Path("results")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in (RANKINGS_CSV, FAITH_CSV, QUALITY_CSV):
        if not p.exists():
            raise SystemExit(f"Required file not found: {p}")
    return (
        pd.read_csv(RANKINGS_CSV),
        pd.read_csv(FAITH_CSV),
        pd.read_csv(QUALITY_CSV),
    )


# ---------------------------------------------------------------------------
# Human rankings
# ---------------------------------------------------------------------------

def human_summary(
    h: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        h["winner"].value_counts()
        .rename_axis("winner")
        .reset_index(name="votes")
    )
    counts["pct"] = (counts["votes"] / len(h) * 100).round(1)

    cat = (
        h.groupby(["category", "winner"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("gemini", "chatgpt", "tie", "neither"):
        if col not in cat.columns:
            cat[col] = 0
    cat["total"]    = cat[["gemini", "chatgpt", "tie", "neither"]].sum(axis=1)
    cat["decisive"] = cat["gemini"] + cat["chatgpt"]
    cat["gemini_rate"]  = (cat["gemini"]  / cat["decisive"].clip(lower=1) * 100).round(1)
    cat["chatgpt_rate"] = (cat["chatgpt"] / cat["decisive"].clip(lower=1) * 100).round(1)

    return counts, cat


# ---------------------------------------------------------------------------
# VLM summaries
# ---------------------------------------------------------------------------

def vlm_by_model_and_cat(
    df: pd.DataFrame, score_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_model = (
        df.groupby("provider")[score_col]
        .agg(n="count", mean="mean", std="std")
        .reset_index()
    )
    by_model["mean_pct"] = (by_model["mean"] * 100).round(1)
    by_model["std_pct"]  = (by_model["std"]  * 100).round(1)

    by_cat = (
        df.groupby(["provider", "category"])[score_col]
        .mean()
        .reset_index(name="mean")
    )
    by_cat["mean_pct"] = (by_cat["mean"] * 100).round(1)
    pivot = (
        by_cat.pivot(index="category", columns="provider", values="mean_pct")
        .reset_index()
    )
    pivot.columns.name = None
    if "gemini" in pivot.columns and "chatgpt" in pivot.columns:
        pivot["delta_gemini_minus_chatgpt"] = (
            pivot["gemini"] - pivot["chatgpt"]
        ).round(1)

    return by_model, pivot


# ---------------------------------------------------------------------------
# Agreement: human vs VLM predicted winner
# ---------------------------------------------------------------------------

def agreement_analysis(
    h: pd.DataFrame,
    f: pd.DataFrame,
    q: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Wide faithfulness (one row per prompt × sample)
    f_wide = (
        f.pivot_table(
            index=["prompt_id", "sample"],
            columns="provider",
            values="faithfulness_score",
        )
        .reset_index()
    )
    f_wide.columns.name = None
    f_wide = f_wide.rename(
        columns={"gemini": "faith_gemini", "chatgpt": "faith_chatgpt"}
    )

    # Wide quality
    q_wide = (
        q.pivot_table(
            index=["prompt_id", "sample"],
            columns="provider",
            values="quality_score",
        )
        .reset_index()
    )
    q_wide.columns.name = None
    q_wide = q_wide.rename(
        columns={"gemini": "qual_gemini", "chatgpt": "qual_chatgpt"}
    )

    # Deduplicate human votes: keep last vote per prompt+sample
    h_dedup = (
        h.sort_values("timestamp")
        .groupby(["prompt_id", "sample"])
        .last()
        .reset_index()
    )

    merged = (
        h_dedup
        .merge(f_wide, on=["prompt_id", "sample"], how="left")
        .merge(q_wide, on=["prompt_id", "sample"], how="left")
    )

    def vlm_winner(a, b, model_a="gemini", model_b="chatgpt"):
        if pd.isna(a) or pd.isna(b):
            return None
        if a > b:
            return model_a
        if b > a:
            return model_b
        return "tie"

    merged["faith_winner"] = merged.apply(
        lambda r: vlm_winner(r["faith_gemini"], r["faith_chatgpt"]), axis=1
    )
    merged["qual_winner"] = merged.apply(
        lambda r: vlm_winner(r["qual_gemini"], r["qual_chatgpt"]), axis=1
    )

    # Restrict agreement calculation to decisive human votes
    decisive = merged[merged["winner"].isin(["gemini", "chatgpt"])].copy()
    decisive["faith_agree"] = decisive["winner"] == decisive["faith_winner"]
    decisive["qual_agree"]  = decisive["winner"] == decisive["qual_winner"]

    return merged, decisive


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", default="", help="Filter human rankings to this annotator only")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    h, f, q = load_data()

    if args.annotator and "annotator" in h.columns:
        h = h[h["annotator"] == args.annotator].copy()
        print(f"  Filtered to annotator '{args.annotator}': {len(h)} votes")
    print(f"  Human votes:       {len(h)}")
    print(f"  Faithfulness rows: {len(f)}")
    print(f"  Quality rows:      {len(q)}")

    lines: list[str] = []

    # -------------------------------------------------------------------
    # 1. Human rankings
    # -------------------------------------------------------------------
    counts, cat = human_summary(h)

    n       = len(h)
    n_g     = int((h["winner"] == "gemini").sum())
    n_c     = int((h["winner"] == "chatgpt").sum())
    n_tie   = int((h["winner"] == "tie").sum())
    n_nei   = int((h["winner"] == "neither").sum())
    n_dec   = n_g + n_c

    lines += [
        "=" * 60,
        "HUMAN RANKINGS SUMMARY",
        "=" * 60,
        f"Total votes  : {n}",
        f"  Gemini wins: {n_g}  ({n_g/n*100:.1f}%)",
        f"  ChatGPT wins:{n_c}  ({n_c/n*100:.1f}%)",
        f"  Tie          {n_tie}  ({n_tie/n*100:.1f}%)",
        f"  Both fail:   {n_nei}  ({n_nei/n*100:.1f}%)",
    ]
    if n_dec > 0:
        lines += [
            f"Decisive (excl. tie/neither): {n_dec}",
            f"  Gemini:  {n_g}/{n_dec} = {n_g/n_dec*100:.1f}%",
            f"  ChatGPT: {n_c}/{n_dec} = {n_c/n_dec*100:.1f}%",
        ]

    lines.append("\nBy category:")
    for _, row in cat.iterrows():
        dec_str = ""
        if row["decisive"] > 0:
            dec_str = f"  -> decisive: Gemini {row['gemini_rate']:.0f}% vs ChatGPT {row['chatgpt_rate']:.0f}%"
        lines.append(
            f"  {row['category']:12s}: G={int(row['gemini'])} C={int(row['chatgpt'])} "
            f"Tie={int(row['tie'])} Neither={int(row['neither'])}{dec_str}"
        )

    counts.to_csv(OUT_DIR / "human_win_rates.csv", index=False)
    cat.to_csv(OUT_DIR / "human_win_rates_by_cat.csv", index=False)

    # -------------------------------------------------------------------
    # 2. VLM faithfulness
    # -------------------------------------------------------------------
    f_model, f_cat = vlm_by_model_and_cat(f, "faithfulness_score")

    lines += [
        "",
        "=" * 60,
        "VLM FAITHFULNESS SCORES",
        "=" * 60,
    ]
    for _, row in f_model.iterrows():
        lines.append(
            f"  {row['provider']:10s}: {row['mean_pct']:.1f}%  "
            f"(+/-{row['std_pct']:.1f}%,  n={int(row['n'])})"
        )

    lines.append("\nBy category:")
    for _, row in f_cat.iterrows():
        g = row.get("gemini", float("nan"))
        c = row.get("chatgpt", float("nan"))
        d = row.get("delta_gemini_minus_chatgpt", float("nan"))
        lines.append(
            f"  {row['category']:12s}: Gemini {g:.1f}% | ChatGPT {c:.1f}% | delta {d:+.1f}%"
        )

    # -------------------------------------------------------------------
    # 3. VLM quality
    # -------------------------------------------------------------------
    q_model, q_cat = vlm_by_model_and_cat(q, "quality_score")

    lines += [
        "",
        "=" * 60,
        "VLM QUALITY SCORES",
        "=" * 60,
    ]
    for _, row in q_model.iterrows():
        lines.append(
            f"  {row['provider']:10s}: {row['mean_pct']:.1f}%  "
            f"(+/-{row['std_pct']:.1f}%,  n={int(row['n'])})"
        )

    lines.append("\nBy category:")
    for _, row in q_cat.iterrows():
        g = row.get("gemini", float("nan"))
        c = row.get("chatgpt", float("nan"))
        d = row.get("delta_gemini_minus_chatgpt", float("nan"))
        lines.append(
            f"  {row['category']:12s}: Gemini {g:.1f}% | ChatGPT {c:.1f}% | delta {d:+.1f}%"
        )

    # -------------------------------------------------------------------
    # 4. Agreement: human vs VLM
    # -------------------------------------------------------------------
    merged, decisive = agreement_analysis(h, f, q)
    merged.to_csv(OUT_DIR / "combined_head_to_head.csv", index=False)

    lines += [
        "",
        "=" * 60,
        "HUMAN vs VLM AGREEMENT  (decisive votes only)",
        "=" * 60,
    ]
    if len(decisive) > 0:
        fa = decisive["faith_agree"].mean() * 100
        qa = decisive["qual_agree"].mean()  * 100
        lines += [
            f"  Faithfulness agreement: {fa:.1f}%  "
            f"({int(decisive['faith_agree'].sum())}/{len(decisive)})",
            f"  Quality agreement:      {qa:.1f}%  "
            f"({int(decisive['qual_agree'].sum())}/{len(decisive)})",
            "",
            "By category:",
        ]
        for cat_name, grp in decisive.groupby("category"):
            fa_c = grp["faith_agree"].mean() * 100
            qa_c = grp["qual_agree"].mean()  * 100
            lines.append(
                f"  {cat_name:12s}: faith {fa_c:.0f}%  |  quality {qa_c:.0f}%  "
                f"(n={len(grp)})"
            )
    else:
        lines.append("  No decisive votes found.")

    # -------------------------------------------------------------------
    # Print + save report
    # -------------------------------------------------------------------
    report = "\n".join(lines)
    print("\n" + report)

    report_path = OUT_DIR / "analysis_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print("\nWrote:")
    for name in (
        "human_win_rates.csv",
        "human_win_rates_by_cat.csv",
        "combined_head_to_head.csv",
        "analysis_report.txt",
    ):
        print(f"  results/{name}")


if __name__ == "__main__":
    main()
