"""
eval/aggregate_faithfulness.py
-------------------------------
Reads results/faithfulness_scores.csv (written by judge_faithfulness.py)
and produces three summary tables:

  results/faithfulness_by_model.csv          — overall per-provider stats
  results/faithfulness_by_model_category.csv — per-provider × category
  results/faithfulness_worst_cases.csv       — 20 worst-scoring images

Usage:
    python -m eval.aggregate_faithfulness
    python -m eval.aggregate_faithfulness --in_csv results/faithfulness_scores.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def verdict_breakdown(series: pd.Series) -> pd.Series:
    """Given a Series of JSON strings (check_verdicts), expand and count."""
    yes = no = unclear = total = 0
    for raw in series:
        try:
            verdicts = json.loads(raw)
        except Exception:
            continue
        for v in verdicts:
            verdict = v.get("verdict", "UNCLEAR")
            if verdict == "YES":
                yes += 1
            elif verdict == "NO":
                no += 1
            else:
                unclear += 1
            total += 1
    return pd.Series(
        {"total_checks": total, "total_yes": yes, "total_no": no, "total_unclear": unclear}
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(in_csv: str) -> None:
    in_path = Path(in_csv)
    if not in_path.exists():
        raise SystemExit(f"Input CSV not found: {in_path}\nRun judge_faithfulness.py first.")

    df = pd.read_csv(in_path)
    print(f"[INFO] Loaded {len(df)} rows from {in_path}")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Overall per-provider (model) summary
    # ------------------------------------------------------------------
    model_agg = (
        df.groupby("provider")
        .agg(
            n_images=("faithfulness_score", "count"),
            score_mean=("faithfulness_score", "mean"),
            score_std=("faithfulness_score", "std"),
            score_median=("faithfulness_score", "median"),
            checks_yes_total=("checks_yes", "sum"),
            checks_total=("checks_total", "sum"),
            parse_errors=("parse_error", "sum"),
        )
        .reset_index()
    )
    model_agg["score_pct"] = (model_agg["score_mean"] * 100).round(1)
    model_agg["yes_rate_pct"] = (
        model_agg["checks_yes_total"] / model_agg["checks_total"] * 100
    ).round(1)

    out_model = out_dir / "faithfulness_by_model.csv"
    model_agg.to_csv(out_model, index=False)

    print("\n=== Faithfulness by model ===")
    print(
        model_agg[
            ["provider", "n_images", "score_pct", "score_std", "yes_rate_pct", "parse_errors"]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 2) Per-provider × category
    # ------------------------------------------------------------------
    cat_agg = (
        df.groupby(["provider", "category"])
        .agg(
            n_images=("faithfulness_score", "count"),
            score_mean=("faithfulness_score", "mean"),
            score_std=("faithfulness_score", "std"),
            checks_yes_total=("checks_yes", "sum"),
            checks_total=("checks_total", "sum"),
        )
        .reset_index()
    )
    cat_agg["score_pct"] = (cat_agg["score_mean"] * 100).round(1)
    cat_agg["yes_rate_pct"] = (
        cat_agg["checks_yes_total"] / cat_agg["checks_total"] * 100
    ).round(1)
    cat_agg = cat_agg.sort_values(["category", "provider"])

    out_cat = out_dir / "faithfulness_by_model_category.csv"
    cat_agg.to_csv(out_cat, index=False)

    print("\n=== Faithfulness by model × category ===")
    print(
        cat_agg[["provider", "category", "n_images", "score_pct", "yes_rate_pct"]].to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # 3) Worst-20 images (lowest faithfulness_score, then most parse errors)
    # ------------------------------------------------------------------
    worst = (
        df.sort_values(["faithfulness_score", "parse_error"], ascending=[True, False])
        .head(20)[
            [
                "provider",
                "prompt_id",
                "category",
                "sample",
                "faithfulness_score",
                "checks_yes",
                "checks_no",
                "checks_unclear",
                "checks_total",
                "parse_error",
                "image_path",
            ]
        ]
        .copy()
    )
    worst["score_pct"] = (worst["faithfulness_score"] * 100).round(1)

    out_worst = out_dir / "faithfulness_worst_cases.csv"
    worst.to_csv(out_worst, index=False)

    print("\n=== Worst 20 images (by faithfulness_score) ===")
    print(
        worst[
            ["provider", "prompt_id", "category", "sample", "score_pct", "checks_yes", "checks_total"]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 4) Head-to-head: gemini vs chatgpt per prompt (mean over samples)
    # ------------------------------------------------------------------
    pivot = (
        df.groupby(["prompt_id", "category", "provider"])["faithfulness_score"]
        .mean()
        .unstack("provider")
        .reset_index()
    )
    if "gemini" in pivot.columns and "chatgpt" in pivot.columns:
        pivot["delta_gemini_minus_chatgpt"] = pivot["gemini"] - pivot["chatgpt"]
        pivot = pivot.sort_values("delta_gemini_minus_chatgpt")
        out_h2h = out_dir / "faithfulness_head_to_head.csv"
        pivot.to_csv(out_h2h, index=False)
        print(f"\n[INFO] Head-to-head written -> {out_h2h.as_posix()}")
    else:
        out_h2h = None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\nWrote:")
    print(f"  {out_model.as_posix()}")
    print(f"  {out_cat.as_posix()}")
    print(f"  {out_worst.as_posix()}")
    if out_h2h:
        print(f"  {out_h2h.as_posix()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv", default="results/faithfulness_scores.csv",
        help="CSV produced by eval/judge_faithfulness.py",
    )
    args = ap.parse_args()
    main(args.in_csv)
