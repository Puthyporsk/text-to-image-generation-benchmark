"""
eval/deep_analysis.py
---------------------
Core analytical engine for deeper benchmark analysis.

Pure-computation module: all functions accept DataFrames and return
DataFrames or dicts.  No CLI, no file I/O.

Sections:
  A. Check-level failure analysis & error taxonomy
  B. Cross-sample consistency
  C. Statistical testing  (bootstrap CIs, permutation tests)
  D. Signal disagreement  (human vs VLM)
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# A. Check-Level Failure Analysis
# ---------------------------------------------------------------------------

def expand_check_verdicts(faith_df: pd.DataFrame) -> pd.DataFrame:
    """Explode check_verdicts JSON into one row per check per image.

    Returns DataFrame with columns:
        provider, prompt_id, category, sample, check_text, verdict, reason
    """
    rows: list[dict] = []
    for _, r in faith_df.iterrows():
        try:
            verdicts = json.loads(r["check_verdicts"])
        except (json.JSONDecodeError, TypeError):
            continue
        for v in verdicts:
            rows.append({
                "provider":  r["provider"],
                "prompt_id": r["prompt_id"],
                "category":  r["category"],
                "sample":    r["sample"],
                "check_text": v.get("check", ""),
                "verdict":   v.get("verdict", "UNCLEAR"),
                "reason":    v.get("reason", ""),
            })
    return pd.DataFrame(rows)


def failure_rate_by_check(expanded_df: pd.DataFrame) -> pd.DataFrame:
    """Per-check failure rate grouped by (provider, prompt_id, check_text).

    A check is considered failed if verdict is NO or UNCLEAR.
    """
    df = expanded_df.copy()
    df["failed"] = df["verdict"].isin(["NO", "UNCLEAR"]).astype(int)
    grouped = (
        df.groupby(["provider", "prompt_id", "category", "check_text"])
        .agg(
            n_samples=("failed", "count"),
            n_failed=("failed", "sum"),
        )
        .reset_index()
    )
    grouped["failure_rate"] = (grouped["n_failed"] / grouped["n_samples"]).round(3)
    return grouped.sort_values("failure_rate", ascending=False)


# -- Check-type classifier --------------------------------------------------

_COUNT_RE    = re.compile(r"exactly\s+\d+|(?:^|\s)\d+\s+\w+", re.IGNORECASE)
_COLOR_RE    = re.compile(
    r"\b(red|blue|green|yellow|orange|purple|pink|black|white|brown|gray|grey|golden|silver)\b",
    re.IGNORECASE,
)
_SPATIAL_RE  = re.compile(
    r"\b(left|right|above|below|behind|front|between|beside|next\s+to|on\s+top|under|over|facing|toward)\b",
    re.IGNORECASE,
)
_EXIST_RE    = re.compile(r"\b(there\s+is|there\s+are|contains?|includes?|has\s+a)\b", re.IGNORECASE)
_TEXT_RE     = re.compile(r'\b(text|reads?|says?|written|inscription|label|sign|word|letter|font)\b', re.IGNORECASE)


def classify_check_type(check_text: str) -> str:
    """Rule-based classifier for check text into a taxonomy."""
    t = check_text.strip()
    # Order matters: more specific first
    if _TEXT_RE.search(t):
        return "text_content"
    if _COUNT_RE.search(t):
        return "count"
    if _SPATIAL_RE.search(t):
        return "spatial"
    if _COLOR_RE.search(t):
        return "color_attribute"
    if _EXIST_RE.search(t):
        return "existence"
    return "other"


def error_taxonomy(expanded_df: pd.DataFrame) -> pd.DataFrame:
    """Failure rate by (provider, check_type).

    Answers: "what kinds of requirements does each model fail on?"
    """
    df = expanded_df.copy()
    df["check_type"] = df["check_text"].apply(classify_check_type)
    df["failed"] = df["verdict"].isin(["NO", "UNCLEAR"]).astype(int)

    grouped = (
        df.groupby(["provider", "check_type"])
        .agg(
            n_checks=("failed", "count"),
            n_failed=("failed", "sum"),
        )
        .reset_index()
    )
    grouped["failure_rate"] = (grouped["n_failed"] / grouped["n_checks"]).round(4)
    grouped["failure_pct"] = (grouped["failure_rate"] * 100).round(2)
    return grouped.sort_values(["check_type", "provider"])


def worst_failures(expanded_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the top_n most-failed checks per provider with reasons."""
    df = expanded_df.copy()
    df["failed"] = df["verdict"].isin(["NO", "UNCLEAR"]).astype(int)
    failed_only = df[df["failed"] == 1].copy()
    failed_only["check_type"] = failed_only["check_text"].apply(classify_check_type)

    # Group by provider + check to get failure counts and example reasons
    agg = (
        failed_only.groupby(["provider", "prompt_id", "check_text", "check_type"])
        .agg(
            times_failed=("failed", "sum"),
            example_reason=("reason", "first"),
        )
        .reset_index()
        .sort_values(["provider", "times_failed"], ascending=[True, False])
    )
    # Top N per provider
    result = agg.groupby("provider").head(top_n).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# B. Cross-Sample Consistency
# ---------------------------------------------------------------------------

def sample_consistency(
    faith_df: pd.DataFrame,
    human_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Measure generation consistency across the 3 samples per prompt.

    Returns dict with:
        faithfulness_consistency: DataFrame with per-(prompt_id, provider) variance
        overall_faith_std:       mean std across all prompt×provider pairs
        human_consistency:       dict with agreement rate (if human_df provided)
    """
    result: dict[str, Any] = {}

    # Faithfulness consistency: std of faithfulness_score across samples
    faith_cons = (
        faith_df.groupby(["provider", "prompt_id", "category"])["faithfulness_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    faith_cons.columns = ["provider", "prompt_id", "category", "mean_score", "std_score", "n_samples"]
    faith_cons["std_score"] = faith_cons["std_score"].fillna(0)
    result["faithfulness_consistency"] = faith_cons
    result["overall_faith_std"] = faith_cons["std_score"].mean()

    # High-variance prompts (std > 0)
    high_var = faith_cons[faith_cons["std_score"] > 0].sort_values("std_score", ascending=False)
    result["high_variance_prompts"] = high_var

    # Human consistency: for each prompt, do all 3 samples have the same winner?
    if human_df is not None and len(human_df) > 0:
        h_cons = (
            human_df.groupby(["prompt_id", "category"])["winner"]
            .agg(list)
            .reset_index()
        )
        h_cons["n_samples"] = h_cons["winner"].apply(len)
        h_cons["all_agree"] = h_cons["winner"].apply(lambda ws: len(set(ws)) == 1)
        h_cons["unique_winners"] = h_cons["winner"].apply(lambda ws: len(set(ws)))

        agreement_rate = h_cons["all_agree"].mean() * 100
        result["human_consistency"] = {
            "detail": h_cons,
            "agreement_rate": round(agreement_rate, 1),
            "n_prompts": len(h_cons),
            "n_agree": int(h_cons["all_agree"].sum()),
        }
    else:
        result["human_consistency"] = None

    return result


# ---------------------------------------------------------------------------
# C. Statistical Testing
# ---------------------------------------------------------------------------

def bootstrap_ci(
    scores: np.ndarray | list,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Returns (mean, ci_lower, ci_upper).
    """
    arr = np.asarray(scores, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(rng_seed)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (float(arr.mean()), float(lo), float(hi))


def permutation_test(
    scores_a: np.ndarray | list,
    scores_b: np.ndarray | list,
    n_perm: int = 10000,
    rng_seed: int = 42,
) -> float:
    """Two-sample permutation test for difference in means.

    Returns p-value (two-sided).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan

    observed_diff = abs(a.mean() - b.mean())
    combined = np.concatenate([a, b])
    n_a = len(a)

    rng = np.random.default_rng(rng_seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = abs(combined[:n_a].mean() - combined[n_a:].mean())
        if perm_diff >= observed_diff:
            count += 1

    return count / n_perm


def compute_statistics(
    faith_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    human_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute bootstrap CIs and permutation tests for all key comparisons.

    Returns dict with structured results for the report.
    """
    providers = sorted(faith_df["provider"].unique())
    categories = sorted(faith_df["category"].unique())
    results: dict[str, Any] = {"providers": providers}

    # --- Faithfulness CIs ---
    faith_cis: list[dict] = []
    for p in providers:
        scores = faith_df[faith_df["provider"] == p]["faithfulness_score"].values
        mean, lo, hi = bootstrap_ci(scores)
        faith_cis.append({
            "provider": p, "metric": "faithfulness", "scope": "overall",
            "mean": round(mean * 100, 2), "ci_lo": round(lo * 100, 2), "ci_hi": round(hi * 100, 2),
            "n": len(scores),
        })
    results["faithfulness_cis"] = faith_cis

    # Per-category faithfulness CIs
    faith_cat_cis: list[dict] = []
    for cat in categories:
        for p in providers:
            scores = faith_df[
                (faith_df["provider"] == p) & (faith_df["category"] == cat)
            ]["faithfulness_score"].values
            mean, lo, hi = bootstrap_ci(scores)
            faith_cat_cis.append({
                "provider": p, "metric": "faithfulness", "scope": cat,
                "mean": round(mean * 100, 2), "ci_lo": round(lo * 100, 2), "ci_hi": round(hi * 100, 2),
                "n": len(scores),
            })
    results["faithfulness_cat_cis"] = faith_cat_cis

    # --- Faithfulness permutation tests ---
    if len(providers) == 2:
        pa, pb = providers
        # Overall
        sa = faith_df[faith_df["provider"] == pa]["faithfulness_score"].values
        sb = faith_df[faith_df["provider"] == pb]["faithfulness_score"].values
        results["faithfulness_perm_overall"] = {
            "p_value": permutation_test(sa, sb),
            "diff": round((sa.mean() - sb.mean()) * 100, 2),
        }
        # Per-category
        faith_perm_cat: list[dict] = []
        for cat in categories:
            sa_c = faith_df[(faith_df["provider"] == pa) & (faith_df["category"] == cat)]["faithfulness_score"].values
            sb_c = faith_df[(faith_df["provider"] == pb) & (faith_df["category"] == cat)]["faithfulness_score"].values
            faith_perm_cat.append({
                "category": cat,
                "p_value": permutation_test(sa_c, sb_c),
                "diff": round((sa_c.mean() - sb_c.mean()) * 100, 2),
            })
        results["faithfulness_perm_cat"] = faith_perm_cat

    # --- Quality dimension CIs ---
    quality_dims = ["subject_clarity", "composition", "technical", "aesthetic", "coherence"]
    qual_dim_cis: list[dict] = []
    for dim in quality_dims:
        if dim not in quality_df.columns:
            continue
        for p in providers:
            scores = quality_df[quality_df["provider"] == p][dim].values / 5.0  # normalize to 0-1
            mean, lo, hi = bootstrap_ci(scores)
            qual_dim_cis.append({
                "provider": p, "dimension": dim,
                "mean": round(mean * 100, 2), "ci_lo": round(lo * 100, 2), "ci_hi": round(hi * 100, 2),
                "n": len(scores),
            })
    results["quality_dim_cis"] = qual_dim_cis

    # --- Human win rate CIs (binomial bootstrap) ---
    non_model = {"tie", "neither"}
    models_in_human = sorted(w for w in human_df["winner"].unique() if w not in non_model)
    human_cis: list[dict] = []
    decisive = human_df[human_df["winner"].isin(models_in_human)]
    for m in models_in_human:
        wins = (decisive["winner"] == m).astype(int).values
        mean, lo, hi = bootstrap_ci(wins)
        human_cis.append({
            "provider": m, "metric": "human_win_rate",
            "mean": round(mean * 100, 2), "ci_lo": round(lo * 100, 2), "ci_hi": round(hi * 100, 2),
            "n_decisive": len(decisive),
        })
    results["human_win_cis"] = human_cis

    # Per-category human CIs
    human_cat_cis: list[dict] = []
    for cat in categories:
        dec_cat = decisive[decisive["category"] == cat]
        for m in models_in_human:
            wins = (dec_cat["winner"] == m).astype(int).values
            if len(wins) == 0:
                continue
            mean, lo, hi = bootstrap_ci(wins)
            human_cat_cis.append({
                "provider": m, "metric": "human_win_rate", "scope": cat,
                "mean": round(mean * 100, 2), "ci_lo": round(lo * 100, 2), "ci_hi": round(hi * 100, 2),
                "n_decisive": len(dec_cat),
            })
    results["human_cat_cis"] = human_cat_cis

    return results


# ---------------------------------------------------------------------------
# D. Signal Disagreement
# ---------------------------------------------------------------------------

def signal_disagreement(
    human_df: pd.DataFrame,
    faith_df: pd.DataFrame,
) -> dict[str, Any]:
    """Analyze where human preference diverges from VLM faithfulness.

    Returns dict with disagreement details and correlation.
    """
    providers = sorted(faith_df["provider"].unique())
    if len(providers) < 2:
        return {"disagreements": pd.DataFrame(), "spearman_rho": None}

    pa, pb = providers[0], providers[1]

    # Compute per-prompt faithfulness delta (pa - pb)
    faith_prompt = (
        faith_df.groupby(["prompt_id", "category", "provider"])["faithfulness_score"]
        .mean()
        .unstack("provider")
        .reset_index()
    )
    faith_prompt["faith_delta"] = faith_prompt[pa] - faith_prompt[pb]

    # Human preference per prompt (aggregate across samples)
    non_model = {"tie", "neither"}
    h_decisive = human_df[~human_df["winner"].isin(non_model)].copy()
    h_prompt = (
        h_decisive.groupby(["prompt_id", "category"])["winner"]
        .agg(lambda ws: ws.mode().iloc[0] if len(ws) > 0 else None)
        .reset_index()
    )
    h_prompt["human_prefers"] = h_prompt["winner"]

    # Merge
    merged = faith_prompt.merge(h_prompt[["prompt_id", "human_prefers"]], on="prompt_id", how="inner")

    # VLM predicted winner based on faithfulness
    def vlm_winner(row):
        if row[pa] > row[pb]:
            return pa
        elif row[pb] > row[pa]:
            return pb
        return "tie"

    merged["vlm_predicts"] = merged.apply(vlm_winner, axis=1)
    merged["agrees"] = merged["human_prefers"] == merged["vlm_predicts"]
    disagreements = merged[~merged["agrees"]].copy()

    # Spearman correlation between faith_delta and human preference direction
    # Encode human_prefers as numeric: pa=+1, pb=-1
    merged["human_dir"] = merged["human_prefers"].map({pa: 1, pb: -1}).fillna(0)
    try:
        from scipy.stats import spearmanr
        rho, p_val = spearmanr(merged["faith_delta"], merged["human_dir"])
    except (ImportError, ValueError):
        rho, p_val = None, None

    return {
        "merged": merged,
        "disagreements": disagreements,
        "agreement_rate": round(merged["agrees"].mean() * 100, 1) if len(merged) > 0 else None,
        "spearman_rho": round(rho, 3) if rho is not None else None,
        "spearman_p": round(p_val, 4) if p_val is not None else None,
        "n_prompts": len(merged),
    }
