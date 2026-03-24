"""
eval/benchmark_report.py
------------------------
Generates a unified Markdown benchmark report from all evaluation signals.

Outputs: results/benchmark_report.md

Usage:
    python -m eval.benchmark_report
    python -m eval.benchmark_report --out results/benchmark_report.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eval.deep_analysis import (
    bootstrap_ci,
    classify_check_type,
    compute_statistics,
    error_taxonomy,
    expand_check_verdicts,
    failure_rate_by_check,
    sample_consistency,
    signal_disagreement,
    worst_failures,
)
from providers.registry import label as provider_label

FAITH_CSV    = Path("results/faithfulness_scores.csv")
QUALITY_CSV  = Path("results/quality_scores.csv")
RANKINGS_CSV = Path("results/human_rankings.csv")

CATEGORIES = ["binding", "count", "spatial", "typography"]
CAT_LABELS = {"binding": "Binding", "count": "Count",
              "spatial": "Spatial", "typography": "Typography"}


# ---------------------------------------------------------------------------
# Section generators — each returns a Markdown string
# ---------------------------------------------------------------------------

def _section_executive_summary(
    stats: dict, taxonomy_df: pd.DataFrame, consistency: dict,
) -> str:
    lines = [
        "## Executive Summary\n",
    ]

    providers = stats["providers"]
    # Human win rates
    human_cis = {r["provider"]: r for r in stats["human_win_cis"]}
    faith_cis = {r["provider"]: r for r in stats["faithfulness_cis"]}

    # Determine overall human leader
    if len(providers) == 2:
        pa, pb = providers
        ha, hb = human_cis.get(pa, {}), human_cis.get(pb, {})
        if ha and hb:
            leader = pa if ha["mean"] > hb["mean"] else pb
            trailer = pb if leader == pa else pa
            lines.append(
                f"- **Human preference**: {provider_label(leader)} wins "
                f"{human_cis[leader]['mean']:.0f}% of decisive votes "
                f"(95% CI: [{human_cis[leader]['ci_lo']:.0f}%, {human_cis[leader]['ci_hi']:.0f}%]) "
                f"vs {provider_label(trailer)} at {human_cis[trailer]['mean']:.0f}%."
            )

        # Per-category highlights
        cat_cis = stats.get("human_cat_cis", [])
        cat_by_scope = {}
        for r in cat_cis:
            cat_by_scope.setdefault(r["scope"], {})[r["provider"]] = r
        highlights = []
        for cat in CATEGORIES:
            if cat in cat_by_scope and len(cat_by_scope[cat]) == 2:
                vals = cat_by_scope[cat]
                winner = max(vals, key=lambda p: vals[p]["mean"])
                if vals[winner]["mean"] > 60:
                    highlights.append(
                        f"{provider_label(winner)} dominates **{CAT_LABELS[cat]}** "
                        f"({vals[winner]['mean']:.0f}%)"
                    )
        if highlights:
            lines.append(f"- **Category strengths**: {'; '.join(highlights)}.")

    # Faithfulness
    faith_summary = []
    for p in providers:
        if p in faith_cis:
            faith_summary.append(
                f"{provider_label(p)} {faith_cis[p]['mean']:.1f}%"
            )
    lines.append(f"- **VLM faithfulness**: {' vs '.join(faith_summary)} — scores near ceiling, limited discriminative power.")

    # Key finding from taxonomy
    text_rows = taxonomy_df[taxonomy_df["check_type"] == "text_content"]
    if len(text_rows) > 0:
        worst_text = text_rows.sort_values("failure_pct", ascending=False).iloc[0]
        lines.append(
            f"- **Hardest requirement type**: text rendering — "
            f"up to {worst_text['failure_pct']:.1f}% failure rate "
            f"({worst_text['provider']})."
        )

    # Consistency
    overall_std = consistency.get("overall_faith_std", 0)
    lines.append(
        f"- **Generation consistency**: Mean cross-sample std = {overall_std:.3f} "
        f"(higher = less consistent)."
    )

    return "\n".join(lines) + "\n"


def _section_methodology() -> str:
    return """## Methodology

| Aspect | Detail |
|--------|--------|
| **Models** | Gemini Imagen 4 (`imagen-4.0-generate-001`), GPT Image 1 Mini (`gpt-image-1-mini`) |
| **Prompt suite** | 40 prompts across 4 categories (binding, count, spatial, typography), 10 each |
| **Samples** | 3 per prompt per model = 240 images per model |
| **Evaluation signals** | (1) Human pairwise ranking, (2) VLM faithfulness (Qwen2-VL-7B), (3) VLM quality (5-dim rubric) |
| **VLM judge** | Qwen2-VL-7B-Instruct, 4-bit quantized, RTX 3060 Ti |
| **Human evaluation** | Pairwise comparison (winner/tie/neither), single annotator |

### Limitations
- Single human annotator — no inter-rater reliability measurement.
- VLM judge is a 7B model; larger models may produce different verdicts.
- Small sample size (n=30 per category per model) limits statistical power.
- VLM scores exhibit strong ceiling effects (>95%), reducing discriminative validity.
"""


def _section_human_results(stats: dict, consistency: dict) -> str:
    lines = ["## Human Preference Results\n"]

    human_cis = stats.get("human_win_cis", [])
    if human_cis:
        lines.append("### Overall Win Rates\n")
        lines.append("| Model | Win Rate | 95% CI | n (decisive) |")
        lines.append("|-------|----------|--------|--------------|")
        for r in human_cis:
            lines.append(
                f"| {provider_label(r['provider'])} | {r['mean']:.1f}% "
                f"| [{r['ci_lo']:.1f}%, {r['ci_hi']:.1f}%] "
                f"| {r['n_decisive']} |"
            )
        lines.append("")

    # Per-category
    cat_cis = stats.get("human_cat_cis", [])
    if cat_cis:
        lines.append("### Win Rates by Category\n")
        lines.append("| Category | Model | Win Rate | 95% CI | n |")
        lines.append("|----------|-------|----------|--------|---|")
        for r in sorted(cat_cis, key=lambda x: (x.get("scope", ""), x["provider"])):
            lines.append(
                f"| {CAT_LABELS.get(r.get('scope', ''), r.get('scope', ''))} "
                f"| {provider_label(r['provider'])} "
                f"| {r['mean']:.1f}% "
                f"| [{r['ci_lo']:.1f}%, {r['ci_hi']:.1f}%] "
                f"| {r['n_decisive']} |"
            )
        lines.append("")

    # Cross-sample consistency
    h_cons = consistency.get("human_consistency")
    if h_cons:
        lines.append("### Cross-Sample Human Consistency\n")
        lines.append(
            f"Across {h_cons['n_prompts']} prompts, humans gave the same verdict for all 3 samples "
            f"**{h_cons['agreement_rate']:.1f}%** of the time ({h_cons['n_agree']}/{h_cons['n_prompts']}).\n"
        )

    return "\n".join(lines) + "\n"


def _section_capabilities(
    taxonomy_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    providers: list[str],
) -> str:
    lines = ["## Systematic Model Capabilities\n"]

    lines.append("### Error Taxonomy: Failure Rate by Requirement Type\n")
    lines.append("| Check Type | " + " | ".join(provider_label(p) for p in providers) + " |")
    lines.append("|------------|" + "|".join("-------" for _ in providers) + "|")

    check_types = sorted(taxonomy_df["check_type"].unique())
    for ct in check_types:
        row_parts = [f"| {ct} "]
        for p in providers:
            match = taxonomy_df[(taxonomy_df["check_type"] == ct) & (taxonomy_df["provider"] == p)]
            if len(match) > 0:
                rate = match.iloc[0]["failure_pct"]
                n = int(match.iloc[0]["n_checks"])
                row_parts.append(f"| {rate:.2f}% (n={n}) ")
            else:
                row_parts.append("| — ")
        lines.append("".join(row_parts) + "|")
    lines.append("")

    # Worst failures per provider
    lines.append("### Specific Failure Examples\n")
    for p in providers:
        p_fails = failures_df[failures_df["provider"] == p].head(5)
        if len(p_fails) == 0:
            continue
        lines.append(f"**{provider_label(p)}** — top failures:\n")
        lines.append("| Prompt | Check | Type | Failed/3 | Example Reason |")
        lines.append("|--------|-------|------|----------|----------------|")
        for _, r in p_fails.iterrows():
            reason = str(r.get("example_reason", ""))[:80]
            lines.append(
                f"| {r['prompt_id']} | {r['check_text'][:50]} | {r['check_type']} "
                f"| {int(r['times_failed'])}/3 | {reason} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _section_vlm_agreement(disagree: dict) -> str:
    lines = ["## VLM-Human Agreement Analysis\n"]

    agreement_rate = disagree.get("agreement_rate")
    rho = disagree.get("spearman_rho")
    p_val = disagree.get("spearman_p")
    n = disagree.get("n_prompts", 0)

    if agreement_rate is not None:
        lines.append(
            f"Across {n} prompts with decisive human votes, the VLM faithfulness "
            f"score predicted the human-preferred model **{agreement_rate:.1f}%** of the time.\n"
        )

    if rho is not None:
        sig = "significant" if (p_val is not None and p_val < 0.05) else "not significant"
        lines.append(
            f"Spearman correlation between faithfulness delta and human preference: "
            f"**rho = {rho:.3f}** (p = {p_val:.4f}, {sig}).\n"
        )

    lines.append(
        "This low agreement suggests humans weigh dimensions (aesthetics, composition, "
        "overall impression) that the faithfulness judge does not capture. The VLM judge "
        "measures *literal compliance* with prompt requirements, while humans evaluate "
        "*holistic quality*.\n"
    )

    # Disagreement examples
    disagreements = disagree.get("disagreements", pd.DataFrame())
    if len(disagreements) > 0:
        lines.append("### Disagreement Examples\n")
        lines.append("| Prompt | Category | Human Prefers | VLM Predicts | Faith Delta |")
        lines.append("|--------|----------|---------------|--------------|-------------|")
        for _, r in disagreements.head(10).iterrows():
            lines.append(
                f"| {r['prompt_id']} | {r.get('category', '')} "
                f"| {provider_label(str(r['human_prefers']))} "
                f"| {provider_label(str(r['vlm_predicts']))} "
                f"| {r['faith_delta']:+.3f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _section_consistency(consistency: dict) -> str:
    lines = ["## Generation Consistency\n"]

    overall_std = consistency.get("overall_faith_std", 0)
    lines.append(
        f"Mean faithfulness standard deviation across 3 samples per prompt: **{overall_std:.4f}**.\n"
    )

    high_var = consistency.get("high_variance_prompts", pd.DataFrame())
    if len(high_var) > 0:
        lines.append("### Prompts with Highest Variance\n")
        lines.append("| Provider | Prompt | Category | Mean Score | Std |")
        lines.append("|----------|--------|----------|------------|-----|")
        for _, r in high_var.head(15).iterrows():
            lines.append(
                f"| {provider_label(r['provider'])} | {r['prompt_id']} | {r['category']} "
                f"| {r['mean_score']*100:.1f}% | {r['std_score']:.4f} |"
            )
        lines.append("")
        lines.append(
            f"**{len(high_var)}** prompt-provider pairs show non-zero variance, "
            f"indicating the model produces inconsistent results across samples for these prompts.\n"
        )
    else:
        lines.append("All prompt-provider pairs show zero variance — perfectly consistent generation.\n")

    return "\n".join(lines) + "\n"


def _section_statistical_appendix(stats: dict) -> str:
    lines = ["## Statistical Appendix\n"]

    # Faithfulness CIs
    lines.append("### Faithfulness Confidence Intervals\n")
    lines.append("| Provider | Scope | Mean | 95% CI | n |")
    lines.append("|----------|-------|------|--------|---|")
    for r in stats.get("faithfulness_cis", []):
        lines.append(
            f"| {provider_label(r['provider'])} | Overall "
            f"| {r['mean']:.2f}% | [{r['ci_lo']:.2f}%, {r['ci_hi']:.2f}%] | {r['n']} |"
        )
    for r in stats.get("faithfulness_cat_cis", []):
        lines.append(
            f"| {provider_label(r['provider'])} | {CAT_LABELS.get(r['scope'], r['scope'])} "
            f"| {r['mean']:.2f}% | [{r['ci_lo']:.2f}%, {r['ci_hi']:.2f}%] | {r['n']} |"
        )
    lines.append("")

    # Permutation tests
    perm_overall = stats.get("faithfulness_perm_overall")
    if perm_overall:
        lines.append("### Permutation Tests (Faithfulness)\n")
        lines.append(
            f"**Overall**: diff = {perm_overall['diff']:+.2f} pp, "
            f"p = {perm_overall['p_value']:.4f}\n"
        )
        perm_cat = stats.get("faithfulness_perm_cat", [])
        if perm_cat:
            lines.append("| Category | Diff (pp) | p-value | Significant? |")
            lines.append("|----------|-----------|---------|--------------|")
            for r in perm_cat:
                sig = "Yes" if r["p_value"] < 0.05 else "No"
                lines.append(
                    f"| {CAT_LABELS.get(r['category'], r['category'])} "
                    f"| {r['diff']:+.2f} | {r['p_value']:.4f} | {sig} |"
                )
            lines.append("")

    # Quality dimension CIs
    qual_cis = stats.get("quality_dim_cis", [])
    if qual_cis:
        lines.append("### Quality Dimension Confidence Intervals\n")
        lines.append("| Provider | Dimension | Mean (%) | 95% CI |")
        lines.append("|----------|-----------|----------|--------|")
        for r in qual_cis:
            lines.append(
                f"| {provider_label(r['provider'])} | {r['dimension']} "
                f"| {r['mean']:.2f}% | [{r['ci_lo']:.2f}%, {r['ci_hi']:.2f}%] |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main: assemble report
# ---------------------------------------------------------------------------

def generate_report(
    faith_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    human_df: pd.DataFrame,
) -> str:
    """Generate the full benchmark report as a Markdown string."""
    providers = sorted(faith_df["provider"].unique())

    # Run all analyses
    expanded = expand_check_verdicts(faith_df)
    taxonomy = error_taxonomy(expanded)
    failures = worst_failures(expanded, top_n=10)
    consistency = sample_consistency(faith_df, human_df)
    stats = compute_statistics(faith_df, quality_df, human_df)
    disagree = signal_disagreement(human_df, faith_df)

    # Assemble sections
    sections = [
        "# Text-to-Image Generation Benchmark Report\n",
        _section_executive_summary(stats, taxonomy, consistency),
        _section_methodology(),
        _section_human_results(stats, consistency),
        _section_capabilities(taxonomy, failures, providers),
        _section_vlm_agreement(disagree),
        _section_consistency(consistency),
        _section_statistical_appendix(stats),
    ]

    return "\n---\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unified benchmark report")
    parser.add_argument("--out", default="results/benchmark_report.md",
                        help="Output path for the Markdown report")
    args = parser.parse_args()

    for p in (FAITH_CSV, QUALITY_CSV, RANKINGS_CSV):
        if not p.exists():
            raise SystemExit(f"Required file not found: {p}")

    print("Loading data...")
    faith_df = pd.read_csv(FAITH_CSV)
    quality_df = pd.read_csv(QUALITY_CSV)
    human_df = pd.read_csv(RANKINGS_CSV)

    print("Running analyses and generating report...")
    report = generate_report(faith_df, quality_df, human_df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
