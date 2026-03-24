# Text-to-Image Generation Benchmark Report

---

## Executive Summary

- **Human preference**: Gemini (Imagen 4) wins 53% of decisive votes (95% CI: [45%, 62%]) vs GPT-Image-1 Mini at 47%.
- **Category strengths**: Gemini (Imagen 4) dominates **Binding** (62%); GPT-Image-1 Mini dominates **Spatial** (92%); Gemini (Imagen 4) dominates **Typography** (72%).
- **VLM faithfulness**: GPT-Image-1 Mini 99.0% vs Gemini (Imagen 4) 97.4% — scores near ceiling, limited discriminative power.
- **Hardest requirement type**: text rendering — up to 10.3% failure rate (gemini).
- **Generation consistency**: Mean cross-sample std = 0.020 (higher = less consistent).

---

## Methodology

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

---

## Human Preference Results

### Overall Win Rates

| Model | Win Rate | 95% CI | n (decisive) |
|-------|----------|--------|--------------|
| GPT-Image-1 Mini | 46.5% | [38.0%, 55.0%] | 129 |
| Gemini (Imagen 4) | 53.5% | [45.0%, 62.0%] | 129 |

### Win Rates by Category

| Category | Model | Win Rate | 95% CI | n |
|----------|-------|----------|--------|---|
| Binding | GPT-Image-1 Mini | 37.5% | [16.7%, 58.3%] | 24 |
| Binding | Gemini (Imagen 4) | 62.5% | [41.7%, 83.3%] | 24 |
| Count | GPT-Image-1 Mini | 48.1% | [29.6%, 66.7%] | 27 |
| Count | Gemini (Imagen 4) | 51.9% | [33.3%, 70.4%] | 27 |
| Spatial | GPT-Image-1 Mini | 92.0% | [80.0%, 100.0%] | 25 |
| Spatial | Gemini (Imagen 4) | 8.0% | [0.0%, 20.0%] | 25 |
| Typography | GPT-Image-1 Mini | 28.3% | [17.0%, 41.5%] | 53 |
| Typography | Gemini (Imagen 4) | 71.7% | [58.5%, 83.0%] | 53 |

### Cross-Sample Human Consistency

Across 40 prompts, humans gave the same verdict for all 3 samples **40.0%** of the time (16/40).


---

## Systematic Model Capabilities

### Error Taxonomy: Failure Rate by Requirement Type

| Check Type | GPT-Image-1 Mini | Gemini (Imagen 4) |
|------------|-------|-------|
| color_attribute | 0.76% (n=132) | 1.52% (n=132) |
| count | 1.96% (n=51) | 0.00% (n=51) |
| existence | 0.00% (n=126) | 0.00% (n=126) |
| other | 0.00% (n=168) | 0.60% (n=168) |
| spatial | 0.00% (n=57) | 0.00% (n=57) |
| text_content | 3.42% (n=117) | 10.26% (n=117) |

### Specific Failure Examples

**GPT-Image-1 Mini** — top failures:

| Prompt | Check | Type | Failed/3 | Example Reason |
|--------|-------|------|----------|----------------|
| count_004 | There are exactly 5 strawberries | count | 1/3 | There are only three strawberries in the image. |
| typ_003 | The background is yellow | color_attribute | 1/3 | missing from output |
| typ_003 | The design resembles a warning or caution label | text_content | 1/3 | missing from output |
| typ_003 | The text is black or dark-colored | text_content | 1/3 | missing from output |
| typ_003 | The text is bold | text_content | 1/3 | missing from output |

**Gemini (Imagen 4)** — top failures:

| Prompt | Check | Type | Failed/3 | Example Reason |
|--------|-------|------|----------|----------------|
| typ_005 | The design resembles a product label or packaging | text_content | 3/3 | missing from output |
| typ_003 | The background is yellow | color_attribute | 2/3 | missing from output |
| typ_003 | The design resembles a warning or caution label | text_content | 2/3 | missing from output |
| typ_003 | The text is black or dark-colored | text_content | 2/3 | missing from output |
| typ_003 | The text is bold | text_content | 2/3 | missing from output |


---

## VLM-Human Agreement Analysis

Across 34 prompts with decisive human votes, the VLM faithfulness score predicted the human-preferred model **2.9%** of the time.

This low agreement suggests humans weigh dimensions (aesthetics, composition, overall impression) that the faithfulness judge does not capture. The VLM judge measures *literal compliance* with prompt requirements, while humans evaluate *holistic quality*.

### Disagreement Examples

| Prompt | Category | Human Prefers | VLM Predicts | Faith Delta |
|--------|----------|---------------|--------------|-------------|
| bind_002 | binding | Gemini (Imagen 4) | tie | +0.000 |
| bind_003 | binding | Gemini (Imagen 4) | tie | +0.000 |
| bind_004 | binding | GPT-Image-1 Mini | tie | +0.000 |
| bind_005 | binding | Gemini (Imagen 4) | tie | +0.000 |
| bind_007 | binding | GPT-Image-1 Mini | tie | +0.000 |
| bind_008 | binding | GPT-Image-1 Mini | tie | +0.000 |
| bind_009 | binding | GPT-Image-1 Mini | tie | +0.000 |
| bind_010 | binding | Gemini (Imagen 4) | tie | +0.000 |
| count_001 | count | Gemini (Imagen 4) | tie | +0.000 |
| count_002 | count | Gemini (Imagen 4) | tie | +0.000 |


---

## Generation Consistency

Mean faithfulness standard deviation across 3 samples per prompt: **0.0203**.

### Prompts with Highest Variance

| Provider | Prompt | Category | Mean Score | Std |
|----------|--------|----------|------------|-----|
| Gemini (Imagen 4) | typ_001 | typography | 73.3% | 0.4619 |
| Gemini (Imagen 4) | typ_003 | typography | 46.7% | 0.4619 |
| GPT-Image-1 Mini | typ_003 | typography | 73.3% | 0.4619 |
| GPT-Image-1 Mini | typ_005 | typography | 91.7% | 0.1443 |
| GPT-Image-1 Mini | count_004 | count | 94.4% | 0.0962 |

**5** prompt-provider pairs show non-zero variance, indicating the model produces inconsistent results across samples for these prompts.


---

## Statistical Appendix

### Faithfulness Confidence Intervals

| Provider | Scope | Mean | 95% CI | n |
|----------|-------|------|--------|---|
| GPT-Image-1 Mini | Overall | 98.99% | [97.31%, 100.00%] | 120 |
| Gemini (Imagen 4) | Overall | 97.38% | [94.71%, 99.38%] | 120 |
| GPT-Image-1 Mini | Binding | 100.00% | [100.00%, 100.00%] | 30 |
| Gemini (Imagen 4) | Binding | 100.00% | [100.00%, 100.00%] | 30 |
| GPT-Image-1 Mini | Count | 99.44% | [98.33%, 100.00%] | 30 |
| Gemini (Imagen 4) | Count | 100.00% | [100.00%, 100.00%] | 30 |
| GPT-Image-1 Mini | Spatial | 100.00% | [100.00%, 100.00%] | 30 |
| Gemini (Imagen 4) | Spatial | 100.00% | [100.00%, 100.00%] | 30 |
| GPT-Image-1 Mini | Typography | 96.50% | [90.33%, 100.00%] | 30 |
| Gemini (Imagen 4) | Typography | 89.50% | [80.00%, 97.33%] | 30 |

### Permutation Tests (Faithfulness)

**Overall**: diff = +1.61 pp, p = 0.2671

| Category | Diff (pp) | p-value | Significant? |
|----------|-----------|---------|--------------|
| Binding | +0.00 | 1.0000 | No |
| Count | -0.56 | 1.0000 | No |
| Spatial | +0.00 | 1.0000 | No |
| Typography | +7.00 | 0.2523 | No |

### Quality Dimension Confidence Intervals

| Provider | Dimension | Mean (%) | 95% CI |
|----------|-----------|----------|--------|
| GPT-Image-1 Mini | subject_clarity | 99.50% | [98.83%, 100.00%] |
| Gemini (Imagen 4) | subject_clarity | 98.67% | [97.67%, 99.50%] |
| GPT-Image-1 Mini | composition | 99.50% | [98.83%, 100.00%] |
| Gemini (Imagen 4) | composition | 98.67% | [97.67%, 99.50%] |
| GPT-Image-1 Mini | technical | 99.50% | [98.83%, 100.00%] |
| Gemini (Imagen 4) | technical | 98.67% | [97.67%, 99.50%] |
| GPT-Image-1 Mini | aesthetic | 99.50% | [98.83%, 100.00%] |
| Gemini (Imagen 4) | aesthetic | 98.67% | [97.67%, 99.50%] |
| GPT-Image-1 Mini | coherence | 99.50% | [98.83%, 100.00%] |
| Gemini (Imagen 4) | coherence | 98.67% | [97.67%, 99.50%] |

