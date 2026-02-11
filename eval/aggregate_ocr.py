# eval/aggregate_ocr.py
from __future__ import annotations
import pandas as pd

IN_CSV = "results/ocr_scores.csv"
OUT_MODEL = "results/ocr_summary_by_model.csv"
OUT_TARGET = "results/ocr_summary_by_model_target.csv"
OUT_WORST = "results/ocr_worst_cases.csv"

df = pd.read_csv(IN_CSV)

# Basic per-model summary
model_summary = (
    df.groupby("model")
    .agg(
        n=("ocr_score_avg", "count"),
        avg_mean=("ocr_score_avg", "mean"),
        avg_std=("ocr_score_avg", "std"),
        min_mean=("ocr_score_min", "mean"),
        min_std=("ocr_score_min", "std"),
        found_all_rate=("ocr_found_all", "mean"),
        found_count_mean=("ocr_found_count", "mean"),
    )
    .reset_index()
)

# Make rates human-friendly (%)
model_summary["found_all_rate"] = model_summary["found_all_rate"] * 100.0

print("=== OCR summary by model ===")
print(model_summary.to_string(index=False))

model_summary.to_csv(OUT_MODEL, index=False)

# Per-target found rate (target_1_found, target_2_found, ...)
found_cols = [c for c in df.columns if c.endswith("_found") and c.startswith("target_")]
score_cols = [c for c in df.columns if c.endswith("_score") and c.startswith("target_")]

rows = []
for model, g in df.groupby("model"):
    for col in found_cols:
        idx = col.split("_")[1]  # "1" from "target_1_found"
        t_text_col = f"target_{idx}_text"
        t_score_col = f"target_{idx}_score"

        # Some rows may not have target_2_* if prompt only has one target
        if t_text_col not in g.columns or t_score_col not in g.columns:
            continue

        # Use the most common target text in that column (should be consistent)
        target_text = g[t_text_col].dropna()
        target_text = target_text.iloc[0] if len(target_text) else f"(target_{idx})"

        found_rate = g[col].mean() * 100.0
        score_mean = g[t_score_col].mean()

        rows.append(
            {
                "model": model,
                "target_index": int(idx),
                "target_text": target_text,
                "found_rate_pct": found_rate,
                "score_mean": score_mean,
            }
        )

target_summary = pd.DataFrame(rows).sort_values(["model", "target_index"])
print("\n=== OCR per-target summary (by model) ===")
print(target_summary.to_string(index=False))

target_summary.to_csv(OUT_TARGET, index=False)

# Worst cases (by ocr_score_min)
worst = df.sort_values("ocr_score_min").head(15)[
    ["model", "prompt_id", "sample_id", "ocr_score_avg", "ocr_score_min", "ocr_found_all", "image_path"]
]
print("\n=== Worst 15 cases (by ocr_score_min) ===")
print(worst.to_string(index=False))

worst.to_csv(OUT_WORST, index=False)

print(f"\nWrote:\n- {OUT_MODEL}\n- {OUT_TARGET}\n- {OUT_WORST}")
