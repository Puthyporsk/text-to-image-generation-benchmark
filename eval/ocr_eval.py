# eval/ocr_eval.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
from PIL import Image
import pytesseract
from rapidfuzz import fuzz


def normalize(s: str) -> str:
    s = s.upper()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ocr_text(img_path: Path) -> str:
    img = Image.open(img_path).convert("RGB")
    text = pytesseract.image_to_string(img)
    return normalize(text)


def best_fuzzy_score(haystack: str, needle: str) -> int:
    # 0..100
    return fuzz.partial_ratio(needle, haystack)


def load_prompts(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    prompts: Dict[str, Dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            prompts[j["prompt_id"]] = j
    return prompts


def iter_images(run_dir: Path, model: str) -> List[Path]:
    img_dir = run_dir / "images" / model
    if not img_dir.exists():
        return []
    exts = (".png", ".jpg", ".jpeg", ".webp")
    return sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])


def parse_filename(p: Path) -> Tuple[str, int]:
    stem = p.stem
    if "__s" not in stem:
        raise ValueError(f"Bad filename (missing __s): {p.name}")
    prompt_id, s_part = stem.split("__s", 1)
    return prompt_id, int(s_part)


def main(run_dir: str, prompts_file: str, out_csv: str, threshold: int):
    run_path = Path(run_dir)
    prompts = load_prompts(Path(prompts_file))

    rows: List[Dict[str, Any]] = []

    for model in ["gemini", "chatgpt"]:
        for img_path in iter_images(run_path, model):
            try:
                prompt_id, sample_id = parse_filename(img_path)
            except Exception:
                continue

            p = prompts.get(prompt_id)
            if not p:
                continue

            targets = (p.get("eval_targets") or {}).get("text", [])
            if not targets:
                continue

            ocr = ocr_text(img_path)

            # per-target scores
            target_norms = [normalize(t) for t in targets]
            per_scores = [best_fuzzy_score(ocr, t) for t in target_norms]
            per_found = [s >= threshold for s in per_scores]

            # store per-target details as additional columns
            row: Dict[str, Any] = {
                "model": model,
                "prompt_id": prompt_id,
                "sample_id": sample_id,
                "image_path": str(img_path),
                "ocr_text": ocr,
                "targets": " | ".join(targets),
                # summary
                "ocr_score_avg": sum(per_scores) / len(per_scores),
                "ocr_score_min": min(per_scores),
                "ocr_found_all": all(per_found),
                "ocr_found_count": sum(1 for x in per_found if x),
                "ocr_target_count": len(per_found),
            }

            # per-target columns (target_1_text, target_1_score, etc.)
            for i, (t_raw, score, found) in enumerate(zip(targets, per_scores, per_found), start=1):
                row[f"target_{i}_text"] = t_raw
                row[f"target_{i}_score"] = score
                row[f"target_{i}_found"] = found

            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} OCR rows to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out_csv", default="results/ocr_scores.csv")
    ap.add_argument(
        "--threshold",
        type=int,
        default=85,
        help="Score threshold (0-100) to count a target as 'found'.",
    )
    args = ap.parse_args()

    main(args.run_dir, args.prompts, args.out_csv, args.threshold)
