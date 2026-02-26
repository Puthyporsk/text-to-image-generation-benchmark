"""
eval/judge_quality.py
---------------------
Rubric-style aesthetic / technical quality judge using Qwen2-VL-2B-Instruct.

Scores five dimensions for each image (1 = very poor → 5 = excellent):
  subject_clarity  — main subject clearly rendered and identifiable
  composition      — framing, balance, visual hierarchy
  technical        — sharpness, no VRAM artifacts, no distortion
  aesthetic        — color harmony, lighting, mood
  coherence        — scene consistency, no impossible geometry or glitches

quality_score = mean(dims) / 5  (normalized to 0–1)

Output: results/quality_scores.csv  (one row per image)

Usage:
    python -m eval.judge_quality \
        --run_dir runs/2026-02-12_core40_k3_1024 \
        --prompts  prompts/core40.jsonl

Add --resume to skip images already in --out_csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ---------------------------------------------------------------------------
# Rubric definition
# ---------------------------------------------------------------------------

RUBRIC: List[Tuple[str, str]] = [
    (
        "subject_clarity",
        "Is the main subject clearly rendered, identifiable, and free of major distortions?",
    ),
    (
        "composition",
        "Is the framing, balance, and visual hierarchy effective and intentional?",
    ),
    (
        "technical",
        "Is the image sharp, free of compression artifacts, unnatural noise, or AI glitches?",
    ),
    (
        "aesthetic",
        "Are the colors, lighting, and overall atmosphere visually appealing?",
    ),
    (
        "coherence",
        "Is the scene internally consistent with no impossible geometry, merged objects, or broken text?",
    ),
]

RUBRIC_DIMS = [dim for dim, _ in RUBRIC]

JUDGE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
PROVIDERS = ("gemini", "chatgpt")

SYSTEM_PROMPT = (
    "You are a professional image quality critic. "
    "Score each dimension 1–5 (integers only). "
    "Output ONLY valid JSON. No markdown fences, no extra text."
)

CSV_FIELDS = (
    ["run_dir", "provider", "prompt_id", "category", "sample", "image_path", "quality_score"]
    + RUBRIC_DIMS
    + ["parse_error", "raw_response"]
)

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def build_quality_prompt() -> str:
    rubric_lines = "\n".join(
        f"  {dim}: {desc}"
        for dim, desc in RUBRIC
    )
    schema = {dim: "integer 1-5" for dim in RUBRIC_DIMS}
    return (
        f"Rate the quality of this image on the following dimensions "
        f"(1 = very poor, 5 = excellent):\n\n"
        f"{rubric_lines}\n\n"
        f"Rules:\n"
        f"  - Use only integers 1, 2, 3, 4, or 5.\n"
        f"  - Be critical: reserve 5 for genuinely outstanding quality.\n"
        f"  - Output ONLY the JSON below, no other text.\n\n"
        f"JSON schema:\n{json.dumps(schema, ensure_ascii=False)}\n"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_scores(raw: str) -> Tuple[Dict[str, int], bool]:
    """Extract per-dimension integer scores from the raw model output."""
    lower = raw.lower()
    search_from = lower.rfind("assistant")
    search_from = max(search_from, 0)
    sub = raw[search_from:]

    j_start = sub.find("{")
    j_end = sub.rfind("}")
    if j_start == -1 or j_end == -1 or j_end <= j_start:
        return _default_scores(), True

    try:
        data = json.loads(sub[j_start : j_end + 1])
    except json.JSONDecodeError:
        return _default_scores(), True

    scores: Dict[str, int] = {}
    parse_error = False
    for dim in RUBRIC_DIMS:
        val = data.get(dim)
        try:
            val = int(val)
            val = max(1, min(5, val))
        except (TypeError, ValueError):
            val = 3  # mid-point fallback
            parse_error = True
        scores[dim] = val
    return scores, parse_error


def _default_scores() -> Dict[str, int]:
    """Return mid-point scores for all dimensions."""
    return {dim: 3 for dim in RUBRIC_DIMS}


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


@torch.inference_mode()
def judge_one(
    processor: AutoProcessor,
    model: Qwen2VLForConditionalGeneration,
    image: Image.Image,
) -> str:
    eval_prompt = build_quality_prompt()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": eval_prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    img_inputs, vid_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=img_inputs,
        videos=vid_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    out_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# I/O helpers  (same as judge_faithfulness)
# ---------------------------------------------------------------------------


def load_prompts(jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pid = item.get("prompt_id") or item.get("id")
            if pid:
                by_id[str(pid)] = item
    return by_id


def normalize_run_dir(run_dir: str | Path) -> Path:
    p = Path(run_dir)
    if "runs" not in p.parts:
        p = Path("runs") / p
    return p


def parse_filename(path: Path) -> Tuple[str, int]:
    stem = path.stem
    if "__s" not in stem:
        raise ValueError(f"Bad filename (missing __s): {path.name}")
    prompt_id, s = stem.rsplit("__s", 1)
    return prompt_id, int(s)


def load_done_keys(csv_path: Path) -> set:
    done: set = set()
    if not csv_path.exists():
        return done
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["provider"], row["prompt_id"], row["sample"]))
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rubric-style quality judge for text-to-image benchmark"
    )
    ap.add_argument(
        "--run_dir",
        required=True,
        help="runs/<run_name> (or bare run_name; auto-prefixed with runs/)",
    )
    ap.add_argument("--prompts", default="prompts/core40.jsonl")
    ap.add_argument("--out_csv", default="results/quality_scores.csv")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip images already present in --out_csv",
    )
    args = ap.parse_args()

    run_path = normalize_run_dir(args.run_dir)
    print(f"[RUN]  run_dir = {run_path.as_posix()}")

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        raise SystemExit(f"Prompts file not found: {prompts_path}")

    by_id = load_prompts(prompts_path)
    print(f"[INFO] Loaded {len(by_id)} prompts")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done_keys = load_done_keys(out_csv) if args.resume else set()
    if done_keys:
        print(f"[RESUME] Skipping {len(done_keys)} already-evaluated images")

    tasks: List[Tuple[str, str, int, Path]] = []
    for provider in PROVIDERS:
        img_dir = run_path / "images" / provider
        if not img_dir.exists():
            print(f"[WARN] Missing dir: {img_dir.as_posix()}")
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            try:
                prompt_id, sample = parse_filename(img_path)
            except ValueError as e:
                print(f"[WARN] {e}")
                continue
            if prompt_id not in by_id:
                continue
            if (provider, prompt_id, str(sample)) in done_keys:
                continue
            tasks.append((provider, prompt_id, sample, img_path))

    print(f"[INFO] {len(tasks)} images to evaluate")
    if not tasks:
        print("[INFO] Nothing to do. Exiting.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENV]  device = {device}")
    print(f"[ENV]  loading {JUDGE_MODEL} ...")
    processor = AutoProcessor.from_pretrained(JUDGE_MODEL)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        JUDGE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()
    if device == "cuda":
        print(
            f"[ENV]  VRAM after load: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

    write_header = not out_csv.exists() or not args.resume

    with out_csv.open(
        "a" if args.resume else "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for provider, prompt_id, sample, img_path in tqdm(tasks, desc="Quality scoring"):
            item = by_id[prompt_id]
            category: str = item.get("category", "")

            image = Image.open(img_path).convert("RGB")
            raw = judge_one(processor, model, image)
            scores, parse_error = parse_scores(raw)

            quality_score = round(sum(scores.values()) / (5.0 * len(RUBRIC_DIMS)), 4)

            row: Dict[str, Any] = {
                "run_dir": run_path.as_posix(),
                "provider": provider,
                "prompt_id": prompt_id,
                "category": category,
                "sample": sample,
                "image_path": img_path.as_posix(),
                "quality_score": quality_score,
                "parse_error": int(parse_error),
                "raw_response": raw[-400:],
            }
            row.update(scores)
            writer.writerow(row)
            f.flush()

    print(f"[DONE] wrote {len(tasks)} rows -> {out_csv.as_posix()}")


if __name__ == "__main__":
    main()
