"""
eval/judge_faithfulness.py
--------------------------
VLM-based faithfulness judge using Qwen2-VL-2B-Instruct.

For every image in runs/<run>/images/{gemini,chatgpt}/ it:
  - matches the filename to a prompt in prompts/core40.jsonl
  - feeds (image, prompt, checks[]) into the VLM
  - extracts YES / NO / UNCLEAR per check
  - computes faithfulness_score = YES / total
  - appends one row to results/faithfulness_scores.csv

Usage:
    python -m eval.judge_faithfulness \
        --run_dir runs/2026-02-12_core40_k3_1024 \
        --prompts  prompts/core40.jsonl \
        --out_csv  results/faithfulness_scores.csv

Add --resume to skip images already present in the CSV.
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
# Constants
# ---------------------------------------------------------------------------
JUDGE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
PROVIDERS = ("gemini", "chatgpt")

SYSTEM_PROMPT = (
    "You are a strict image evaluation judge. "
    "Output ONLY valid JSON. No markdown fences, no extra text."
)

CSV_FIELDS = [
    "run_dir",
    "provider",
    "prompt_id",
    "category",
    "sample",
    "image_path",
    "faithfulness_score",
    "checks_total",
    "checks_yes",
    "checks_no",
    "checks_unclear",
    "check_verdicts",   # JSON array of {check, verdict, reason}
    "parse_error",      # 0 or 1
    "raw_response",     # last 600 chars of model output
]

VERDICT_VALUES = ("YES", "NO", "UNCLEAR")

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def build_judge_prompt(prompt_text: str, checks: List[str]) -> str:
    checks_lines = "\n".join(f"{i+1}. {c}" for i, c in enumerate(checks))
    schema_example = json.dumps(
        {
            "verdicts": [
                {
                    "check": "<check text>",
                    "verdict": "YES",
                    "reason": "<one sentence evidence>",
                }
            ]
        },
        ensure_ascii=False,
    )
    return (
        f"Evaluate this generated image against the prompt and checklist.\n\n"
        f"PROMPT: {prompt_text}\n\n"
        f"CHECKLIST:\n{checks_lines}\n\n"
        f"Verdict definitions:\n"
        f"  YES     = requirement clearly satisfied in the image\n"
        f"  NO      = requirement clearly not satisfied\n"
        f"  UNCLEAR = impossible to determine from the image\n\n"
        f"Rules:\n"
        f"  - Be conservative: default to NO when uncertain.\n"
        f"  - Output ONLY the JSON below, no other text.\n\n"
        f"JSON schema:\n{schema_example}\n"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_verdicts(
    raw: str, checks: List[str]
) -> Tuple[List[Dict[str, Any]], bool]:
    """Extract per-check verdicts from the raw model output.

    Returns (verdicts_list, parse_error_flag).
    On parse failure returns all-UNCLEAR verdicts with parse_error=True.
    """
    # The model echoes system+user before the assistant reply.
    # Search for the last JSON object after the "assistant" marker.
    lower = raw.lower()
    search_from = lower.rfind("assistant")
    search_from = max(search_from, 0)
    sub = raw[search_from:]

    j_start = sub.find("{")
    j_end = sub.rfind("}")
    if j_start == -1 or j_end == -1 or j_end <= j_start:
        return _all_unclear(checks, "no json found"), True

    try:
        data = json.loads(sub[j_start : j_end + 1])
    except json.JSONDecodeError:
        return _all_unclear(checks, "json decode error"), True

    raw_verdicts = data.get("verdicts", [])
    if not isinstance(raw_verdicts, list):
        return _all_unclear(checks, "verdicts not a list"), True

    result: List[Dict[str, Any]] = []
    for i, check in enumerate(checks):
        if i < len(raw_verdicts):
            v = raw_verdicts[i]
            verdict = str(v.get("verdict", "UNCLEAR")).upper().strip()
            if verdict not in VERDICT_VALUES:
                verdict = "UNCLEAR"
            result.append(
                {
                    "check": check,
                    "verdict": verdict,
                    "reason": str(v.get("reason", ""))[:200],
                }
            )
        else:
            result.append(
                {"check": check, "verdict": "UNCLEAR", "reason": "missing from output"}
            )
    return result, False


def _all_unclear(checks: List[str], reason: str) -> List[Dict[str, Any]]:
    return [{"check": c, "verdict": "UNCLEAR", "reason": reason} for c in checks]


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def judge_one(
    processor: AutoProcessor,
    model: Qwen2VLForConditionalGeneration,
    image: Image.Image,
    prompt_text: str,
    checks: List[str],
) -> str:
    eval_prompt = build_judge_prompt(prompt_text, checks)
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

    out_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# I/O helpers
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
    """Extract (prompt_id, sample) from <prompt_id>__s<sample>.png."""
    stem = path.stem
    if "__s" not in stem:
        raise ValueError(f"Bad filename (missing __s): {path.name}")
    prompt_id, s = stem.rsplit("__s", 1)
    return prompt_id, int(s)


def load_done_keys(csv_path: Path) -> set:
    """Return set of (provider, prompt_id, sample_str) already in the CSV."""
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
        description="VLM faithfulness judge for text-to-image benchmark"
    )
    ap.add_argument(
        "--run_dir",
        required=True,
        help="runs/<run_name> (or bare run_name; auto-prefixed with runs/)",
    )
    ap.add_argument("--prompts", default="prompts/core40.jsonl")
    ap.add_argument("--out_csv", default="results/faithfulness_scores.csv")
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

    # Collect work items
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
                print(f"[WARN] prompt_id '{prompt_id}' not in prompts file — skipping")
                continue
            if (provider, prompt_id, str(sample)) in done_keys:
                continue
            tasks.append((provider, prompt_id, sample, img_path))

    print(f"[INFO] {len(tasks)} images to evaluate")
    if not tasks:
        print("[INFO] Nothing to do. Exiting.")
        return

    # Load model
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

        for provider, prompt_id, sample, img_path in tqdm(tasks, desc="Judging"):
            item = by_id[prompt_id]
            checks: List[str] = item.get("checks", [])
            prompt_text: str = item.get("prompt", "")
            category: str = item.get("category", "")

            image = Image.open(img_path).convert("RGB")
            raw = judge_one(processor, model, image, prompt_text, checks)

            verdicts, parse_error = parse_verdicts(raw, checks)

            yes = sum(1 for v in verdicts if v["verdict"] == "YES")
            no  = sum(1 for v in verdicts if v["verdict"] == "NO")
            unc = sum(1 for v in verdicts if v["verdict"] == "UNCLEAR")
            total = len(verdicts)
            score = yes / total if total > 0 else 0.0

            writer.writerow(
                {
                    "run_dir": run_path.as_posix(),
                    "provider": provider,
                    "prompt_id": prompt_id,
                    "category": category,
                    "sample": sample,
                    "image_path": img_path.as_posix(),
                    "faithfulness_score": round(score, 4),
                    "checks_total": total,
                    "checks_yes": yes,
                    "checks_no": no,
                    "checks_unclear": unc,
                    "check_verdicts": json.dumps(verdicts, ensure_ascii=False),
                    "parse_error": int(parse_error),
                    "raw_response": raw[-600:],
                }
            )
            f.flush()

    print(f"[DONE] wrote {len(tasks)} rows -> {out_csv.as_posix()}")


if __name__ == "__main__":
    main()
