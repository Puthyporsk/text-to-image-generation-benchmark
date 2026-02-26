from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


JUDGE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
ALLOWED_PROVIDERS = {"gemini", "chatgpt"}

SYSTEM_INSTR = (
    "You are a strict evaluation judge for text-to-image outputs. "
    "You MUST output valid JSON only. No extra text."
)

def normalize_run_dir(run_dir: str | Path) -> Path:
    # Required by your conventions:
    # run_dir = Path(run_dir); if no parent OR does not include 'runs', rewrite to Path('runs')/run_dir
    p = Path(run_dir)
    if p.parent == Path(".") or "runs" not in p.parts:
        p = Path("runs") / p
    return p

def enforce_run_dir(run_dir: str | Path) -> Dict[str, Path]:
    rd = normalize_run_dir(run_dir)

    # MUST print resolved run_dir before running anything
    print(f"[RUN] resolved run_dir = {rd.as_posix()}")

    # MUST confirm it starts with "runs/"
    if len(rd.parts) == 0 or rd.parts[0] != "runs":
        corrected = (Path("runs") / rd).as_posix()
        raise SystemExit(
            f"Refusing: run_dir must start with 'runs/'.\n"
            f"Corrected command:\n"
            f"  python -m eval.vlm_judge --run_dir {corrected} --prompts prompts/core40.jsonl"
        )

    # required structure
    images_gemini = rd / "images" / "gemini"
    images_chatgpt = rd / "images" / "chatgpt"
    logs_dir = rd / "logs"
    rd.mkdir(parents=True, exist_ok=True)
    images_gemini.mkdir(parents=True, exist_ok=True)
    images_chatgpt.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": rd,
        "images_gemini": images_gemini,
        "images_chatgpt": images_chatgpt,
        "logs_dir": logs_dir,
        "manifest": rd / "manifest.json",
    }

def parse_prompt_file(prompts_path: Path) -> Dict[str, Dict[str, Any]]:
    # expects prompts/core40.jsonl with fields:
    # - id (prompt_id)
    # - prompt (text)
    # - checks[] (judge requirements)
    # - typography eval_targets.text[] (as you described)
    by_id: Dict[str, Dict[str, Any]] = {}
    with prompts_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            pid = str(item.get("id"))
            by_id[pid] = item
    return by_id

def enforce_filename_convention(path: Path) -> tuple[str, int]:
    # MUST be: <prompt_id>__s<sample>.png (sample starts at 1)
    stem = path.stem  # e.g. "core40_12__s1"
    if "__s" not in stem:
        raise ValueError(f"Bad filename (missing '__s'): {path.name}")
    prompt_id, s = stem.rsplit("__s", 1)
    try:
        sample = int(s)
    except ValueError:
        raise ValueError(f"Bad filename (sample not int): {path.name}")
    if sample < 1:
        raise ValueError(f"Bad filename (sample must start at 1): {path.name}")
    if path.suffix.lower() != ".png":
        raise ValueError(f"Bad filename (must be .png): {path.name}")
    return prompt_id, sample

def validate_image_write_location(run_paths: Dict[str, Path], provider: str, out_path: Path) -> None:
    # Required: validate that images are only written under runs/<run>/images/<provider>/
    provider = provider.lower().strip()
    if provider not in ALLOWED_PROVIDERS:
        raise ValueError(f"Invalid provider '{provider}'. Must be one of {sorted(ALLOWED_PROVIDERS)}")
    allowed_root = (run_paths["run_dir"] / "images" / provider).resolve()
    out_resolved = out_path.resolve()
    if allowed_root not in out_resolved.parents:
        raise ValueError(
            f"Refusing: image output must be under {allowed_root.as_posix()}\nGot: {out_path.as_posix()}"
        )

def build_eval_prompt(item: Dict[str, Any], prompt_id: str) -> str:
    prompt_text = item.get("prompt", "")
    checks = item.get("checks", [])
    # You said: typography eval_targets.text[]
    target_texts = []
    et = item.get("eval_targets", {})
    if isinstance(et, dict):
        # allow either eval_targets.text[] OR eval_targets.typography.text[]
        if isinstance(et.get("text"), list):
            target_texts = et.get("text")
        elif isinstance(et.get("typography"), dict) and isinstance(et["typography"].get("text"), list):
            target_texts = et["typography"]["text"]

    schema = {
        "prompt_id": prompt_id,
        "faithfulness": {
            "overall": "number 0..1",
            "checks": [
                {"name": "string", "pass": "bool", "confidence": "0..1", "evidence": "string"}
            ],
            "hallucinations": ["string"],
            "notes": "string"
        },
        "typography": {
            "targets": [
                {"text": "string", "present": "bool", "confidence": "0..1", "evidence": "string"}
            ]
        }
    }

    return (
        f"Evaluate the image against the prompt and requirements.\n\n"
        f"PROMPT_ID: {prompt_id}\n"
        f"PROMPT: {prompt_text}\n\n"
        f"REQUIREMENTS (checks):\n{json.dumps(checks, ensure_ascii=False, indent=2)}\n\n"
        f"TYPOGRAPHY_TARGET_TEXT (exact strings that should appear, if any):\n"
        f"{json.dumps(target_texts, ensure_ascii=False, indent=2)}\n\n"
        f"Return JSON ONLY with this schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"Rules:\n"
        f"- Be conservative: if unsure, pass=false and lower confidence.\n"
        f"- overall = passed_checks/total_checks (if no checks, overall=1.0).\n"
        f"- Output JSON only.\n"
    )

@torch.inference_mode()
def judge_one(processor, model, image: Image.Image, eval_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_INSTR},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": eval_prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=700,
        do_sample=False,
        temperature=0.0,
    )
    out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    # defensive JSON extraction (still must be JSON-only ideally)
    start = out_text.find("{")
    end = out_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        out_text = out_text[start:end+1]
    return out_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Must be runs/<run_name> (or run_name; will normalize to runs/<run_name>)")
    ap.add_argument("--prompts", required=True, help="prompts/core40.jsonl")
    ap.add_argument("--judge_out", default=None, help="Optional override for judge JSONL output (default: runs/<run>/logs/vlm_judge.jsonl)")
    args = ap.parse_args()

    run_paths = enforce_run_dir(args.run_dir)

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        raise SystemExit(f"Prompts not found: {prompts_path.as_posix()}")

    # Output locations:
    # - judge JSONL logs: runs/<run>/logs/
    judge_out = Path(args.judge_out) if args.judge_out else (run_paths["logs_dir"] / "vlm_judge_qwen2vl2b.jsonl")

    # sanity: refuse writing judge logs outside this run (it’s not CSV, but keep it tidy)
    if run_paths["run_dir"].resolve() not in judge_out.resolve().parents:
        raise SystemExit(
            f"Refusing: judge_out must be inside {run_paths['run_dir'].as_posix()} (recommend logs/).\n"
            f"Got: {judge_out.as_posix()}"
        )

    by_id = parse_prompt_file(prompts_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENV] torch.cuda.is_available() = {torch.cuda.is_available()}")

    processor = AutoProcessor.from_pretrained(JUDGE_MODEL)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        JUDGE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()

    image_dirs = {
        "gemini": run_paths["images_gemini"],
        "chatgpt": run_paths["images_chatgpt"],
    }

    total = 0
    with judge_out.open("w", encoding="utf-8") as f_out:
        for provider, img_dir in image_dirs.items():
            if not img_dir.exists():
                continue

            # only consume PNGs matching <prompt_id>__s<sample>.png
            for img_path in sorted(img_dir.glob("*.png")):
                prompt_id, sample = enforce_filename_convention(img_path)

                if prompt_id not in by_id:
                    # if your prompt ids differ (e.g. numeric vs "core40_12"), you’ll see it here
                    continue

                item = by_id[prompt_id]
                eval_prompt = build_eval_prompt(item, prompt_id)

                image = Image.open(img_path).convert("RGB")

                raw = judge_one(processor, model, image, eval_prompt)
                record = {
                    "run_dir": run_paths["run_dir"].as_posix(),
                    "provider": provider,
                    "prompt_id": prompt_id,
                    "sample": sample,
                    "image_path": img_path.as_posix(),
                    "judge_model": JUDGE_MODEL,
                    "raw": raw,
                    "parsed": None,
                }
                try:
                    record["parsed"] = json.loads(raw)
                except Exception:
                    record["parsed"] = None

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1

    print(f"[DONE] wrote judge JSONL = {judge_out.as_posix()}")
    print(f"[DONE] judged images = {total}")

if __name__ == "__main__":
    main()
