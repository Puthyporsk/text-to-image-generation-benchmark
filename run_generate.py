from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
import csv
from tqdm import tqdm

from providers.gemini_imagen import GeminiImagenProvider
from providers.chatgpt_ui import ChatGPTUIProvider

def load_prompts(jsonl_path: Path) -> List[Dict[str, Any]]:
    prompts = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts

def main(
    run_dir: str,
    prompts_file: str,
    provider_name: str,
    k: int = 3,
):
    run_path = Path(run_dir)
    if run_path.name.startswith("2026-") and run_path.parent == Path("."):
        # user passed just the run name; assume runs/<run_name>
        run_path = Path("runs") / run_path

    images_dir = run_path / "images" / ("gemini" if provider_name == "gemini" else "chatgpt")
    logs_dir = run_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(Path(prompts_file))

    if provider_name == "gemini":
        provider = GeminiImagenProvider()
    elif provider_name == "chatgpt":
        provider = ChatGPTUIProvider()
    else:
        raise ValueError("provider_name must be 'gemini' or 'chatgpt' for now")

    log_path = logs_dir / f"{provider_name}_generation_log.csv"
    write_header = not log_path.exists()

    with log_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["prompt_id", "sample_id", "image_path", "model", "provider"])
        if write_header:
            writer.writeheader()

        for p in tqdm(prompts, desc=f"provider={provider_name}"):
            prompt_id = p["prompt_id"]
            prompt_text = p["prompt"]
            for s in range(1, k + 1):
                result = provider.generate_one(
                    prompt_id=prompt_id,
                    prompt=prompt_text,
                    sample_id=s,
                    out_dir=images_dir,
                    settings={},
                )
                writer.writerow({
                    "prompt_id": prompt_id,
                    "sample_id": s,
                    "image_path": str(result.image_path),
                    "model": result.meta.get("model", ""),
                    "provider": result.meta.get("provider", provider_name),
                })

    print(f"Done. Log: {log_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--prompts_file", required=True)
    ap.add_argument("--provider", required=True, choices=["gemini", "chatgpt"])
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    main(args.run_dir, args.prompts_file, args.provider, args.k)
