from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from .base import GenerationResult

@dataclass
class ChatGPTUIProvider:
    name: str = "chatgpt_ui"

    def generate_one(
        self,
        prompt_id: str,
        prompt: str,
        sample_id: int,
        out_dir: Path,
        settings: Dict[str, Any],
    ) -> GenerationResult:
        """
        Ingestion mode: expects the image already exists at:
        out_dir/<prompt_id>__s<sample_id>.png
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{prompt_id}__s{sample_id}.png"
        if not out_path.exists():
            raise FileNotFoundError(
                f"Missing ChatGPT UI image: {out_path}\n"
                f"Generate in ChatGPT UI, download, and name it exactly like this."
            )

        meta = {
            "provider": self.name,
            "prompt_id": prompt_id,
            "sample_id": sample_id,
            "out_path": str(out_path),
        }
        return GenerationResult(image_path=out_path, meta=meta)
