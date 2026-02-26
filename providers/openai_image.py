from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from .base import GenerationResult


@dataclass
class OpenAIImageProvider:
    name: str = "chatgpt"
    model: str = "gpt-image-1-mini"

    def __post_init__(self):
        load_dotenv()
        # Reads OPENAI_API_KEY from environment / .env automatically
        self.client = OpenAI()

    def generate_one(
        self,
        prompt_id: str,
        prompt: str,
        sample_id: int,
        out_dir: Path,
        settings: Dict[str, Any],
    ) -> GenerationResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{prompt_id}__s{sample_id}.png"

        # gpt-image-1-mini always returns b64 — no response_format param.
        # Quality accepts: low | medium | high | auto
        response = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size=settings.get("size", "1024x1024"),
            quality=settings.get("quality", "medium"),
            n=1,
        )

        img_b64 = response.data[0].b64_json
        raw = base64.b64decode(img_b64)
        img = Image.open(BytesIO(raw))
        img.save(out_path, format="PNG")

        meta = {
            "provider": self.name,
            "model": self.model,
            "prompt_id": prompt_id,
            "sample_id": sample_id,
            "out_path": str(out_path),
        }
        return GenerationResult(image_path=out_path, meta=meta)
