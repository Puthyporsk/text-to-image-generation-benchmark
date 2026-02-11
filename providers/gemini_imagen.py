from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .base import GenerationResult

from PIL import Image
from io import BytesIO
import base64

@dataclass
class GeminiImagenProvider:
    name: str = "gemini"
    model: str = "imagen-4.0-generate-001"  # change later if you choose another Imagen model

    def __post_init__(self):
        load_dotenv()
        # Per Google docs, genai.Client() will use GEMINI_API_KEY from env
        # (or you can pass api_key explicitly if you prefer).
        self.client = genai.Client()

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

        # Imagen generation call (Google docs example uses models.generate_images)
        # Config supports number_of_images, aspect_ratio, image_size, etc.
        # We'll generate exactly 1 image per call for simpler logging.
        cfg = types.GenerateImagesConfig(
            number_of_images=1,
            aspect_ratio=settings.get("aspect_ratio", "1:1"),
            image_size=settings.get("image_size", "1K"),
        )

        resp = self.client.models.generate_images(
            model=self.model,
            prompt=prompt,
            config=cfg,
        )

        # Save the first image
        gi = resp.generated_images[0]

        img_data = gi.image.image_bytes  # may be base64 str OR bytes

        # 1) normalize to raw bytes
        if isinstance(img_data, str):
            raw = base64.b64decode(img_data)
        else:
            raw = img_data

        # 2) decode as an image (handles png/jpg/webp), then re-save as PNG
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
