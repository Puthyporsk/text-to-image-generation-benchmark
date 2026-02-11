from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Protocol, List

@dataclass
class GenerationResult:
    image_path: Path
    meta: Dict[str, Any]

class Provider(Protocol):
    name: str

    def generate_one(
        self,
        prompt_id: str,
        prompt: str,
        sample_id: int,
        out_dir: Path,
        settings: Dict[str, Any],
    ) -> GenerationResult:
        """Generate (or ingest) exactly one image for (prompt_id, sample_id)."""
        ...
