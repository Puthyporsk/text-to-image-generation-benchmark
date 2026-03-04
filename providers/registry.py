"""
providers/registry.py
---------------------
Single source of truth for registered providers.

To add a new model:
  1. Create providers/<your_model>.py implementing the Provider protocol
  2. Add an entry to LABELS and the get_provider() function below
"""
from __future__ import annotations

# Display labels shown in the UI and charts.
# Key = provider name used in filesystem paths and CSV columns.
LABELS: dict[str, str] = {
    "gemini":  "Gemini (Imagen 4)",
    "chatgpt": "GPT-Image-1 Mini",
}


def get_provider(name: str):
    """Instantiate and return a provider by name."""
    if name == "gemini":
        from providers.gemini_imagen import GeminiImagenProvider
        return GeminiImagenProvider()
    if name == "chatgpt":
        from providers.openai_image import OpenAIImageProvider
        return OpenAIImageProvider()
    raise ValueError(
        f"Unknown provider '{name}'. Registered: {list(LABELS.keys())}"
    )


def registered_names() -> list[str]:
    return list(LABELS.keys())


def label(name: str) -> str:
    """Return the display label for a provider, falling back to the name itself."""
    return LABELS.get(name, name)
