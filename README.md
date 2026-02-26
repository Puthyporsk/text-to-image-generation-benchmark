# Gemini (Imagen) vs GPT Image 1 Mini: Text-to-Image Generation Benchmark

## Project structure
- `prompts/`: prompt suite in JSONL (each prompt has atomic `checks`)
- `runs/<run_name>/images/`: generated images by provider
- `eval/`: evaluation scripts (VLM faithfulness, VLM quality, OCR)
- `human_ui/`: simple UI for pairwise ranking + tags
- `results/`: aggregated outputs (CSV + plots)

## Naming conventions
- Images: `<prompt_id>__s<sample>.png`
- Runs: `YYYY-MM-DD_core40_k3_1024`
