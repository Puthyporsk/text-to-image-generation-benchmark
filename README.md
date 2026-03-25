# Text-to-Image Generation Benchmark

A multi-signal benchmark comparing **Google Gemini Imagen 4** and **OpenAI GPT Image 1 Mini** across 40 structured prompts spanning four capability categories: attribute binding, object counting, spatial reasoning, and typography.

Each model generates 3 samples per prompt at 1024x1024 resolution (240 images per model, 480 total). Models are evaluated using three independent signals:

1. **Human pairwise preference** via a Streamlit UI with blind mode
2. **VLM faithfulness judging** (Qwen2-VL-7B-Instruct) with per-check YES/NO/UNCLEAR verdicts
3. **VLM quality scoring** (same model) on a 5-dimension rubric (subject clarity, composition, technical quality, aesthetic appeal, coherence)

Typography prompts are additionally evaluated with **Tesseract OCR + fuzzy string matching**.

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Create .env with your API keys
echo GEMINI_API_KEY=your_key_here > .env
echo OPENAI_API_KEY=your_key_here >> .env
```

**Additional requirements:**
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on PATH (for OCR evaluation)
- NVIDIA GPU with ~6 GB VRAM for VLM judges (runs in 4-bit quantization via bitsandbytes)
- Set `HF_HOME` to redirect the Hugging Face model cache if your system drive is low on space

## Usage

### 1. Generate images

```bash
python run_generate.py --run_dir runs/my_run --provider gemini --prompts prompts/core40.jsonl
python run_generate.py --run_dir runs/my_run --provider chatgpt --prompts prompts/core40.jsonl
```

### 2. Run evaluations

```bash
# VLM faithfulness judge (GPU required, ~130-200s per image)
python -m eval.judge_faithfulness --run_dir runs/my_run --prompts prompts/core40.jsonl --resume

# VLM quality judge (GPU required)
python -m eval.judge_quality --run_dir runs/my_run --prompts prompts/core40.jsonl --resume

# OCR evaluation (typography prompts only)
python -m eval.ocr_eval --run_dir runs/my_run --prompts prompts/core40.jsonl
```

### 3. Human evaluation

```bash
streamlit run human_ui/app.py
```

### 4. Analyze and plot

```bash
# Analysis with statistical tests, error taxonomy, and benchmark report
python -m eval.analyze_results --extended

# Generate all plots
python -m eval.plot_results --extended
```

## Project Structure

```
├── run_generate.py              # CLI entry point for image generation
├── providers/
│   ├── base.py                  # Provider protocol and GenerationResult dataclass
│   ├── registry.py              # Provider registry and display labels
│   ├── gemini_imagen.py         # Google Gemini Imagen 4 provider
│   └── openai_image.py          # OpenAI GPT Image 1 Mini provider
├── prompts/
│   └── core40.jsonl             # 40 prompts with atomic checks (4 categories x 10)
├── eval/
│   ├── judge_faithfulness.py    # VLM faithfulness judge (per-check verdicts)
│   ├── judge_quality.py         # VLM quality judge (5-dimension rubric)
│   ├── ocr_eval.py              # OCR evaluation for typography prompts
│   ├── deep_analysis.py         # Statistical analysis engine (bootstrap, permutation tests, error taxonomy)
│   ├── analyze_results.py       # Merges all signals into unified analysis
│   ├── benchmark_report.py      # Generates Markdown benchmark report
│   └── plot_results.py          # Visualization suite (9 plot types)
├── human_ui/
│   └── app.py                   # Streamlit pairwise ranking interface
├── results/                     # Evaluation outputs (CSVs, plots, reports)
├── runs/                        # Generated images organized by run
├── report/                      # LaTeX course report and figures
└── requirements.txt
```

## Naming Conventions

- **Images:** `<prompt_id>__s<sample>.png` (e.g., `typ_003__s2.png`)
- **Runs:** `YYYY-MM-DD_core40_k3_1024`
- **Providers in paths:** `gemini` or `chatgpt`

## Key Results

- **Neither model dominates overall.** Gemini leads in human preference 53.5% vs 46.5%, but confidence intervals overlap.
- **Complementary strengths:** ChatGPT dominates spatial reasoning (92% win rate); Gemini dominates typography (72%).
- **Text rendering is the hardest capability**, accounting for >90% of all detected check failures.
- **Automated faithfulness scores predict human preference only 2.9% of the time**, confirming that prompt compliance and perceptual quality are largely orthogonal.
