# Auctions with LLM Summaries

Implementation of the factorized auction framework from:
> *Auctions with LLM Summaries*, Dubey et al., KDD 2024.

Extended with multi-model support to compare summarization backends: **Gemini**, **Claude**, **GPT**, **HuggingFace**, and **Mistral**.

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure API keys**

Add to `.env` in this directory:
```
GEMINI_API_KEY=your_key_here        # https://aistudio.google.com/app/apikey
ANTHROPIC_API_KEY=your_key_here     # https://console.anthropic.com/
OPENAI_API_KEY=your_key_here        # https://platform.openai.com/
HF_TOKEN=your_key_here              # https://huggingface.co/settings/tokens
MISTRAL_API_KEY=your_key_here       # https://console.mistral.ai/
```

Only the key for your chosen model is required.

**3. Set your model in `config.py`**

```python
# Gemini
MODEL_NAME = "gemini-2.5-flash"

# Claude
MODEL_NAME = "claude-haiku-4-5-20251001"

# GPT
MODEL_NAME = "gpt-4o-mini"

# HuggingFace (model ID with slash)
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# Mistral AI
MODEL_NAME = "open-mistral-7b"
```

Each model automatically saves results and cache to separate files — no overwriting between runs.

---

## Running the experiment

**Step 1 — Generate the dataset** (1000 synthetic queries + ads via Gemini)
```bash
python src/data_gen.py
# Output: results/dataset.json
```

**Step 2 — Run the experiment**
```bash
python src/experiment.py          # auto-resume from last checkpoint
python src/experiment.py 50       # fast test on first 50 queries
```

Use `caffeinate` on macOS to prevent sleep during long runs:
```bash
caffeinate -i python src/experiment.py
```

**Auto-resume**: the experiment tracks how many queries are done and picks up exactly where it left off on restart. Cache and results checkpoint every 10 queries.

**Running in batches** (optional manual control)
```bash
python src/experiment.py 0 50     # queries 0–49   (batch 1)
python src/experiment.py 50 100   # queries 50–99  (batch 2)
python src/experiment.py 100 150  # queries 100–149 (batch 3)
# ... continue in increments of 50 up to N_QUERIES
```

**Output files** (per model, no overwriting):
```
results/welfare_results_<model>.json   — welfare numbers
results/summary_cache_<model>.json     — LLM summary cache
results/figures/welfare_<model>.png    — Figure 2 plot
```

**Run tests**
```bash
pytest tests/ -v
```

---

## Results

| Model | Result file | Figure |
|---|---|---|
| Gemini 2.5 Flash | `results/results_gemini.md` | `figures/welfare_gemini-2-5-flash.png` |
| Claude Haiku 4.5 | `results/results_claude.md` | `figures/welfare_claude-haiku-4-5-20251001.png` |
| GPT-4o Mini | `results/results_gpt.md` | `figures/welfare_gpt-4o-mini.png` |

---

## Architecture

```
src/
  data_gen.py       — LLM-based query + ad generation (Section 5.1, Appendix A)
  auction.py        — GPA, Greedy, POS-FL mechanisms (Section 4.2)
  pctr_model.py     — Factorized pCTR: base_ctr × pos_norm × Prom^β (Section 4.2)
  llm_summarizer.py — Few-shot CoT ad compression, multi-model backend (Section 4.1)
  evaluation.py     — ROUGE-based welfare (Definition 5.1)
  experiment.py     — Full pipeline + Figure 2 plot, auto-resume (Section 5)

prompts/
  data_gen_prompt.txt       — Appendix A prompt
  summarization_prompt.txt  — 3-shot CoT summarization prompt

results/
  dataset.json              — 1000 synthetic queries
  results_gemini.md         — Gemini welfare summary
  results_claude.md         — Claude welfare summary
  results_gpt.md            — GPT welfare summary
  figures/                  — Per-model welfare plots

config.py   — MODEL_NAME, β list, word budget list, paths
```

---

## Key parameters (`config.py`)

| Parameter | Value | Paper reference |
|---|---|---|
| `MODEL_NAME` | see above | summarization backend |
| `K` | 4 | max ads shown |
| `BETA_LIST` | [0.5, 0.333, 0.25] | β ∈ {1/2, 1/3, 1/4}, Section 5.4 |
| `N_WORDS_LIST` | [30, 40, 50, 60, 70, 80] | Figure 2 x-axis |
| `POSITION_DECAY` | 0.9 | norm_i = 0.9^(rank-1), Section 5.4 |
| `N_QUERIES` | 500 | queries to evaluate |

The optimal GPA parameter is `α = 1/(1−β)` (Theorem 4.2).
