# Auctions with LLM Summaries

Implementation of the factorized auction framework from:
> *Auctions with LLM Summaries*, Dubey et al., KDD 2024.

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set your Gemini API key**

Add to `.env` in this directory:
```
GEMINI_API_KEY=your_key_here
```
Get a key at https://aistudio.google.com/app/apikey

## Running the experiment

**Step 1 — Generate the dataset** (1000 synthetic queries + ads via Gemini)
```bash
python src/data_gen.py
# Output: results/dataset.json
```

**Step 2 — Run the experiment** (all three mechanisms across β and word budgets)
```bash
python src/experiment.py          # full 1000-query run
python src/experiment.py 50       # fast test on first 50 queries
# Output: results/welfare_results.json
#         results/figures/figure2_welfare.png
```

**Running in batches** (recommended for large runs)

Each batch merges into the same results file and reuses the LLM cache, so interrupted runs can be resumed without redoing work:
```bash
python src/experiment.py 0 50     # queries 0–49   (batch 1)
python src/experiment.py 50 100   # queries 50–99  (batch 2)
python src/experiment.py 100 150  # queries 100–149 (batch 3)
# ... continue in increments of 50 up to 1000
```

**Run tests**
```bash
pytest tests/ -v
```

## Architecture

```
src/
  data_gen.py       — LLM-based query + ad generation (Section 5.1, Appendix A)
  auction.py        — GPA, Greedy, POS-FL mechanisms (Section 4.2)
  pctr_model.py     — Factorized pCTR: base_ctr × pos_norm × Prom^β (Section 4.2)
  llm_summarizer.py — Few-shot CoT ad compression to word limit (Section 4.1)
  evaluation.py     — ROUGE-based welfare (Definition 5.1)
  experiment.py     — Full pipeline + Figure 2 plot (Section 5)

prompts/
  data_gen_prompt.txt       — Appendix A prompt
  summarization_prompt.txt  — 3-shot CoT summarization prompt

config.py   — k, β list, word budget list, paths
```

## Key parameters (`config.py`)

| Parameter | Value | Paper reference |
|---|---|---|
| `K` | 4 | max ads shown |
| `BETA_LIST` | [0.5, 0.333, 0.25] | β ∈ {1/2, 1/3, 1/4}, Section 5.4 |
| `N_WORDS_LIST` | [30, 40, 50, 60, 70, 80] | Figure 2 x-axis |
| `POSITION_DECAY` | 0.9 | norm_i = 0.9^(rank-1), Section 5.4 |
| `N_QUERIES` | 1000 | Section 5.4 |

The optimal GPA parameter is `α = 1/(1−β)` (Theorem 4.2).
