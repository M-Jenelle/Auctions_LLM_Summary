"""
LLM Module for Dynamic Word-Length Summary (DWLS) — Section 4.1.

Each shown ad is summarized individually (Section 4.1: "we wish to summarize
each ad separately") using few-shot Chain-of-Thought prompting.

Token optimizations (paper-compliant):
  - max_output_tokens caps each response to prevent over-generation.
  - temperature=0.4 reduces sampling variance and retry frequency.
  Prompt structure and CoT examples are unchanged from the paper.
"""

import os
import time
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "summarization_prompt.txt"
_PROMPT_TEMPLATE: str | None = None

_NO_THINKING = types.ThinkingConfig(thinking_budget=0)

# key phrases line (~15 tok) + summary line (~40 tok) + headroom; thinking disabled
_SUMM_CONFIG = types.GenerateContentConfig(
    max_output_tokens=200, temperature=0.4, thinking_config=_NO_THINKING
)


def _get_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        _PROMPT_TEMPLATE = _PROMPT_PATH.read_text()
    return _PROMPT_TEMPLATE


def _call_gemini(prompt: str, model_name: str, retries: int = 6) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name, contents=prompt, config=_SUMM_CONFIG
            )
            return response.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = min(60, 5 * (2 ** attempt))  # 5, 10, 20, 40, 60, 60 ...
                time.sleep(wait)
            else:
                raise e
    return ""


def _extract_summary(raw: str) -> str:
    """Pull the Summary line out of the CoT response."""
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("summary:"):
            return line[len("summary:"):].strip()
    # Fallback: last non-empty line
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return lines[-1] if lines else raw.strip()


def summarize_ad(ad_text: str, word_limit: int, model_name: str) -> str:
    """
    Compress a single ad to at most word_limit words using few-shot CoT.

    The LLM first extracts key phrases (intermediate reasoning step) then
    writes the summary — matching Section 4.1's prompting strategy exactly.
    """
    template = _get_prompt_template()
    prompt = template.format(ad_text=ad_text, word_limit=word_limit)
    raw = _call_gemini(prompt, model_name)
    return _extract_summary(raw)


def summarize_shown_ads(
    ads: list[str],
    prominences: np.ndarray,
    n_words: int,
    model_name: str,
    cache: dict | None = None,
) -> list[str]:
    """
    Summarize each shown ad to its allocated word count (Prom_i * n_words).

    Returns a list aligned with `ads`; not-shown ads get "".
    Results written into `cache` keyed by (ad, word_limit) to avoid
    duplicate LLM calls across (n_words, beta) combinations in experiment.py.
    """
    summaries = [""] * len(ads)

    for i, (ad, prom) in enumerate(zip(ads, prominences)):
        if prom <= 0:
            continue

        word_limit = max(5, round(float(prom) * n_words))
        cache_key = f"{ad[:80]}|{word_limit}"

        if cache is not None and cache_key in cache:
            summaries[i] = cache[cache_key]
            continue

        summary = summarize_ad(ad, word_limit, model_name)

        if cache is not None:
            cache[cache_key] = summary

        summaries[i] = summary

    return summaries
