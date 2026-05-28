"""
LLM Module for Dynamic Word-Length Summary (DWLS) — Section 4.1.

Each shown ad is summarized individually (Section 4.1: "we wish to summarize
each ad separately") using few-shot Chain-of-Thought prompting.

Supports two backends selected by model_name:
  - Gemini  : any "gemini-*" model name
  - HuggingFace: any HF model ID (e.g. "mistralai/Mistral-7B-Instruct-v0.3")

Token optimizations (paper-compliant):
  - max_output_tokens caps each response to prevent over-generation.
  - temperature=0.4 reduces sampling variance and retry frequency.
  Prompt structure and CoT examples are unchanged from the paper.
"""

import os
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "summarization_prompt.txt"
_PROMPT_TEMPLATE: str | None = None

# ── Gemini client (lazy-initialised) ─────────────────────────────────────────
_gemini_client = None

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gemini_client

# ── HuggingFace client (lazy-initialised) ────────────────────────────────────
_hf_client = None

def _get_hf_client():
    global _hf_client
    if _hf_client is None:
        from huggingface_hub import InferenceClient
        _hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))
    return _hf_client

# ── Mistral AI client (lazy-initialised) ─────────────────────────────────────
_mistral_client = None

def _get_mistral_client():
    global _mistral_client
    if _mistral_client is None:
        from mistralai import Mistral
        _mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    return _mistral_client

# ── Anthropic client (lazy-initialised) ──────────────────────────────────────
_anthropic_client = None

def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _anthropic_client

# ── OpenAI client (lazy-initialised) ─────────────────────────────────────────
_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _get_prompt_template() -> str:
    global _PROMPT_TEMPLATE
    if _PROMPT_TEMPLATE is None:
        _PROMPT_TEMPLATE = _PROMPT_PATH.read_text()
    return _PROMPT_TEMPLATE


def _is_gemini(model_name: str) -> bool:
    return model_name.startswith("gemini")

def _is_claude(model_name: str) -> bool:
    return model_name.startswith("claude")

def _is_gpt(model_name: str) -> bool:
    return model_name.startswith("gpt") or model_name.startswith("o1") or model_name.startswith("o3")

def _is_hf(model_name: str) -> bool:
    return "/" in model_name  # HF model IDs always contain a slash


def _call_gemini(prompt: str, model_name: str, retries: int = 6) -> str:
    from google.genai import types
    no_thinking = types.ThinkingConfig(thinking_budget=0)
    config = types.GenerateContentConfig(
        max_output_tokens=200, temperature=0.4, thinking_config=no_thinking
    )
    client = _get_gemini_client()
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name, contents=prompt, config=config
            )
            return response.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = min(60, 5 * (2 ** attempt))
                time.sleep(wait)
            else:
                raise e
    return ""


def _format_instruct(prompt: str, model_name: str) -> str:
    """Wrap prompt in the model's instruct template."""
    name = model_name.lower()
    if "mistral" in name or "mixtral" in name:
        return f"<s>[INST] {prompt} [/INST]"
    if "llama" in name:
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    if "qwen" in name:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    # Generic fallback
    return f"### User:\n{prompt}\n### Assistant:\n"


def _call_hf(prompt: str, model_name: str, retries: int = 6) -> str:
    client = _get_hf_client()
    for attempt in range(retries):
        try:
            response = client.chat_completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.4,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = min(60, 5 * (2 ** attempt))
                time.sleep(wait)
            else:
                raise e
    return ""


def _call_mistral(prompt: str, model_name: str, retries: int = 6) -> str:
    client = _get_mistral_client()
    for attempt in range(retries):
        try:
            response = client.chat.complete(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.4,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = min(60, 5 * (2 ** attempt))
                time.sleep(wait)
            else:
                raise e
    return ""


def _call_claude(prompt: str, model_name: str, retries: int = 6) -> str:
    client = _get_anthropic_client()
    for attempt in range(retries):
        try:
            message = client.messages.create(
                model=model_name,
                max_tokens=200,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = min(60, 5 * (2 ** attempt))
                time.sleep(wait)
            else:
                raise e
    return ""


def _call_gpt(prompt: str, model_name: str, retries: int = 6) -> str:
    client = _get_openai_client()
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.4,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                wait = min(60, 5 * (2 ** attempt))
                time.sleep(wait)
            else:
                raise e
    return ""


def _call_llm(prompt: str, model_name: str) -> str:
    if _is_gemini(model_name):
        return _call_gemini(prompt, model_name)
    if _is_claude(model_name):
        return _call_claude(prompt, model_name)
    if _is_gpt(model_name):
        return _call_gpt(prompt, model_name)
    if _is_hf(model_name):
        return _call_hf(prompt, model_name)
    return _call_mistral(prompt, model_name)


def _extract_summary(raw: str) -> str:
    """Pull the Summary line out of the CoT response."""
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("summary:"):
            return line[len("summary:"):].strip()
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
    raw = _call_llm(prompt, model_name)
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
