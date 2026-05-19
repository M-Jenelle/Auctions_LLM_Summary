import json
import os
import time
import numpy as np
from google import genai
from google.genai import types
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

_NO_THINKING = types.ThinkingConfig(thinking_budget=0)

# 50 queries × ~6 words each ≈ 300 output tokens
_QUERY_CONFIG = types.GenerateContentConfig(
    max_output_tokens=500, temperature=0.9, thinking_config=_NO_THINKING
)

# 4 ads × ~50 words each ≈ 270 output tokens
_AD_CONFIG = types.GenerateContentConfig(
    max_output_tokens=400, temperature=0.7, thinking_config=_NO_THINKING
)

# Appendix A prompt — extended to enforce <urlN> tag format for reliable parsing
AD_GEN_PROMPT = (
    "Generate two to four ads for the following query. "
    "Each ad must start with a URL tag (<url1>, <url2>, <url3>, <url4>) followed by the ad text. "
    "Each ad must be at most 40 words. The ads should be different from each other. "
    "Output only the ads, one per line, no extra commentary.\n\n"
    "Example format:\n"
    "<url1> First ad text here, at most 40 words.\n"
    "<url2> Second ad text here, at most 40 words.\n\n"
    "Original: {original}\n"
    "Ads:"
)

QUERY_GEN_PROMPT = (
    "Generate {n} diverse commercial search queries a user might type into a search engine. "
    "Vary the industries: retail, travel, health, education, finance, food, technology, home services. "
    "Return exactly one query per line, no numbering, no extra text."
)


def _call_gemini(
    prompt: str,
    model_name: str,
    config: types.GenerateContentConfig,
    retries: int = 3,
) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name, contents=prompt, config=config
            )
            return response.text.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e
    return ""


def generate_queries(n: int, model_name: str, batch_size: int = 50) -> list[str]:
    queries: list[str] = []
    while len(queries) < n:
        batch_n = min(batch_size, n - len(queries))
        text = _call_gemini(QUERY_GEN_PROMPT.format(n=batch_n), model_name, _QUERY_CONFIG)
        batch = [line.strip() for line in text.splitlines() if line.strip()]
        queries.extend(batch[:batch_n])
    return queries[:n]


def parse_ads(raw: str) -> list[str]:
    import re

    # Primary format: lines starting with <url1>, <url2>, ...
    url_lines = [l.strip() for l in raw.splitlines() if re.match(r"^<url\d+>", l.strip())]
    if len(url_lines) >= 2:
        return url_lines[:4]

    # Fallback: split on **Ad N:** or numbered "1." / "1)" headers
    chunks = re.split(r"\*\*Ad\s*\d+[:\*]*\*?\*?|\n\d+[\.\)]\s", raw)
    ads = []
    for chunk in chunks:
        # Drop description/metadata lines the model appends after the ad copy
        lines = [
            l.strip() for l in chunk.splitlines()
            if l.strip() and not re.match(r"^\*", l.strip())
        ]
        text = " ".join(lines).strip()
        if len(text.split()) >= 3:   # ignore near-empty fragments
            ads.append(text)

    return ads[:4]


def generate_ads(query: str, model_name: str, max_retries: int = 3) -> list[str]:
    for _ in range(max_retries):
        raw = _call_gemini(AD_GEN_PROMPT.format(original=query), model_name, _AD_CONFIG)
        ads = parse_ads(raw)
        if len(ads) >= 2:
            return ads
    return ads


def assign_bids_and_ctrs(
    n_ads: int, rng: np.random.Generator
) -> tuple[list[float], list[float]]:
    # Section 5.1: bids ~ LogNormal(0.5, 1), base CTRs ~ Uniform[0, 1]
    bids = rng.lognormal(mean=0.5, sigma=1.0, size=n_ads).tolist()
    ctrs = rng.uniform(0.0, 1.0, size=n_ads).tolist()
    return bids, ctrs


def generate_dataset(
    n_queries: int = 1000,
    model_name: str = "gemini-2.5-flash",
    output_path: str | None = None,
    seed: int = 42,
) -> list[dict]:
    if output_path is None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import config
        output_path = config.DATA_PATH

    rng = np.random.default_rng(seed)

    print(f"Generating {n_queries} queries with {model_name}...")
    queries = generate_queries(n_queries, model_name)
    print(f"Got {len(queries)} queries.")

    dataset: list[dict] = []
    for i, query in enumerate(queries):
        try:
            ads = generate_ads(query, model_name)
        except Exception as e:
            print(f"  [{i+1}/{n_queries}] Skipping '{query[:40]}': {e}")
            continue

        if len(ads) < 2:
            print(f"  [{i+1}/{n_queries}] Too few ads for '{query[:40]}', skipping.")
            continue

        bids, base_ctrs = assign_bids_and_ctrs(len(ads), rng)
        dataset.append({
            "query": query,
            "ads": ads,
            "bids": bids,
            "base_ctrs": base_ctrs,
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_queries} queries ({len(dataset)} kept).")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset saved to '{output_path}' ({len(dataset)} entries).")
    return dataset


if __name__ == "__main__":
    generate_dataset()
