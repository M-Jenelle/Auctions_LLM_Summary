"""
Full experiment pipeline — Section 5 of the paper.

Replicates Figure 2: average welfare vs. total word budget for three mechanisms
(GPA+LLM, Greedy, POS-FL) across β ∈ {1/2, 1/3, 1/4}.

Usage:
  python src/experiment.py              # run on full dataset
  python src/experiment.py 50           # run on first 50 queries (fast test)
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.auction import run_gpa, run_greedy, run_position_fixed_length, default_position_discounts
from src.llm_summarizer import summarize_shown_ads
from src.evaluation import compute_welfare, compute_welfare_greedy


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run data generation first:  python src/data_gen.py"
        )
    with open(path) as f:
        return json.load(f)


def load_cache(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)


def _key_to_str(key: tuple) -> str:
    mech, beta, n_words = key
    return f"{mech}|{beta}|{n_words}"


def _str_to_key(s: str) -> tuple:
    if "|" in s:
        parts = s.split("|")
        return (parts[0], float(parts[1]), int(parts[2]))
    # Legacy format: "('GPA+LLM', 0.5, 30)"
    import ast
    t = ast.literal_eval(s)
    return (t[0], float(t[1]), int(t[2]))


def save_results(results: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serializable = {_key_to_str(k): v for k, v in results.items()}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_results(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        raw = json.load(f)
    return {_str_to_key(k): v for k, v in raw.items()}


def merge_results(existing: dict, new: dict) -> dict:
    merged = dict(existing)
    for key, vals in new.items():
        merged[key] = merged.get(key, []) + vals
    return merged


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    dataset: list[dict],
    n_words_list: list[int],
    beta_list: list[float],
    k: int,
    model_name: str,
    max_queries: int | None = None,
    cache: dict | None = None,
) -> dict:
    """
    For each query, run GPA+LLM, Greedy, and POS-FL across all (β, n_words).

    Returns dict: (mechanism, beta, n_words) -> list[float] of per-query welfare.
    """
    if max_queries is not None:
        dataset = dataset[:max_queries]
    if cache is None:
        cache = {}

    pos_disc = default_position_discounts(k)
    results: dict = defaultdict(list)

    for q_idx, entry in enumerate(tqdm(dataset, desc="Queries")):
        ads: list[str] = entry["ads"]
        bids = np.array(entry["bids"])
        base_ctrs = np.array(entry["base_ctrs"])
        ad_word_counts = [len(ad.split()) for ad in ads]

        # Pre-compute auction allocations once per query (they don't depend on n_words).
        p_res = run_position_fixed_length(bids, base_ctrs, k, pos_disc)
        gpa_res_by_beta = {
            beta: run_gpa(bids, base_ctrs, k, pos_disc, beta, compute_payments=False)
            for beta in beta_list
        }

        for n_words in n_words_list:

            # ── Greedy ────────────────────────────────────────────────────
            g_res = run_greedy(bids, base_ctrs, k, n_words, ad_word_counts, pos_disc)
            g_welfare = compute_welfare_greedy(bids, base_ctrs, g_res.pos_norms)
            for beta in beta_list:
                results[("Greedy", beta, n_words)].append(g_welfare)

            # ── POS-FL ────────────────────────────────────────────────────
            p_summaries = summarize_shown_ads(ads, p_res.prominences, n_words, model_name, cache)
            for beta in beta_list:
                w = compute_welfare(bids, base_ctrs, p_summaries, ads, p_res.pos_norms, beta)
                results[("POS-FL", beta, n_words)].append(w)

            # ── GPA + LLM ─────────────────────────────────────────────────
            for beta in beta_list:
                gpa_res = gpa_res_by_beta[beta]
                gpa_summaries = summarize_shown_ads(
                    ads, gpa_res.prominences, n_words, model_name, cache
                )
                w = compute_welfare(bids, base_ctrs, gpa_summaries, ads, gpa_res.pos_norms, beta)
                results[("GPA+LLM", beta, n_words)].append(w)

        # Save cache every 50 queries so progress survives a crash.
        if cache is not None and (q_idx + 1) % 50 == 0:
            save_cache(cache, config.CACHE_PATH)

    return dict(results)


# ---------------------------------------------------------------------------
# Plotting — Figure 2
# ---------------------------------------------------------------------------

def plot_figure2(
    results: dict,
    beta_list: list[float],
    n_words_list: list[int],
    output_dir: str,
) -> None:
    """Replicate Figure 2: welfare vs. total word budget, one subplot per β."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    beta_labels = {0.5: "β=1/2", 1 / 3: "β=1/3", 0.25: "β=1/4"}
    styles = {
        "GPA+LLM": ("o-", "tab:blue"),
        "Greedy":  ("s--", "tab:orange"),
        "POS-FL":  ("^-.", "tab:green"),
    }

    fig, axes = plt.subplots(1, len(beta_list), figsize=(5 * len(beta_list), 4), sharey=False)
    if len(beta_list) == 1:
        axes = [axes]

    for ax, beta in zip(axes, beta_list):
        for mech, (style, color) in styles.items():
            welfare_means = [
                np.mean(results.get((mech, beta, nw), [0.0]))
                for nw in n_words_list
            ]
            ax.plot(n_words_list, welfare_means, style, label=mech, color=color, markersize=5)

        ax.set_xlabel("Total # of Words")
        ax.set_ylabel("Avg Welfare / Query")
        ax.set_title(beta_labels.get(beta, f"β={beta:.3f}"))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir) / "figure2_welfare.png"
    plt.savefig(out_path, dpi=150)
    print(f"Figure saved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Usage:
    #   python src/experiment.py              → auto-resume from last checkpoint
    #   python src/experiment.py 100          → run first 100 queries (from 0)
    #   python src/experiment.py 0 100        → run queries 0–99 explicitly
    #   python src/experiment.py 100 200      → run queries 100–199 explicitly

    dataset  = load_dataset(config.DATA_PATH)
    cache    = load_cache(config.CACHE_PATH)
    existing = load_results(config.RESULTS_PATH)

    already_done = len(next(iter(existing.values()), [])) if existing else 0

    if len(sys.argv) == 3:
        start_idx = int(sys.argv[1])
        end_idx   = int(sys.argv[2])
    elif len(sys.argv) == 2:
        start_idx = 0
        end_idx   = int(sys.argv[1])
    else:
        # Auto-resume: pick up exactly where the last run stopped
        start_idx = already_done
        end_idx   = config.N_QUERIES

    if start_idx >= config.N_QUERIES:
        print(f"All {len(dataset)} queries already processed. Nothing to do.")
        plot_figure2(existing, config.BETA_LIST, config.N_WORDS_LIST, config.FIGURES_DIR)
        sys.exit(0)

    print(f"Dataset cap: {config.N_QUERIES}  |  already done: {already_done}  |  running: [{start_idx}, {end_idx})")

    new_results = run_experiment(
        dataset=dataset[start_idx:end_idx],
        n_words_list=config.N_WORDS_LIST,
        beta_list=config.BETA_LIST,
        k=config.K,
        model_name=config.MODEL_NAME,
        cache=cache,
    )

    save_cache(cache, config.CACHE_PATH)

    merged = merge_results(existing, new_results)
    save_results(merged, config.RESULTS_PATH)

    plot_figure2(merged, config.BETA_LIST, config.N_WORDS_LIST, config.FIGURES_DIR)
    total = len(next(iter(merged.values()), []))
    print(f"Done. Total queries in results: {total} / {len(dataset)}")
