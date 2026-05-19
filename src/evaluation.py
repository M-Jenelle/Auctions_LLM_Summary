"""
Evaluation module — Section 5.2.

Implements Definition 5.1: the synthetic CTR function used to evaluate welfare
in experiments, since real user feedback requires live production deployment.

  CTR_i(s, z) = base_ctr_i * f_i(s, z) * norm_i(s)

where:
  f_i(s, z)  = ROUGE(s_i, z_i)^β   — summary quality multiplier
  norm_i(s)  = pos_norm_i           — UI position discount (from auction result)
"""

import numpy as np
from rouge_score import rouge_scorer as rs

_scorer = rs.RougeScorer(["rouge1"], use_stemmer=True)


def compute_rouge(summary: str, original: str) -> float:
    """
    ROUGE-1 F1 score between summary and original ad text.
    Returns 0.0 for empty inputs. Capped to [0, 1].
    """
    if not summary.strip() or not original.strip():
        return 0.0
    scores = _scorer.score(original, summary)
    return float(np.clip(scores["rouge1"].fmeasure, 0.0, 1.0))


def compute_welfare(
    bids: np.ndarray,
    base_ctrs: np.ndarray,
    summaries: list[str],
    originals: list[str],
    pos_norms: np.ndarray,
    beta: float,
) -> float:
    """
    Empirical welfare (Definition 5.1):
      Welfare = Σ_i b_i * base_ctr_i * ROUGE(s_i, z_i)^β * pos_norm_i

    Ads with pos_norm = 0 (not shown) or empty summary contribute 0.
    """
    total = 0.0
    for bid, ctr, summary, original, pos_norm in zip(
        bids, base_ctrs, summaries, originals, pos_norms
    ):
        if pos_norm <= 0 or not summary.strip():
            continue
        rouge = compute_rouge(summary, original)
        total += float(bid) * float(ctr) * (rouge ** beta) * float(pos_norm)
    return total


def compute_welfare_greedy(
    bids: np.ndarray,
    base_ctrs: np.ndarray,
    pos_norms: np.ndarray,
) -> float:
    """
    Welfare for the Greedy baseline where full ads are shown (ROUGE = 1):
      Welfare = Σ_i b_i * base_ctr_i * pos_norm_i

    Beta has no effect since 1^β = 1 for any β (Section 5.5.2).
    """
    return float(np.dot(bids * base_ctrs, pos_norms))
