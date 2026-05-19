"""
pCTR Module — Section 2.3 and Section 4.2.

Implements the factorized click-through rate model used by the auction:
  pctr_final_i = base_ctr_i * pos_norm_i * Prom_i^β

This satisfies Assumption 2.3 (unbiased estimation) under the DWLS setting,
where the LLM faithfully produces summaries at the allocated word length and
ROUGE ≈ Prom (Section 5.2 evaluation model).
"""

import numpy as np


def compute_pctr_final(
    base_ctrs: np.ndarray,
    pos_norms: np.ndarray,
    prominences: np.ndarray,
    beta: float,
) -> np.ndarray:
    """
    Factorized pCTR (Section 4.2):
      pctr_final_i = base_ctr_i * pos_norm_i * Prom_i^β

    For not-shown ads (prominence = 0), returns 0 without numeric warnings.
    """
    prom_discount = np.where(prominences > 0, prominences ** beta, 0.0)
    return base_ctrs * pos_norms * prom_discount


def compute_auction_welfare(
    bids: np.ndarray,
    base_ctrs: np.ndarray,
    pos_norms: np.ndarray,
    prominences: np.ndarray,
    beta: float,
) -> float:
    """
    Expected welfare under the pCTR model (Equation 3 / Proposition 3.6):
      Welfare = Σ_i b_i * pctr_final_i
    """
    pctr_finals = compute_pctr_final(base_ctrs, pos_norms, prominences, beta)
    return float(np.dot(bids, pctr_finals))


def compute_ecpm(bids: np.ndarray, base_ctrs: np.ndarray) -> np.ndarray:
    """Base ECPM: b_i * pctr_i (pre-position, pre-prominence)."""
    return bids * base_ctrs
