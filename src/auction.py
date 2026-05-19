"""
Auction module for Auctions with LLM Summaries (KDD '24).

Implements three mechanisms from the paper:
  - GPAAuction : Generalized Proportional Allocation (Def 4.1, Theorem 4.2)
  - GreedyAuction : baseline that shows full-length ads until word budget runs out
  - PositionFixedLengthAuction : baseline that gives each shown ad an equal word share
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AuctionResult:
    """Output of any auction mechanism."""
    prominences: np.ndarray   # Prom_i ∈ [0,1], fraction of L allocated; 0 = not shown
    pos_norms: np.ndarray     # position discount r_t for each ad; 0 = not shown
    payments: np.ndarray      # per-click Myerson payments (0 for baselines)
    shown_order: list[int]    # indices of shown ads in display order (best position first)
    ecpms: np.ndarray         # b_i * pctr_i (base ECPM, pre-position)


def default_position_discounts(k: int, decay: float = 0.9) -> np.ndarray:
    """
    Section 5.4: norm_i(s) = 0.9^(rank_i - 1), giving [1.0, 0.9, 0.81, ...].
    """
    return decay ** np.arange(k)


# ---------------------------------------------------------------------------
# Internal helpers shared by GPA
# ---------------------------------------------------------------------------

def _select_shown(ecpms: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top min(n, k) ads by base ECPM, best first."""
    n_show = min(len(ecpms), k)
    return np.argsort(-ecpms)[:n_show]


def _gpa_allocation(
    bids: np.ndarray,
    base_ctrs: np.ndarray,
    k: int,
    position_discounts: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Core GPA step (Definition 4.1, Equation 5).

    Returns (prominences, pos_norms) for all n ads.
    """
    n = len(bids)
    ecpms = bids * base_ctrs

    shown = _select_shown(ecpms, k)

    pos_norms = np.zeros(n)
    for rank, idx in enumerate(shown):
        pos_norms[idx] = position_discounts[rank]

    # ecpm_final_i = b_i * pctr_i * pos_norm_i  (Section 4.2)
    ecpms_final = ecpms * pos_norms  # zero for not-shown

    # Prom_i = (ecpm_final_i)^alpha / Σ_j (ecpm_final_j)^alpha  (Eq. 5)
    prominences = np.zeros(n)
    if len(shown) > 0:
        powered = ecpms_final[shown] ** alpha
        total = powered.sum()
        if total > 0:
            prominences[shown] = powered / total
        else:
            prominences[shown] = 1.0 / len(shown)

    return prominences, pos_norms


def _pctr_final_gpa(
    bids: np.ndarray,
    base_ctrs: np.ndarray,
    k: int,
    position_discounts: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    pctr_final_i = pctr_i * pos_norm_i * Prom_i^β  (Section 4.2).

    This is the effective CTR the auction uses for IC checking and payment.
    """
    prominences, pos_norms = _gpa_allocation(bids, base_ctrs, k, position_discounts, alpha)
    return base_ctrs * pos_norms * (prominences ** beta)


def _myerson_payment(
    i: int,
    bids: np.ndarray,
    base_ctrs: np.ndarray,
    k: int,
    position_discounts: np.ndarray,
    alpha: float,
    beta: float,
    n_steps: int = 200,
) -> float:
    """
    Numerically integrate Myerson's Lemma (Equation 2):
      p_i(b) = b_i * pctr_final_i(x(b)) - ∫_0^{b_i} pctr_final_i(x(y, b_{-i})) dy

    The integrand is pctr_final_i evaluated while sweeping bidder i's bid from 0 to b_i,
    holding all other bids fixed. Trapezoidal rule with n_steps intervals.
    """
    b_i = bids[i]
    b_trial = bids.copy()

    ys = np.linspace(0.0, b_i, n_steps + 1)
    integrand = np.empty(n_steps + 1)

    for t, y in enumerate(ys):
        b_trial[i] = y
        integrand[t] = _pctr_final_gpa(b_trial, base_ctrs, k, position_discounts, alpha, beta)[i]

    integral = np.trapezoid(integrand, ys)
    pctr_at_bi = _pctr_final_gpa(bids, base_ctrs, k, position_discounts, alpha, beta)[i]
    return float(b_i * pctr_at_bi - integral)


# ---------------------------------------------------------------------------
# Public auction functions
# ---------------------------------------------------------------------------

def run_gpa(
    bids: np.ndarray,
    base_ctrs: np.ndarray,
    k: int,
    position_discounts: np.ndarray | None = None,
    beta: float = 0.5,
    compute_payments: bool = True,
    n_steps: int = 200,
) -> AuctionResult:
    """
    Generalized Proportional Auction (GPA) — Theorem 4.2.

    Optimal alpha is 1/(1-beta) for the CTR model f(Prom_i) = Prom_i^beta.
    Picks top min(n, k) ads by b_i*pctr_i, then allocates word fractions via
    the generalized proportional rule, and computes Myerson payments.

    Args:
        bids: per-click bids, shape (n,)
        base_ctrs: base click-through rates, shape (n,)
        k: max ads to show
        position_discounts: r_1 >= ... >= r_k; defaults to 0.9^(rank-1)
        beta: compression discount exponent in f(Prom) = Prom^beta, beta in (0,1)
        compute_payments: set False to skip payment integration (faster)
        n_steps: trapezoidal integration steps for Myerson payments
    """
    bids = np.asarray(bids, dtype=float)
    base_ctrs = np.asarray(base_ctrs, dtype=float)
    if position_discounts is None:
        position_discounts = default_position_discounts(k)
    position_discounts = np.asarray(position_discounts, dtype=float)

    # alpha = 1/(1-beta) is the welfare-maximising choice (Theorem 4.2)
    alpha = 1.0 / (1.0 - beta)

    prominences, pos_norms = _gpa_allocation(bids, base_ctrs, k, position_discounts, alpha)
    ecpms = bids * base_ctrs
    shown_order = _select_shown(ecpms, k).tolist()

    payments = np.zeros(len(bids))
    if compute_payments:
        for i in range(len(bids)):
            if prominences[i] > 0:
                payments[i] = _myerson_payment(
                    i, bids, base_ctrs, k, position_discounts, alpha, beta, n_steps
                )

    return AuctionResult(prominences, pos_norms, payments, shown_order, ecpms)


def run_greedy(
    bids: np.ndarray,
    base_ctrs: np.ndarray,
    k: int,
    n_words: int,
    ad_word_counts: list[int],
    position_discounts: np.ndarray | None = None,
) -> AuctionResult:
    """
    Greedy baseline (Section 5.3).

    Shows ads in descending ECPM order using their original full-length creatives.
    An ad is included only if its word count fits in the remaining budget.
    No LLM summarization — ads are shown as-is.

    prominence_i = ad_word_count_i / n_words for shown ads (actual fraction used).
    """
    bids = np.asarray(bids, dtype=float)
    base_ctrs = np.asarray(base_ctrs, dtype=float)
    if position_discounts is None:
        position_discounts = default_position_discounts(k)
    position_discounts = np.asarray(position_discounts, dtype=float)

    n = len(bids)
    ecpms = bids * base_ctrs
    sorted_idx = np.argsort(-ecpms)

    prominences = np.zeros(n)
    pos_norms = np.zeros(n)
    shown_order: list[int] = []
    words_left = n_words

    for idx in sorted_idx:
        if len(shown_order) >= k:
            break
        ad_words = ad_word_counts[idx]
        if ad_words <= words_left:
            rank = len(shown_order)
            prominences[idx] = ad_words / n_words
            pos_norms[idx] = position_discounts[rank]
            words_left -= ad_words
            shown_order.append(int(idx))

    return AuctionResult(prominences, pos_norms, np.zeros(n), shown_order, ecpms)


def run_position_fixed_length(
    bids: np.ndarray,
    base_ctrs: np.ndarray,
    k: int,
    position_discounts: np.ndarray | None = None,
) -> AuctionResult:
    """
    Position Auction with Fixed Length baseline (Section 5.3, "POS-FL").

    Selects top min(n, k) ads by ECPM and gives each an equal word share 1/n_show.
    Ads are summarized to equal length by the LLM module.
    """
    bids = np.asarray(bids, dtype=float)
    base_ctrs = np.asarray(base_ctrs, dtype=float)
    if position_discounts is None:
        position_discounts = default_position_discounts(k)
    position_discounts = np.asarray(position_discounts, dtype=float)

    n = len(bids)
    ecpms = bids * base_ctrs
    shown = _select_shown(ecpms, k)
    n_show = len(shown)

    prominences = np.zeros(n)
    pos_norms = np.zeros(n)
    equal_prom = 1.0 / n_show if n_show > 0 else 0.0

    for rank, idx in enumerate(shown):
        prominences[idx] = equal_prom
        pos_norms[idx] = position_discounts[rank]

    return AuctionResult(prominences, pos_norms, np.zeros(n), shown.tolist(), ecpms)
