import numpy as np
import pytest
from src.pctr_model import compute_pctr_final, compute_auction_welfare, compute_ecpm


def test_pctr_zero_for_not_shown():
    base_ctrs = np.array([0.5, 0.3, 0.4])
    pos_norms = np.array([1.0, 0.9, 0.0])   # ad 2 not shown
    prominences = np.array([0.5, 0.5, 0.0])

    result = compute_pctr_final(base_ctrs, pos_norms, prominences, beta=0.5)
    assert result[2] == 0.0


def test_pctr_monotone_in_prominence():
    """Higher prominence → higher pctr_final (Faithfulness / monotonicity)."""
    base_ctrs = np.array([0.5, 0.5])
    pos_norms = np.array([1.0, 1.0])

    res_lo = compute_pctr_final(base_ctrs, pos_norms, np.array([0.2, 0.8]), beta=0.5)
    res_hi = compute_pctr_final(base_ctrs, pos_norms, np.array([0.8, 0.2]), beta=0.5)

    assert res_hi[0] > res_lo[0]
    assert res_hi[1] < res_lo[1]


def test_pctr_formula():
    """Spot-check: pctr_final = base * pos_norm * prom^beta."""
    result = compute_pctr_final(
        np.array([0.4]),
        np.array([0.9]),
        np.array([0.5]),
        beta=0.5,
    )
    expected = 0.4 * 0.9 * (0.5 ** 0.5)
    np.testing.assert_allclose(result[0], expected, rtol=1e-9)


def test_pctr_beta_effect():
    """Larger beta means smaller compression discount for prom < 1."""
    base_ctrs = np.array([0.5])
    pos_norms = np.array([1.0])
    proms = np.array([0.4])

    # prom^beta is smaller when beta is larger (since prom < 1)
    r_small = compute_pctr_final(base_ctrs, pos_norms, proms, beta=0.25)
    r_large = compute_pctr_final(base_ctrs, pos_norms, proms, beta=0.75)
    assert r_small[0] > r_large[0]


def test_welfare_nonneg():
    bids = np.array([1.0, 0.5])
    base_ctrs = np.array([0.4, 0.3])
    pos_norms = np.array([1.0, 0.9])
    prominences = np.array([0.6, 0.4])

    w = compute_auction_welfare(bids, base_ctrs, pos_norms, prominences, beta=0.5)
    assert w >= 0.0


def test_welfare_zero_nothing_shown():
    bids = np.array([1.0, 0.5])
    base_ctrs = np.array([0.4, 0.3])
    pos_norms = np.zeros(2)
    prominences = np.zeros(2)

    w = compute_auction_welfare(bids, base_ctrs, pos_norms, prominences, beta=0.5)
    assert w == 0.0


def test_ecpm():
    bids = np.array([1.0, 2.0])
    ctrs = np.array([0.5, 0.3])
    np.testing.assert_allclose(compute_ecpm(bids, ctrs), [0.5, 0.6])
