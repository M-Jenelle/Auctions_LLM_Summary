import numpy as np
import pytest
from src.auction import (
    run_gpa,
    run_greedy,
    run_position_fixed_length,
    default_position_discounts,
)


@pytest.fixture
def three_ads():
    return {
        "bids": np.array([0.645, 0.641, 0.617]),
        "base_ctrs": np.ones(3),
        "k": 3,
        "position_discounts": default_position_discounts(3),
    }


# ── GPA allocation ─────────────────────────────────────────────────────────

def test_gpa_table2_prominences(three_ads):
    """Table 2 Summary A: ecpms=[0.645,0.641,0.617], α=2 (β=0.5), pos_disc=[1,0.9,0.81]
    → prominences [0.417, 0.333, 0.250] (verified analytically in the paper)."""
    res = run_gpa(**three_ads, beta=0.5, compute_payments=False)
    np.testing.assert_allclose(res.prominences, [0.417, 0.333, 0.250], atol=0.01)


def test_gpa_prominences_sum_to_one(three_ads):
    res = run_gpa(**{k: v for k, v in three_ads.items()}, beta=0.5, compute_payments=False)
    assert abs(res.prominences.sum() - 1.0) < 1e-9


def test_gpa_monotone_allocation():
    """Higher bid must give higher or equal prominence (Definition 3.1)."""
    base_ctrs = np.array([0.5, 0.5, 0.5])
    pos_disc = default_position_discounts(3)
    bids_lo = np.array([0.5, 0.8, 0.6])
    bids_hi = np.array([1.5, 0.8, 0.6])  # bidder 0 raises bid

    res_lo = run_gpa(bids_lo, base_ctrs, 3, pos_disc, beta=0.5, compute_payments=False)
    res_hi = run_gpa(bids_hi, base_ctrs, 3, pos_disc, beta=0.5, compute_payments=False)

    assert res_hi.prominences[0] >= res_lo.prominences[0] - 1e-9


def test_gpa_k_limits_shown_ads():
    bids = np.array([1.0, 0.8, 0.6, 0.4])
    base_ctrs = np.ones(4)
    res = run_gpa(bids, base_ctrs, k=2, position_discounts=default_position_discounts(2),
                  beta=0.5, compute_payments=False)
    assert len(res.shown_order) <= 2
    assert (res.prominences > 0).sum() <= 2


def test_gpa_payments_nonneg(three_ads):
    """Myerson payments must be non-negative (IC + IR)."""
    res = run_gpa(**{k: v for k, v in three_ads.items()},
                  beta=0.5, compute_payments=True, n_steps=50)
    assert all(p >= -1e-6 for p in res.payments)


def test_gpa_single_ad():
    """Single ad always gets full prominence."""
    res = run_gpa(np.array([1.0]), np.array([0.5]), k=2,
                  position_discounts=default_position_discounts(2),
                  beta=0.5, compute_payments=False)
    assert abs(res.prominences[0] - 1.0) < 1e-9


# ── Greedy baseline ────────────────────────────────────────────────────────

def test_greedy_respects_word_budget():
    bids = np.array([1.0, 0.8, 0.6])
    res = run_greedy(bids, np.ones(3), k=3, n_words=40,
                     ad_word_counts=[25, 20, 15],
                     position_discounts=default_position_discounts(3))
    total_words = sum([25, 20, 15][i] for i in res.shown_order)
    assert total_words <= 40


def test_greedy_skips_oversized_ad():
    """An ad larger than the remaining budget must be skipped."""
    bids = np.array([1.0, 0.8])
    res = run_greedy(bids, np.ones(2), k=2, n_words=15,
                     ad_word_counts=[20, 10],  # ad 0 doesn't fit
                     position_discounts=default_position_discounts(2))
    assert 0 not in res.shown_order
    assert 1 in res.shown_order


def test_greedy_shown_order_by_ecpm():
    bids = np.array([0.5, 1.0, 0.8])
    res = run_greedy(bids, np.ones(3), k=3, n_words=100,
                     ad_word_counts=[5, 5, 5],
                     position_discounts=default_position_discounts(3))
    # Expected order: ad 1 (ecpm=1.0) > ad 2 (ecpm=0.8) > ad 0 (ecpm=0.5)
    assert res.shown_order[0] == 1


# ── POS-FL baseline ────────────────────────────────────────────────────────

def test_posfl_equal_prominences():
    bids = np.array([1.0, 0.8, 0.6])
    res = run_position_fixed_length(bids, np.ones(3), k=3,
                                    position_discounts=default_position_discounts(3))
    shown_proms = res.prominences[res.shown_order]
    assert np.allclose(shown_proms, shown_proms[0])


def test_posfl_shows_top_k():
    bids = np.array([1.0, 0.8, 0.6, 0.2])
    res = run_position_fixed_length(bids, np.ones(4), k=2,
                                    position_discounts=default_position_discounts(2))
    assert len(res.shown_order) == 2
    assert set(res.shown_order) == {0, 1}  # top-2 by ecpm
