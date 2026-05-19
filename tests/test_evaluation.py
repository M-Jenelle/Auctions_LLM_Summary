import numpy as np
import pytest
from src.evaluation import compute_rouge, compute_welfare, compute_welfare_greedy


# ── ROUGE ──────────────────────────────────────────────────────────────────

def test_rouge_identical():
    text = "Buy the best golf clubs online at great prices with free delivery."
    score = compute_rouge(text, text)
    assert score == pytest.approx(1.0, abs=0.01)


def test_rouge_empty_summary():
    assert compute_rouge("", "some original text here") == 0.0


def test_rouge_empty_original():
    assert compute_rouge("some summary", "") == 0.0


def test_rouge_partial_overlap():
    summary = "Great golf clubs low prices."
    original = "Buy the best golf clubs online at great prices with free shipping."
    score = compute_rouge(summary, original)
    assert 0.0 < score < 1.0


def test_rouge_in_unit_interval():
    summary = "Completely unrelated text about cooking recipes and food."
    original = "Buy the best golf clubs online at great prices."
    score = compute_rouge(summary, original)
    assert 0.0 <= score <= 1.0


# ── compute_welfare ────────────────────────────────────────────────────────

def test_welfare_zero_pos_norm():
    """Nothing shown → welfare = 0."""
    bids = np.array([1.0, 0.5])
    base_ctrs = np.array([0.4, 0.3])
    ads = ["Great golf clubs!", "Affordable lessons."]
    pos_norms = np.zeros(2)

    w = compute_welfare(bids, base_ctrs, ads, ads, pos_norms, beta=0.5)
    assert w == 0.0


def test_welfare_empty_summary():
    """Empty summary treated as not shown."""
    bids = np.array([1.0])
    base_ctrs = np.array([0.4])
    pos_norms = np.array([1.0])
    w = compute_welfare(bids, base_ctrs, [""], ["original text"], pos_norms, beta=0.5)
    assert w == 0.0


def test_welfare_nonneg():
    bids = np.array([1.0, 0.8])
    base_ctrs = np.array([0.4, 0.3])
    ads = ["Golf clubs for beginners.", "Pro lessons at home."]
    pos_norms = np.array([1.0, 0.9])

    w = compute_welfare(bids, base_ctrs, ads, ads, pos_norms, beta=0.5)
    assert w >= 0.0


def test_welfare_full_ad_approx_upper_bound():
    """Showing full ads (ROUGE→1) gives welfare ≈ Σ b_i * ctr_i * pos_norm_i."""
    bids = np.array([1.0, 0.5])
    base_ctrs = np.array([0.4, 0.3])
    ads = ["Great golf clubs at low prices today.", "Learn to play golf with professionals."]
    pos_norms = np.array([1.0, 0.9])
    beta = 0.5

    w = compute_welfare(bids, base_ctrs, ads, ads, pos_norms, beta)
    upper = float(np.dot(bids * base_ctrs, pos_norms))  # ROUGE=1 case
    assert w <= upper + 1e-6


def test_welfare_beta_effect():
    """Lower beta → higher welfare for same ROUGE < 1 (prom^beta increases as beta→0)."""
    bids = np.array([1.0])
    base_ctrs = np.array([0.5])
    summary = "Golf clubs."
    original = "Buy the best golf clubs online at great prices with free shipping today."
    pos_norms = np.array([1.0])

    w_lo = compute_welfare(bids, base_ctrs, [summary], [original], pos_norms, beta=0.25)
    w_hi = compute_welfare(bids, base_ctrs, [summary], [original], pos_norms, beta=0.75)
    assert w_lo >= w_hi - 1e-9  # lower beta inflates quality score less harshly


# ── compute_welfare_greedy ─────────────────────────────────────────────────

def test_greedy_welfare_formula():
    """Greedy welfare = Σ b_i * ctr_i * pos_norm_i (ROUGE = 1 always)."""
    bids = np.array([1.0, 0.5])
    base_ctrs = np.array([0.4, 0.3])
    pos_norms = np.array([1.0, 0.9])

    w = compute_welfare_greedy(bids, base_ctrs, pos_norms)
    expected = 1.0 * 0.4 * 1.0 + 0.5 * 0.3 * 0.9
    assert w == pytest.approx(expected, rel=1e-9)


def test_greedy_welfare_zero_nothing_shown():
    bids = np.array([1.0, 0.5])
    base_ctrs = np.array([0.4, 0.3])
    w = compute_welfare_greedy(bids, base_ctrs, np.zeros(2))
    assert w == 0.0
