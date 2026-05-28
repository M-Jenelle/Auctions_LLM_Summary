# Experiment Results — Claude Haiku (claude-haiku-4-5-20251001)
**Model:** Claude Haiku 4.5 (claude-haiku-4-5-20251001)  
**Queries evaluated:** 500  
**Mechanisms:** GPA+LLM, Greedy, POS-FL  
**Word budgets (n_words):** 30, 40, 50, 60, 70, 80  
**Beta values (β):** 0.250, 0.333, 0.500

---

## What is Welfare?

```
Welfare = Σ  bid_i × base_ctr_i × ROUGE(summary_i, original_i)^β × pos_norm_i
```

- **bid** — advertiser's willingness to pay per click
- **base_ctr** — baseline click-through rate
- **ROUGE^β** — summary quality penalty (higher β = harsher penalty)
- **pos_norm** — position discount (0.9^(rank−1))

---

## Results by Beta

### β = 0.250 (mild summary quality penalty)

| n_words | GPA+LLM | Greedy | POS-FL | Best |
|---|---|---|---|---|
| 30 | 4.4762 | 3.1909 | 4.3369 | **GPA+LLM** |
| 40 | 4.5341 | 3.4341 | 4.3750 | **GPA+LLM** |
| 50 | 4.5691 | 4.0177 | 4.4313 | **GPA+LLM** |
| 60 | 4.5894 | 4.4313 | 4.4813 | **GPA+LLM** |
| 70 | 4.6057 | 4.6161 | 4.5407 | Greedy |
| 80 | 4.6204 | 4.8022 | 4.5522 | Greedy |

### β = 0.333 (moderate summary quality penalty)

| n_words | GPA+LLM | Greedy | POS-FL | Best |
|---|---|---|---|---|
| 30 | 4.2964 | 3.1909 | 4.1178 | **GPA+LLM** |
| 40 | 4.3696 | 3.4341 | 4.1674 | **GPA+LLM** |
| 50 | 4.4151 | 4.0177 | 4.2391 | **GPA+LLM** |
| 60 | 4.4444 | 4.4313 | 4.3044 | **GPA+LLM** |
| 70 | 4.4599 | 4.6161 | 4.3781 | Greedy |
| 80 | 4.4785 | 4.8022 | 4.3951 | Greedy |

### β = 0.500 (strong summary quality penalty)

| n_words | GPA+LLM | Greedy | POS-FL | Best |
|---|---|---|---|---|
| 30 | 3.9725 | 3.1909 | 3.7153 | **GPA+LLM** |
| 40 | 4.0666 | 3.4341 | 3.7845 | **GPA+LLM** |
| 50 | 4.1104 | 4.0177 | 3.8826 | **GPA+LLM** |
| 60 | 4.1463 | 4.4313 | 3.9747 | Greedy |
| 70 | 4.1496 | 4.6161 | 4.0738 | Greedy |
| 80 | 4.1656 | 4.8022 | 4.1003 | Greedy |

---

## Overall Average Welfare (across all word budgets)

| β | GPA+LLM | Greedy | POS-FL | Winner |
|---|---|---|---|---|
| 0.250 | **4.5658** | 4.0821 | 4.4529 | GPA+LLM |
| 0.333 | **4.4106** | 4.0821 | 4.2670 | GPA+LLM |
| 0.500 | **4.1018** | 4.0821 | 3.9219 | GPA+LLM |

---

## Key Findings

### 1. GPA+LLM wins overall across all beta values
GPA+LLM achieves the highest average welfare in every beta condition, consistent with the paper's main claim.

### 2. Strong advantage at tight word budgets
At n_words=30, GPA+LLM outperforms Greedy by:
- **+40.3%** at β=0.250
- **+34.6%** at β=0.333
- **+24.5%** at β=0.500

### 3. Greedy overtakes at large word budgets
Greedy wins at n_words ≥ 70 for β=0.250 and n_words ≥ 70 for β=0.333, and n_words ≥ 60 for β=0.500.

### 4. Crossover point by beta

| β | Crossover n_words |
|---|---|
| 0.250 | ~70 |
| 0.333 | ~70 |
| 0.500 | ~60 |

---

## Figure

See `figures/figure2_welfare_claude-haiku-4-5-20251001.png` for the Figure 2 replication with Claude Haiku.
