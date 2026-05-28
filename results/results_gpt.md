# Experiment Results — GPT-4o Mini
**Model:** GPT-4o Mini (gpt-4o-mini)  
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
| 30 | 4.4783 | 3.1909 | 4.3259 | **GPA+LLM** |
| 40 | 4.5240 | 3.4341 | 4.3851 | **GPA+LLM** |
| 50 | 4.5658 | 4.0177 | 4.4206 | **GPA+LLM** |
| 60 | 4.5884 | 4.4313 | 4.4649 | **GPA+LLM** |
| 70 | 4.6072 | 4.6161 | 4.5031 | Greedy |
| 80 | 4.6237 | 4.8022 | 4.5250 | Greedy |

### β = 0.333 (moderate summary quality penalty)

| n_words | GPA+LLM | Greedy | POS-FL | Best |
|---|---|---|---|---|
| 30 | 4.2961 | 3.1909 | 4.1034 | **GPA+LLM** |
| 40 | 4.3604 | 3.4341 | 4.1786 | **GPA+LLM** |
| 50 | 4.4124 | 4.0177 | 4.2239 | **GPA+LLM** |
| 60 | 4.4445 | 4.4313 | 4.2803 | **GPA+LLM** |
| 70 | 4.4593 | 4.6161 | 4.3295 | Greedy |
| 80 | 4.4848 | 4.8022 | 4.3575 | Greedy |

### β = 0.500 (strong summary quality penalty)

| n_words | GPA+LLM | Greedy | POS-FL | Best |
|---|---|---|---|---|
| 30 | 3.9828 | 3.1909 | 3.6950 | **GPA+LLM** |
| 40 | 4.0538 | 3.4341 | 3.7976 | **GPA+LLM** |
| 50 | 4.1170 | 4.0177 | 3.8595 | **GPA+LLM** |
| 60 | 4.1595 | 4.4313 | 3.9369 | Greedy |
| 70 | 4.1843 | 4.6161 | 4.0053 | Greedy |
| 80 | 4.1929 | 4.8022 | 4.0440 | Greedy |

---

## Overall Average Welfare (across all word budgets)

| β | GPA+LLM | Greedy | POS-FL | Winner |
|---|---|---|---|---|
| 0.250 | **4.5646** | 4.0821 | 4.4374 | GPA+LLM |
| 0.333 | **4.4096** | 4.0821 | 4.2455 | GPA+LLM |
| 0.500 | **4.1151** | 4.0821 | 3.8897 | GPA+LLM |

---

## Key Findings

### 1. GPA+LLM wins overall across all beta values
GPA+LLM achieves the highest average welfare in every beta condition, consistent with the paper's main claim.

### 2. Strong advantage at tight word budgets
At n_words=30, GPA+LLM outperforms Greedy by:
- **+40.3%** at β=0.250
- **+34.6%** at β=0.333
- **+24.8%** at β=0.500

### 3. Greedy overtakes at large word budgets
Greedy wins at n_words ≥ 70 for β=0.250 and β=0.333, and n_words ≥ 60 for β=0.500.

### 4. Crossover point by beta

| β | Crossover n_words |
|---|---|
| 0.250 | ~70 |
| 0.333 | ~70 |
| 0.500 | ~60 |

---

## Figure

See `figures/welfare_gpt-4o-mini.png` for the Figure 2 replication with GPT-4o Mini.
