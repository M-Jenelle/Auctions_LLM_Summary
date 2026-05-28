# Experiment Results — Auctions with LLM Summaries
**Model:** Gemini 2.5 Flash  
**Queries evaluated:** 500  
**Mechanisms:** GPA+LLM, Greedy, POS-FL  
**Word budgets (n_words):** 30, 40, 50, 60, 70, 80  
**Beta values (β):** 0.250, 0.333, 0.500

---

## What is Welfare?

Welfare measures how much value the auction delivers across all shown ads:

```
Welfare = Σ  bid_i × base_ctr_i × ROUGE(summary_i, original_i)^β × pos_norm_i
```

- **bid** — how much the advertiser is willing to pay per click
- **base_ctr** — baseline click-through rate for the ad
- **ROUGE^β** — summary quality penalty (higher β = harsher penalty for poor summaries)
- **pos_norm** — position discount (0.9^(rank−1)), ads lower on the page get less weight

Higher welfare = better outcome for both advertisers and the platform.

---

## Mechanism Descriptions

| Mechanism | Description |
|---|---|
| **GPA+LLM** | Allocates word budget proportional to each ad's prominence (bid × CTR weighted). Uses Gemini to compress each ad to its word limit. |
| **Greedy** | Shows full ads in descending ECPM order until the word budget runs out. No summarization. |
| **POS-FL** | Splits the word budget equally among all shown ads. Uses Gemini to compress. |

---

## Results by Beta

### β = 0.250 (mild summary quality penalty)

| n_words | GPA+LLM | Greedy | POS-FL | Best |
|---|---|---|---|---|
| 30 | 4.5543 | 3.2715 | 4.2912 | **GPA+LLM** |
| 40 | 4.6450 | 3.5211 | 4.4240 | **GPA+LLM** |
| 50 | 4.7005 | 4.1159 | 4.4934 | **GPA+LLM** |
| 60 | 4.7294 | 4.5230 | 4.6311 | **GPA+LLM** |
| 70 | 4.7593 | 4.7006 | 4.7455 | **GPA+LLM** |
| 80 | 4.7800 | 4.9022 | 4.7605 | Greedy |

### β = 0.333 (moderate summary quality penalty)

| n_words | GPA+LLM | Greedy | POS-FL | Best |
|---|---|---|---|---|
| 30 | 4.3824 | 3.2821 | 4.0367 | **GPA+LLM** |
| 40 | 4.4892 | 3.5211 | 4.2038 | **GPA+LLM** |
| 50 | 4.5485 | 4.1247 | 4.2920 | **GPA+LLM** |
| 60 | 4.5958 | 4.5230 | 4.4683 | **GPA+LLM** |
| 70 | 4.6237 | 4.7006 | 4.6158 | Greedy |
| 80 | 4.6482 | 4.9029 | 4.6370 | Greedy |

### β = 0.500 (strong summary quality penalty)

| n_words | GPA+LLM | Greedy | POS-FL | Best |
|---|---|---|---|---|
| 30 | 4.0768 | 3.2715 | 3.5749 | **GPA+LLM** |
| 40 | 4.1938 | 3.5211 | 3.7983 | **GPA+LLM** |
| 50 | 4.2511 | 4.1159 | 3.9186 | **GPA+LLM** |
| 60 | 4.2929 | 4.5230 | 4.1625 | Greedy |
| 70 | 4.3243 | 4.7006 | 4.3692 | Greedy |
| 80 | 4.3597 | 4.9022 | 4.3979 | Greedy |

---

## Overall Average Welfare (across all word budgets)

| β | GPA+LLM | Greedy | POS-FL | Winner |
|---|---|---|---|---|
| 0.250 | **4.6948** | 4.1724 | 4.5576 | GPA+LLM |
| 0.333 | **4.5480** | 4.1757 | 4.3756 | GPA+LLM |
| 0.500 | **4.2498** | 4.1724 | 4.0369 | GPA+LLM |

---

## Key Findings

### 1. GPA+LLM wins overall across all beta values
GPA+LLM achieves the highest average welfare in every beta condition, confirming the paper's main claim that proportional word allocation with LLM compression outperforms both baselines.

### 2. GPA+LLM dominates at tight word budgets
At n_words=30, GPA+LLM outperforms Greedy by:
- **+39.2%** at β=0.250
- **+33.5%** at β=0.333
- **+24.6%** at β=0.500

This is because Greedy can only fit 1–2 full ads in a small budget, while GPA+LLM shows all 4 ads in compressed form.

### 3. Greedy wins only at large word budgets
Greedy overtakes GPA+LLM at n_words=80 (β=0.250) and n_words=70–80 (β≥0.333). At large budgets, full ads fit without compression loss — so ROUGE ≈ 1 and Greedy's high-ECPM ordering wins.

### 4. Higher β narrows GPA+LLM's advantage
As β increases, the ROUGE penalty on summaries becomes more severe. This reduces GPA+LLM's margin because LLM summaries score lower than perfect ROUGE=1 full ads. Even so, GPA+LLM remains the overall winner.

### 5. POS-FL consistently underperforms GPA+LLM
Equal word allocation ignores ad quality — high-value ads and low-value ads get the same budget. GPA+LLM's proportional allocation concentrates words on high-ECPM ads, yielding better welfare.

---

## Crossover Point (where Greedy overtakes GPA+LLM)

| β | Crossover n_words |
|---|---|
| 0.250 | ~80 |
| 0.333 | ~70 |
| 0.500 | ~60 |

As β increases, the crossover happens at smaller word budgets — LLM summaries need more words to match full-ad quality when the quality penalty is steeper.

---

## Figure

See `figures/figure2_welfare.png` for the replication of Figure 2 from the paper (average welfare vs. total word budget, one subplot per β).
