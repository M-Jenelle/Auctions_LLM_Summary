# Experiment configuration matching Section 5 of the paper

MODEL_NAME = "gemini-2.5-flash"
N_QUERIES = 500
K = 4                            # max ads to show per query
POSITION_DECAY = 0.9             # r_t = POSITION_DECAY^(t-1)
SEED = 42

# Section 5.4: x-axis values from Figure 2
N_WORDS_LIST = [30, 40, 50, 60, 70, 80]

# Section 5.4: β ∈ {1/2, 1/3, 1/4} for f(Prom) = Prom^β (Section 5.4)
BETA_LIST = [0.5, 1 / 3, 0.25]

# Integration steps for Myerson payment numerical integration
MYERSON_STEPS = 200

# Paths — all absolute so scripts work regardless of working directory
from pathlib import Path
_ROOT = Path(__file__).parent
DATA_PATH    = str(_ROOT / "results" / "dataset.json")
RESULTS_PATH = str(_ROOT / "results" / "welfare_results.json")
FIGURES_DIR  = str(_ROOT / "results" / "figures")
CACHE_PATH   = str(_ROOT / "results" / "summary_cache.json")
