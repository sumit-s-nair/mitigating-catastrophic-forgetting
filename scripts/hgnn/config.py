"""
Configuration — HGNN Catastrophic Forgetting Experiments
=========================================================
All constants for the hypergraph continual learning pipeline.
No CLI/argparse — edit this file directly for experiment variations.
"""

from pathlib import Path
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "res" / "hgnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Subreddit / Label Mapping ───────────────────────────────────────────────
SUBREDDITS = {
    "python": 0,
    "learnpython": 1,
    "django": 2,
    "javascript": 3,
    "node": 4,
    "webdev": 5,
}
LABEL_NAMES = {v: k for k, v in SUBREDDITS.items()}
NUM_CLASSES = 6

# Task split: T1 = Python ecosystem, T2 = JavaScript ecosystem
TASK_SPLIT = {
    "T1": [0, 1, 2],  # python, learnpython, django
    "T2": [3, 4, 5],  # javascript, node, webdev
}
T1_LABELS = TASK_SPLIT["T1"]
T2_LABELS = TASK_SPLIT["T2"]
ALL_LABELS = list(range(NUM_CLASSES))

# ─── Hypergraph Construction ────────────────────────────────────────────────
# Nodes = comments.  Hyperedge types:
#   Thread:   all comments on the same post (on_post edges grouped by post)
#   User:     all comments by the same user (authored edges grouped by user)
HYPEREDGE_MIN_SIZE = 2    # min members for a valid hyperedge
HYPEREDGE_MAX_SIZE = 50   # filter out mega-threads / super-active users
NODE_FEATURE_DIM = 384    # all-MiniLM-L6-v2 sentence-transformer embedding dim

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_TYPE = "HGNN"       # "HGNN" or "AllDeepSets"
HIDDEN_DIM = 128
DROPOUT = 0.5

# ─── Training ────────────────────────────────────────────────────────────────
EPOCHS_PER_TASK = 50
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
TRAIN_RATIO = 0.8

# ─── Continual Learning ─────────────────────────────────────────────────────
BUFFER_SIZE = 200         # replay buffer size
LWF_ALPHA = 1.0           # LwF distillation weight
LWF_TEMPERATURE = 2.0     # LwF temperature for softening
EWC_LAMBDA = 5000         # EWC penalty weight
EWC_FISHER_SAMPLES = 2000 # subsample for Fisher computation efficiency

# ─── Experiment ──────────────────────────────────────────────────────────────
SEED = 42
SEEDS = [42, 123, 456, 789, 1024]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
