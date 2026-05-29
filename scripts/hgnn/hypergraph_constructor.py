"""
Hypergraph Constructor — Build PyG Data from Reddit CSVs
=========================================================
Loads the 9-subreddit node/edge CSV files (same source as the GNN pipeline)
and constructs a *hypergraph* instead of a pairwise graph.

Hyperedge types
---------------
  Thread:  all comments that belong to the same Reddit post
           (identified via `on_post` edges: comment → post)
  User:    all comments written by the same user
           (identified via `authored` edges: user → comment)

Both edge types are filtered by HYPEREDGE_MIN_SIZE / HYPEREDGE_MAX_SIZE
from config to remove trivial singleton hyperedges and mega-threads.

Node features
-------------
TF-IDF (NODE_FEATURE_DIM = 384) on comment text — no extra dependencies
beyond those already required by the GNN pipeline.

Public API
----------
  build_hypergraph_data(seed)  →  (data, class_weights, info)
  validate_hypergraph(data, info)
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.data import Data

# ── import config from the same package ───────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_subreddit(subreddit: str):
    """Load nodes and edges CSVs for one subreddit.  Returns (nodes_df, edges_df)."""
    data_dir = config.DATA_DIR
    nodes_df = pd.read_csv(data_dir / f"{subreddit}_gnn_nodes.csv")
    edges_df = pd.read_csv(data_dir / f"{subreddit}_gnn_edges.csv")
    return nodes_df, edges_df


def _build_hyperedges(post_to_comments: dict, user_to_comments: dict):
    """
    Convert raw {group_key → [global_comment_idx, ...]} dicts into COO tensors.

    Applies HYPEREDGE_MIN_SIZE / HYPEREDGE_MAX_SIZE filters.

    Returns
    -------
    node_indices : list[int]    — row 0 of hyperedge_index
    he_indices   : list[int]    — row 1 of hyperedge_index
    he_id        : int          — total number of hyperedges
    thread_range : (int, int)   — [start, end) of thread hyperedge IDs
    user_range   : (int, int)   — [start, end) of user hyperedge IDs
    """
    node_indices, he_indices = [], []
    he_id = 0
    mn, mx = config.HYPEREDGE_MIN_SIZE, config.HYPEREDGE_MAX_SIZE

    thread_start = 0
    for members in post_to_comments.values():
        unique = list(set(members))
        if mn <= len(unique) <= mx:
            for c in unique:
                node_indices.append(c)
                he_indices.append(he_id)
            he_id += 1
    thread_end = he_id

    user_start = he_id
    for members in user_to_comments.values():
        unique = list(set(members))
        if mn <= len(unique) <= mx:
            for c in unique:
                node_indices.append(c)
                he_indices.append(he_id)
            he_id += 1
    user_end = he_id

    return node_indices, he_indices, he_id, (thread_start, thread_end), (user_start, user_end)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_hypergraph_data(seed: int):
    """Build a single unified hypergraph from all 9 subreddits.

    Parameters
    ----------
    seed : int
        Random seed for the stratified train/test split.

    Returns
    -------
    data : torch_geometric.data.Data
        Fields: x, y, train_mask, test_mask, hyperedge_index, hyperedge_weight
    class_weights : torch.Tensor  [NUM_CLASSES]
    info : dict
        Metadata including comment_counts, he_type_offsets, etc.
    """
    all_texts:  list[str] = []
    all_labels: list[int] = []
    comment_counts: dict[str, int] = {}

    # {(sub_idx, post_local_idx)  → [global_comment_idx, ...]}
    post_to_comments: dict = defaultdict(list)
    # {(sub_idx, user_local_idx)  → [global_comment_idx, ...]}
    user_to_comments: dict = defaultdict(list)

    global_comment_idx = 0

    for sub_idx, (subreddit, label) in enumerate(config.SUBREDDITS.items()):
        nodes_df, edges_df = _load_subreddit(subreddit)

        # ── map local node row index → global comment index ──────────────────
        local_to_global: dict[int, int | None] = {}
        n_comments = 0

        for local_idx in range(len(nodes_df)):
            row = nodes_df.iloc[local_idx]
            if row["node_type"] == "comment":
                local_to_global[local_idx] = global_comment_idx
                text = row.get("text", "")
                all_texts.append(str(text) if pd.notna(text) else "")
                all_labels.append(label)
                global_comment_idx += 1
                n_comments += 1
            else:
                local_to_global[local_idx] = None

        comment_counts[subreddit] = n_comments

        # ── scan edges to build group memberships ─────────────────────────────
        for _, edge in edges_df.iterrows():
            src = int(edge["src"])
            dst = int(edge["dst"])
            rel = str(edge["relation"])

            if rel == "on_post":
                # src = comment node, dst = post node
                g_c = local_to_global.get(src)
                if g_c is not None:
                    post_to_comments[(sub_idx, dst)].append(g_c)

            elif rel == "authored":
                # src = user node, dst = comment node
                g_c = local_to_global.get(dst)
                if g_c is not None:
                    user_to_comments[(sub_idx, src)].append(g_c)

    # ── node features: TF-IDF at NODE_FEATURE_DIM dims ───────────────────────
    print(f"    Building TF-IDF features ({config.NODE_FEATURE_DIM} dims) "
          f"over {global_comment_idx:,} comment nodes …")
    vectorizer = TfidfVectorizer(
        max_features=config.NODE_FEATURE_DIM,
        stop_words="english",
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    x = torch.tensor(
        vectorizer.fit_transform(all_texts).toarray(), dtype=torch.float32
    )

    # ── labels and stratified split ───────────────────────────────────────────
    y = torch.tensor(all_labels, dtype=torch.long)
    N = len(all_texts)

    torch.manual_seed(seed)
    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)

    for label_id in range(config.NUM_CLASSES):
        idxs = (y == label_id).nonzero(as_tuple=True)[0]
        perm  = idxs[torch.randperm(len(idxs))]
        split = int(len(perm) * config.TRAIN_RATIO)
        train_mask[perm[:split]] = True
        test_mask[perm[split:]]  = True

    # ── class weights (inverse frequency, normalised) ─────────────────────────
    counts = torch.zeros(config.NUM_CLASSES)
    for c in range(config.NUM_CLASSES):
        counts[c] = (y[train_mask] == c).sum().float()
    counts = counts.clamp(min=1)
    class_weights = (counts.sum() / (config.NUM_CLASSES * counts))
    class_weights = class_weights / class_weights.sum() * config.NUM_CLASSES

    # ── build hyperedges ──────────────────────────────────────────────────────
    print("    Building hyperedges …")
    node_indices, he_indices, n_he, thread_range, user_range = _build_hyperedges(
        post_to_comments, user_to_comments
    )

    if node_indices:
        hyperedge_index  = torch.tensor([node_indices, he_indices], dtype=torch.long)
        hyperedge_weight = torch.ones(n_he, dtype=torch.float32)
    else:
        hyperedge_index  = torch.zeros((2, 0), dtype=torch.long)
        hyperedge_weight = torch.zeros(0, dtype=torch.float32)

    data = Data(
        x=x,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
        hyperedge_index=hyperedge_index,
        hyperedge_weight=hyperedge_weight,
    )

    info = {
        "total_nodes":    N,
        "num_hyperedges": n_he,
        "num_thread_he":  thread_range[1] - thread_range[0],
        "num_user_he":    user_range[1]   - user_range[0],
        "comment_counts": comment_counts,
        "he_type_offsets": {
            "thread": thread_range,
            "user":   user_range,
        },
        "class_weights": class_weights.tolist(),
    }

    return data, class_weights, info


def validate_hypergraph(data: Data, info: dict) -> None:
    """Print a quick sanity-check summary of the built hypergraph."""
    N        = data.x.shape[0]
    n_train  = data.train_mask.sum().item()
    n_test   = data.test_mask.sum().item()
    n_he     = (int(data.hyperedge_index[1].max()) + 1
                if data.hyperedge_index.shape[1] > 0 else 0)
    n_inc    = data.hyperedge_index.shape[1]
    feat_dim = data.x.shape[1]

    print(f"    Nodes:       {N:,}  (train {n_train:,} / test {n_test:,})")
    print(f"    Hyperedges:  {n_he:,}  "
          f"(Thread {info['num_thread_he']:,}  +  User {info['num_user_he']:,})")
    print(f"    Incidences:  {n_inc:,}")
    print(f"    Feature dim: {feat_dim}")

    # Per-task node counts
    for task_id, labels in config.TASK_SPLIT.items():
        t = torch.tensor(labels)
        n_task = torch.isin(data.y, t).sum().item()
        if n_task == 0:
            print(f"    ⚠  {task_id} has 0 nodes — check data files!")
        else:
            print(f"    {task_id}: {n_task:,} nodes")
