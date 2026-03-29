"""
Hypergraph ER-GNN — Catastrophic Forgetting Baseline
======================================================
Fuses the bulletproof data loading of `main.py` with true PyG HypergraphConv.
Sweeps buffer sizes for random experience replay to mitigate forgetting.

Edge -> Hyperedge mapping:
  - Posts act as the "Hyperedges".
  - A hyperedge contains: The Post itself, its Comments (on_post), and the Users (authored) of those comments.
"""

import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import HypergraphConv
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Config (Inherited from main.py) ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../mitigating-catastrophic-forgetting
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]  # .../project
DATA_DIR = WORKSPACE_ROOT / "output" / "combined"    # Directory containing *_gnn_nodes.csv and *_gnn_edges.csv
RES_DIR = PROJECT_ROOT / "res" / "hypergraph_replay"
RES_DIR.mkdir(parents=True, exist_ok=True)

SUBREDDITS = {
    "python": 0, "learnpython": 1, "django": 2,
    "javascript": 3, "node": 4, "webdev": 5,
}
LABEL_NAMES = {v: k for k, v in SUBREDDITS.items()}
PYTHON_LABELS = [0, 1, 2]
JS_LABELS = [3, 4, 5]
NUM_CLASSES = 6

TFIDF_MAX_FEATURES = 512
HIDDEN_DIM = 128
EPOCHS_PER_ROUND = 60    # CRITICAL: 60 epochs prevents class collapse
LEARNING_RATE = 0.005
TRAIN_RATIO = 0.8
SEEDS = [42, 123, 456]   # Reduced for speed, expand as needed
BUFFER_SIZES = [100, 500, 1000, 2000]


def resolve_dataset_dir():
    """Resolve a valid dataset directory with subreddit graph CSVs."""
    candidates = [
        DATA_DIR,
        PROJECT_ROOT / "output" / "combined",
        WORKSPACE_ROOT / "output" / "combined",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    searched = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find dataset directory. Searched:\n{searched}")


def resolve_subreddit_files(dataset_dir: Path, subreddit_name: str):
    """Find subreddit node/edge CSVs, including nested directories."""
    nodes_name = f"{subreddit_name}_gnn_nodes.csv"
    edges_name = f"{subreddit_name}_gnn_edges.csv"

    direct_nodes = dataset_dir / nodes_name
    direct_edges = dataset_dir / edges_name
    if direct_nodes.exists() and direct_edges.exists():
        return direct_nodes, direct_edges

    recursive_nodes = list(dataset_dir.rglob(nodes_name))
    recursive_edges = list(dataset_dir.rglob(edges_name))
    if recursive_nodes and recursive_edges:
        return recursive_nodes[0], recursive_edges[0]

    raise FileNotFoundError(
        f"Missing files for r/{subreddit_name}: {nodes_name} and/or {edges_name} under {dataset_dir}"
    )


def validate_graph_frames(subreddit_name: str, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """Validate required schema before graph construction."""
    required_node_cols = {"node_id", "node_type", "text"}
    required_edge_cols = {"src", "dst", "relation"}

    missing_node = required_node_cols.difference(nodes_df.columns)
    missing_edge = required_edge_cols.difference(edges_df.columns)
    if missing_node:
        raise ValueError(f"r/{subreddit_name} nodes missing columns: {sorted(missing_node)}")
    if missing_edge:
        raise ValueError(f"r/{subreddit_name} edges missing columns: {sorted(missing_edge)}")

    comment_count = int((nodes_df["node_type"] == "comment").sum())
    on_post_count = int((edges_df["relation"] == "on_post").sum())
    authored_count = int((edges_df["relation"] == "authored").sum())
    if comment_count == 0:
        raise ValueError(f"r/{subreddit_name} has zero comment nodes.")
    if on_post_count == 0:
        raise ValueError(f"r/{subreddit_name} has zero on_post edges.")

    return {
        "comments": comment_count,
        "on_post": on_post_count,
        "authored": authored_count,
    }

# ─── Data Loading & Hypergraph Construction ──────────────────────────────────

def load_subreddit_data(dataset_dir: Path, subreddit_name: str, label: int):
    nodes_path, edges_path = resolve_subreddit_files(dataset_dir, subreddit_name)
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    stats = validate_graph_frames(subreddit_name, nodes_df, edges_df)

    print(
        f"  r/{subreddit_name}: comments={stats['comments']}, "
        f"on_post={stats['on_post']}, authored={stats['authored']}"
    )

    nodes_df["label"] = label
    nodes_df.loc[nodes_df["node_type"] != "comment", "label"] = -1
    return nodes_df, edges_df

def build_unified_hypergraph(seed: int):
    dataset_dir = resolve_dataset_dir()

    all_texts, all_labels, all_node_types = [], [], []
    all_edges = []
    node_offset = 0

    # 1. Load all nodes and standard edges
    print(f"Loading dataset from: {dataset_dir}")
    for subreddit, label in SUBREDDITS.items():
        nodes_df, edges_df = load_subreddit_data(dataset_dir, subreddit, label)
        n_nodes = len(nodes_df)

        all_texts.extend(nodes_df["text"].fillna("").astype(str).tolist())
        all_labels.extend(nodes_df["label"].tolist())
        all_node_types.extend(nodes_df["node_type"].tolist())

        # Offset edges to global indexing
        for row in edges_df.itertuples(index=False):
            all_edges.append((row.src + node_offset, row.dst + node_offset, row.relation))

        node_offset += n_nodes

    total_nodes = len(all_texts)

    # 2. Build Hyperedges (Posts group their comments and users)
    post_to_nodes = defaultdict(set)
    comment_to_post = {}

    # Map comments to posts
    for src, dst, rel in all_edges:
        if rel == "on_post":
            comment_to_post[src] = dst
            post_to_nodes[dst].add(src)  # Add comment to post hyperedge
            post_to_nodes[dst].add(dst)  # Add post itself

    # Map users to the posts they commented on
    for src, dst, rel in all_edges:
        if rel == "authored":
            if dst in comment_to_post:
                post = comment_to_post[dst]
                post_to_nodes[post].add(src) # Add user to post hyperedge

    # Create Incidence Matrix [2, num_incidence_edges]
    hyperedge_index_list = []
    hyperedge_id = 0
    for post, nodes in post_to_nodes.items():
        if len(nodes) > 0:
            for n in nodes:
                hyperedge_index_list.append([n, hyperedge_id])
            hyperedge_id += 1

    hyperedge_index = torch.tensor(hyperedge_index_list, dtype=torch.long).t().contiguous()

    # 3. TF-IDF Features
    print("Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, stop_words="english",
        min_df=2, max_df=0.95, sublinear_tf=True,
    )
    x = torch.tensor(vectorizer.fit_transform(all_texts).toarray(), dtype=torch.float32)
    y = torch.tensor(all_labels, dtype=torch.long)

    # 4. Stratified Train/Test Splits
    torch.manual_seed(seed)
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    test_mask = torch.zeros(total_nodes, dtype=torch.bool)

    for label_id in range(NUM_CLASSES):
        label_indices = (y == label_id).nonzero(as_tuple=True)[0]
        if len(label_indices) == 0: continue
        perm = label_indices[torch.randperm(len(label_indices))]
        split = int(len(perm) * TRAIN_RATIO)
        train_mask[perm[:split]] = True
        test_mask[perm[split:]] = True

    # 5. Class Weights (Inverse Frequency)
    class_counts = torch.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        class_counts[c] = (y[train_mask] == c).sum().float()
    
    # Avoid div by zero
    class_counts = class_counts + 1e-6 
    total_labeled = class_counts.sum()
    class_weights = total_labeled / (NUM_CLASSES * class_counts)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES

    data = Data(x=x, edge_index=hyperedge_index, y=y,
                train_mask=train_mask, test_mask=test_mask)

    print(f"Hypergraph unified. Nodes: {total_nodes}, Hyperedges: {hyperedge_id}")
    return data, class_weights

# ─── Models ──────────────────────────────────────────────────────────────────

class HypergraphClassifier(nn.Module):
    """2-Layer Hypergraph Neural Network."""
    def __init__(self, in_ch, hid_ch, out_ch):
        super().__init__()
        # HypergraphConv handles message passing from nodes -> hyperedges -> nodes
        self.conv1 = HypergraphConv(in_ch, hid_ch)
        self.conv2 = HypergraphConv(hid_ch, out_ch)

    def forward(self, x, hyperedge_index):
        x = F.elu(self.conv1(x, hyperedge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, hyperedge_index)

# ─── Training & Replay Logic ─────────────────────────────────────────────────

def build_random_buffer(data, buffer_size, seed):
    """Selects node indices randomly from R1 training set for Replay."""
    r1_labels = torch.tensor(PYTHON_LABELS, device=data.y.device)
    r1_train_mask = data.train_mask & torch.isin(data.y, r1_labels)
    r1_indices = r1_train_mask.nonzero(as_tuple=True)[0].cpu()

    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(len(r1_indices), generator=gen)
    n = min(buffer_size, len(r1_indices))
    return r1_indices[perm[:n]]

def train_epoch(model, data, optimizer, task_labels, class_weights, device):
    """Standard task training."""
    model.train()
    task_tensor = torch.tensor(task_labels, device=device)
    mask = data.train_mask & torch.isin(data.y, task_tensor)
    if mask.sum() == 0: return 0.0
    
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[mask], data.y[mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_epoch_with_replay(model, data, optimizer, task_labels, class_weights, replay_indices, device):
    """Trains on new task + interleaves buffered nodes from old task."""
    model.train()
    task_tensor = torch.tensor(task_labels, device=device)
    task_mask = data.train_mask & torch.isin(data.y, task_tensor)
    
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    
    # Task 2 (JS) Loss
    loss_task = F.cross_entropy(out[task_mask], data.y[task_mask], weight=class_weights)
    
    # Replay (Python) Loss
    buf_idx = replay_indices.to(device)
    loss_replay = F.cross_entropy(out[buf_idx], data.y[buf_idx], weight=class_weights)
    
    loss = loss_task + loss_replay
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, task_labels, device):
    model.eval()
    task_tensor = torch.tensor(task_labels, device=device)
    mask = data.test_mask & torch.isin(data.y, task_tensor)
    if mask.sum() == 0: return 0.0
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=-1)
    return (pred == data.y[mask]).float().mean().item()

@torch.no_grad()
def evaluate_per_class(model, data, device):
    model.eval()
    results = {}
    for label_id in range(NUM_CLASSES):
        t = torch.tensor([label_id], device=device)
        mask = data.test_mask & torch.isin(data.y, t)
        if mask.sum() == 0:
            results[LABEL_NAMES[label_id]] = 0.0
            continue
        out = model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=-1)
        results[LABEL_NAMES[label_id]] = (pred == data.y[mask]).float().mean().item()
    return results

# ─── Runner ──────────────────────────────────────────────────────────────────

def run_experiment(buffer_size, data, class_weights, device, seed):
    torch.manual_seed(seed)
    model = HypergraphClassifier(TFIDF_MAX_FEATURES, HIDDEN_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    history = {"r1_loss": [], "r1_py_acc": [], "r1_js_acc": [], "r2_loss": [], "r2_py_acc": [], "r2_js_acc": []}

    # ROUND 1 (Python)
    for _ in range(EPOCHS_PER_ROUND):
        loss = train_epoch(model, data, optimizer, PYTHON_LABELS, class_weights, device)
        history["r1_loss"].append(loss)
        history["r1_py_acc"].append(evaluate(model, data, PYTHON_LABELS, device))
        history["r1_js_acc"].append(evaluate(model, data, JS_LABELS, device))

    r1_per_class = evaluate_per_class(model, data, device)
    
    # Snapshot Buffer
    replay_indices = build_random_buffer(data, buffer_size, seed)

    # ROUND 2 (JS + Replay)
    for _ in range(EPOCHS_PER_ROUND):
        loss = train_epoch_with_replay(model, data, optimizer, JS_LABELS, class_weights, replay_indices, device)
        history["r2_loss"].append(loss)
        history["r2_py_acc"].append(evaluate(model, data, PYTHON_LABELS, device))
        history["r2_js_acc"].append(evaluate(model, data, JS_LABELS, device))

    r2_per_class = evaluate_per_class(model, data, device)

    return {
        "history": history,
        "r1_python_acc": history["r1_py_acc"][-1],
        "r2_python_acc": history["r2_py_acc"][-1],
        "r2_js_acc": history["r2_js_acc"][-1],
        "forgetting": history["r1_py_acc"][-1] - history["r2_py_acc"][-1],
        "r1_per_class": r1_per_class,
        "r2_per_class": r2_per_class,
    }

# ─── Main Execution ──────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Epochs/round: {EPOCHS_PER_ROUND}\n")

    all_results = {bs: [] for bs in BUFFER_SIZES}

    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        data, class_weights = build_unified_hypergraph(seed)
        
        data = data.to(device)
        class_weights = class_weights.to(device)

        for bs in BUFFER_SIZES:
            print(f"\n  Running Buffer size: {bs}")
            result = run_experiment(bs, data, class_weights, device, seed)
            all_results[bs].append(result)
            
            print(f"    R1 Py: {result['r1_python_acc']:.4f}  "
                  f"R2 Py: {result['r2_python_acc']:.4f}  "
                  f"R2 JS: {result['r2_js_acc']:.4f}  "
                  f"Forgetting: {result['forgetting']:.4f}")

    # Save to disk
    json_path = RES_DIR / "hypergraph_replay_results.json"
    with open(json_path, "w") as f:
        # Avoid saving massive histories if not needed, but leaving intact for your plots
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

if __name__ == "__main__":
    main()