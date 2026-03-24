"""
GNN Node Classification — Catastrophic Forgetting Baseline (Research-Grade)
============================================================================
Demonstrates catastrophic forgetting on a Reddit subreddit graph.

Experiments run:
  1. Naive Sequential GAT  — Train Python (R1), fine-tune JS (R2), measure forgetting
  2. Joint Training GAT    — Train all 6 subreddits simultaneously (upper bound)
  3. MLP Ablation          — Same as (1) but without graph convolutions

All experiments run over multiple seeds with mean ± std reported.
Results (JSON, plots, summary) saved to /res/gnn_baseline/.

Edge Semantics:
  - authored : user → comment  (user wrote the comment)
  - on_post  : comment → post  (comment belongs to a post)
  - reply_to : comment → comment (threaded reply)
  These encode real social structure: user activity, discussion trees, reply chains.
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
from torch_geometric.nn import GATConv
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RES_DIR = PROJECT_ROOT / "res" / "gnn_baseline"
RES_DIR.mkdir(parents=True, exist_ok=True)

SUBREDDITS = {
    "python": 0,
    "learnpython": 1,
    "django": 2,
    "javascript": 3,
    "node": 4,
    "webdev": 5,
}
LABEL_NAMES = {v: k for k, v in SUBREDDITS.items()}
PYTHON_LABELS = [0, 1, 2]
JS_LABELS = [3, 4, 5]
ALL_LABELS = list(range(6))
NUM_CLASSES = 6

TFIDF_MAX_FEATURES = 512
HIDDEN_DIM = 128
GAT_HEADS = 4
EPOCHS_PER_ROUND = 60
LEARNING_RATE = 0.005
TRAIN_RATIO = 0.8
SEEDS = [42, 123, 456, 789, 1024]


# ─── Data Loading ────────────────────────────────────────────────────────────
def load_subreddit_data(subreddit_name: str, label: int):
    """Load nodes and edges CSVs for a single subreddit."""
    nodes_df = pd.read_csv(DATA_DIR / f"{subreddit_name}_gnn_nodes.csv")
    edges_df = pd.read_csv(DATA_DIR / f"{subreddit_name}_gnn_edges.csv")

    nodes_df["label"] = label
    nodes_df.loc[nodes_df["node_type"] != "comment", "label"] = -1
    return nodes_df, edges_df


def build_unified_graph(seed: int):
    """Merge all subreddit graphs into one PyG Data object."""
    all_texts, all_labels, all_node_types = [], [], []
    all_edges_src, all_edges_dst = [], []
    edge_relation_counts = defaultdict(int)
    node_offset = 0
    comment_counts = {}

    for subreddit, label in SUBREDDITS.items():
        nodes_df, edges_df = load_subreddit_data(subreddit, label)
        n_nodes = len(nodes_df)
        n_comments = (nodes_df["node_type"] == "comment").sum()
        comment_counts[subreddit] = n_comments

        all_texts.extend(nodes_df["text"].fillna("").astype(str).tolist())
        all_labels.extend(nodes_df["label"].tolist())
        all_node_types.extend(nodes_df["node_type"].tolist())

        all_edges_src.extend((edges_df["src"] + node_offset).tolist())
        all_edges_dst.extend((edges_df["dst"] + node_offset).tolist())

        for rel in edges_df["relation"]:
            edge_relation_counts[rel] += 1

        node_offset += n_nodes

    total_nodes = len(all_texts)

    # TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES, stop_words="english",
        min_df=2, max_df=0.95, sublinear_tf=True,
    )
    x = torch.tensor(vectorizer.fit_transform(all_texts).toarray(), dtype=torch.float32)

    # Bidirectional edges
    src = torch.tensor(all_edges_src, dtype=torch.long)
    dst = torch.tensor(all_edges_dst, dtype=torch.long)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    y = torch.tensor(all_labels, dtype=torch.long)
    labeled_mask = y >= 0

    # Stratified train/test split
    torch.manual_seed(seed)
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    test_mask = torch.zeros(total_nodes, dtype=torch.bool)

    for label_id in range(NUM_CLASSES):
        label_indices = (y == label_id).nonzero(as_tuple=True)[0]
        perm = label_indices[torch.randperm(len(label_indices))]
        split = int(len(perm) * TRAIN_RATIO)
        train_mask[perm[:split]] = True
        test_mask[perm[split:]] = True

    # Class weights (inverse frequency)
    class_counts = torch.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        class_counts[c] = (y[train_mask] == c).sum().float()
    total_labeled = class_counts.sum()
    class_weights = total_labeled / (NUM_CLASSES * class_counts)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES  # normalize

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, test_mask=test_mask)

    info = {
        "total_nodes": total_nodes,
        "total_edges": len(all_edges_src),
        "comment_counts": comment_counts,
        "edge_relations": dict(edge_relation_counts),
        "class_weights": class_weights.tolist(),
    }
    return data, class_weights, info


# ─── Models ──────────────────────────────────────────────────────────────────
class GATClassifier(nn.Module):
    """2-layer Graph Attention Network."""
    def __init__(self, in_ch, hid_ch, out_ch, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_ch, hid_ch, heads=heads, concat=True)
        self.conv2 = GATConv(hid_ch * heads, out_ch, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


class MLPClassifier(nn.Module):
    """2-layer MLP baseline (no graph structure)."""
    def __init__(self, in_ch, hid_ch, out_ch):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hid_ch)
        self.fc2 = nn.Linear(hid_ch, out_ch)

    def forward(self, x, edge_index=None):  # edge_index ignored
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)


# ─── Training & Evaluation ──────────────────────────────────────────────────
def train_epoch(model, data, optimizer, task_labels, class_weights, device):
    model.train()
    data = data.to(device)
    task_tensor = torch.tensor(task_labels, device=device)
    mask = data.train_mask & torch.isin(data.y, task_tensor)
    if mask.sum() == 0:
        return 0.0
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[mask], data.y[mask], weight=class_weights.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, task_labels, device):
    model.eval()
    data = data.to(device)
    task_tensor = torch.tensor(task_labels, device=device)
    mask = data.test_mask & torch.isin(data.y, task_tensor)
    if mask.sum() == 0:
        return 0.0
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=-1)
    return (pred == data.y[mask]).float().mean().item()


@torch.no_grad()
def evaluate_per_class(model, data, device):
    model.eval()
    data = data.to(device)
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


# ─── Experiment Runners ──────────────────────────────────────────────────────

def run_naive_sequential(model_cls, model_kwargs, data, class_weights, device, seed):
    """Train R1 (Python) → R2 (JS). Returns per-epoch metrics."""
    torch.manual_seed(seed)
    model = model_cls(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    history = {"r1_loss": [], "r1_py_acc": [], "r1_js_acc": [],
               "r2_loss": [], "r2_py_acc": [], "r2_js_acc": []}

    # Round 1
    for _ in range(EPOCHS_PER_ROUND):
        loss = train_epoch(model, data, optimizer, PYTHON_LABELS, class_weights, device)
        history["r1_loss"].append(loss)
        history["r1_py_acc"].append(evaluate(model, data, PYTHON_LABELS, device))
        history["r1_js_acc"].append(evaluate(model, data, JS_LABELS, device))

    r1_per_class = evaluate_per_class(model, data, device)

    # Round 2
    for _ in range(EPOCHS_PER_ROUND):
        loss = train_epoch(model, data, optimizer, JS_LABELS, class_weights, device)
        history["r2_loss"].append(loss)
        history["r2_py_acc"].append(evaluate(model, data, PYTHON_LABELS, device))
        history["r2_js_acc"].append(evaluate(model, data, JS_LABELS, device))

    r2_per_class = evaluate_per_class(model, data, device)

    return {
        "history": history,
        "r1_python_acc": history["r1_py_acc"][-1],
        "r1_js_acc": history["r1_js_acc"][-1],
        "r2_python_acc": history["r2_py_acc"][-1],
        "r2_js_acc": history["r2_js_acc"][-1],
        "forgetting": history["r1_py_acc"][-1] - history["r2_py_acc"][-1],
        "r1_per_class": r1_per_class,
        "r2_per_class": r2_per_class,
    }


def run_joint_training(model_cls, model_kwargs, data, class_weights, device, seed):
    """Train on all 6 subreddits simultaneously (upper bound)."""
    torch.manual_seed(seed)
    model = model_cls(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    history = {"loss": [], "py_acc": [], "js_acc": []}

    for _ in range(EPOCHS_PER_ROUND):
        loss = train_epoch(model, data, optimizer, ALL_LABELS, class_weights, device)
        history["loss"].append(loss)
        history["py_acc"].append(evaluate(model, data, PYTHON_LABELS, device))
        history["js_acc"].append(evaluate(model, data, JS_LABELS, device))

    per_class = evaluate_per_class(model, data, device)

    return {
        "history": history,
        "python_acc": history["py_acc"][-1],
        "js_acc": history["js_acc"][-1],
        "per_class": per_class,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Epochs/round: {EPOCHS_PER_ROUND}\n")

    # Build graph once with first seed (features are deterministic, only masks change)
    # Actually rebuild per seed for different train/test splits
    gat_kwargs = dict(in_ch=TFIDF_MAX_FEATURES, hid_ch=HIDDEN_DIM,
                      out_ch=NUM_CLASSES, heads=GAT_HEADS)
    mlp_kwargs = dict(in_ch=TFIDF_MAX_FEATURES, hid_ch=HIDDEN_DIM * GAT_HEADS,
                      out_ch=NUM_CLASSES)  # match param count roughly

    all_results = {
        "config": {
            "model": "GAT", "hidden_dim": HIDDEN_DIM, "gat_heads": GAT_HEADS,
            "epochs_per_round": EPOCHS_PER_ROUND, "lr": LEARNING_RATE,
            "tfidf_features": TFIDF_MAX_FEATURES, "train_ratio": TRAIN_RATIO,
            "seeds": SEEDS,
        },
        "naive_gat": [], "joint_gat": [], "naive_mlp": [],
        "data_info": None,
    }

    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"SEED {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'='*60}")

        data, class_weights, info = build_unified_graph(seed)
        if all_results["data_info"] is None:
            all_results["data_info"] = info
            print(f"\n  Comment counts: {info['comment_counts']}")
            print(f"  Edge relations: {info['edge_relations']}")
            print(f"  Class weights:  {[f'{w:.3f}' for w in info['class_weights']]}\n")

        # 1. Naive Sequential GAT
        print(f"\n  [1/3] Naive Sequential GAT...")
        res_gat = run_naive_sequential(
            GATClassifier, gat_kwargs, data, class_weights, device, seed)
        all_results["naive_gat"].append(res_gat)
        print(f"        R1 Py: {res_gat['r1_python_acc']:.4f}  "
              f"R2 Py: {res_gat['r2_python_acc']:.4f}  "
              f"R2 JS: {res_gat['r2_js_acc']:.4f}  "
              f"Forgetting: {res_gat['forgetting']:.4f}")

        # 2. Joint Training GAT
        print(f"  [2/3] Joint Training GAT...")
        res_joint = run_joint_training(
            GATClassifier, gat_kwargs, data, class_weights, device, seed)
        all_results["joint_gat"].append(res_joint)
        print(f"        Py: {res_joint['python_acc']:.4f}  "
              f"JS: {res_joint['js_acc']:.4f}")

        # 3. Naive Sequential MLP (ablation)
        print(f"  [3/3] Naive Sequential MLP (ablation)...")
        res_mlp = run_naive_sequential(
            MLPClassifier, mlp_kwargs, data, class_weights, device, seed)
        all_results["naive_mlp"].append(res_mlp)
        print(f"        R1 Py: {res_mlp['r1_python_acc']:.4f}  "
              f"R2 Py: {res_mlp['r2_python_acc']:.4f}  "
              f"R2 JS: {res_mlp['r2_js_acc']:.4f}  "
              f"Forgetting: {res_mlp['forgetting']:.4f}")

    # ─── Aggregate Results ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS (mean ± std over {} seeds)".format(len(SEEDS)))
    print(f"{'='*60}\n")

    summary = aggregate_and_print(all_results)
    all_results["summary"] = summary

    # ─── Save Everything ─────────────────────────────────────────────
    save_all(all_results, summary)


def aggregate_and_print(results):
    """Compute mean ± std for all experiments and print tables."""
    summary = {}

    for exp_name, label in [("naive_gat", "Naive GAT"),
                             ("joint_gat", "Joint GAT"),
                             ("naive_mlp", "Naive MLP")]:
        runs = results[exp_name]
        s = {}

        if "forgetting" in runs[0]:  # naive experiments
            for key in ["r1_python_acc", "r2_python_acc", "r2_js_acc", "forgetting"]:
                vals = [r[key] for r in runs]
                s[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        else:  # joint
            for key in ["python_acc", "js_acc"]:
                vals = [r[key] for r in runs]
                s[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        summary[exp_name] = s

        print(f"  {label}:")
        for k, v in s.items():
            print(f"    {k:>20}: {v['mean']:.4f} ± {v['std']:.4f}")
        print()

    # Per-class breakdown (mean over seeds)
    for exp_name, label in [("naive_gat", "Naive GAT"), ("naive_mlp", "Naive MLP")]:
        runs = results[exp_name]
        print(f"  {label} — Per-class after R2:")
        for sub in SUBREDDITS:
            vals = [r["r2_per_class"][sub] for r in runs]
            print(f"    {sub:>12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        print()

    print(f"  Joint GAT — Per-class:")
    for sub in SUBREDDITS:
        vals = [r["per_class"][sub] for r in results["joint_gat"]]
        print(f"    {sub:>12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print()

    return summary


# ─── Saving ──────────────────────────────────────────────────────────────────

def save_all(results, summary):
    """Save JSON, plots, and text summary."""
    print(f"Saving results to {RES_DIR}/\n")

    # 1. JSON (strip history from per-seed to save space, keep one representative)
    results_slim = {k: v for k, v in results.items()}
    json_path = RES_DIR / "experiment_results.json"
    with open(json_path, "w") as f:
        json.dump(results_slim, f, indent=2, default=str)
    print(f"  ✓ {json_path.name}")

    # 2. Training curves (averaged over seeds)
    _plot_training_curves(results)
    print(f"  ✓ training_curves.png")

    # 3. Accuracy comparison bar chart with error bars
    _plot_accuracy_comparison(results, summary)
    print(f"  ✓ accuracy_comparison.png")

    # 4. Forgetting visualization
    _plot_forgetting_viz(results)
    print(f"  ✓ forgetting_visualization.png")

    # 5. GAT vs MLP ablation
    _plot_ablation(results, summary)
    print(f"  ✓ ablation_comparison.png")

    # 6. Text summary
    _save_text_summary(results, summary)
    print(f"  ✓ forgetting_summary.txt")

    print(f"\nAll results saved to {RES_DIR}/")


def _plot_training_curves(results):
    """Loss and accuracy curves averaged over seeds with shaded std."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    E = EPOCHS_PER_ROUND
    epochs_r1 = list(range(1, E + 1))
    epochs_r2 = list(range(E + 1, 2 * E + 1))
    all_epochs = epochs_r1 + epochs_r2

    runs = results["naive_gat"]

    # Collect per-seed curves
    all_loss = np.array([r["history"]["r1_loss"] + r["history"]["r2_loss"] for r in runs])
    all_py = np.array([r["history"]["r1_py_acc"] + r["history"]["r2_py_acc"] for r in runs])
    all_js = np.array([r["history"]["r1_js_acc"] + r["history"]["r2_js_acc"] for r in runs])

    for ax, data_arr, color, title, ylabel in [
        (axes[0], all_loss, "#e74c3c", "Training Loss", "Loss"),
        (axes[1], all_py, "#2ecc71", "Python Subreddits Accuracy", "Accuracy"),
        (axes[2], all_js, "#3498db", "JS Subreddits Accuracy", "Accuracy"),
    ]:
        mean = data_arr.mean(axis=0)
        std = data_arr.std(axis=0)
        ax.plot(all_epochs, mean, "-", color=color, linewidth=2)
        ax.fill_between(all_epochs, mean - std, mean + std, alpha=0.2, color=color)
        ax.axvline(x=E + 0.5, color="gray", linestyle="--", alpha=0.7, label="Round switch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if "Accuracy" in ylabel:
            ax.set_ylim(0, 1.05)

    plt.suptitle(f"GAT Node Classification — Training Dynamics (mean ± std, n={len(SEEDS)})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RES_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_accuracy_comparison(results, summary):
    """Bar chart with error bars: R1 vs R2 vs Joint."""
    fig, ax = plt.subplots(figsize=(12, 7))

    subs = list(SUBREDDITS.keys())
    x_pos = np.arange(len(subs))
    width = 0.25

    # After R1
    r1_means = [np.mean([r["r1_per_class"][s] for r in results["naive_gat"]]) for s in subs]
    r1_stds = [np.std([r["r1_per_class"][s] for r in results["naive_gat"]]) for s in subs]

    # After R2
    r2_means = [np.mean([r["r2_per_class"][s] for r in results["naive_gat"]]) for s in subs]
    r2_stds = [np.std([r["r2_per_class"][s] for r in results["naive_gat"]]) for s in subs]

    # Joint
    jt_means = [np.mean([r["per_class"][s] for r in results["joint_gat"]]) for s in subs]
    jt_stds = [np.std([r["per_class"][s] for r in results["joint_gat"]]) for s in subs]

    ax.bar(x_pos - width, r1_means, width, yerr=r1_stds, capsize=3,
           label="After Round 1 (Python)", color="#2ecc71", edgecolor="white")
    ax.bar(x_pos, r2_means, width, yerr=r2_stds, capsize=3,
           label="After Round 2 (JS fine-tune)", color="#e74c3c", edgecolor="white")
    ax.bar(x_pos + width, jt_means, width, yerr=jt_stds, capsize=3,
           label="Joint Training (upper bound)", color="#3498db", edgecolor="white")

    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(1.0, -0.1, "Python Group", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=11, fontstyle="italic", color="#2ecc71")
    ax.text(4.0, -0.1, "JS Group", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=11, fontstyle="italic", color="#3498db")

    fgt = summary["naive_gat"]["forgetting"]
    ax.set_title(f"Per-Subreddit Accuracy Comparison\n"
                 f"Forgetting = {fgt['mean']:.4f} ± {fgt['std']:.4f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Subreddit")
    ax.set_ylabel("Test Accuracy")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace("learnpython", "learn\npython") for s in subs])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(RES_DIR / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_forgetting_viz(results):
    """Area chart showing Python vs JS accuracy over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    E = EPOCHS_PER_ROUND
    all_epochs = list(range(1, 2 * E + 1))

    runs = results["naive_gat"]
    all_py = np.array([r["history"]["r1_py_acc"] + r["history"]["r2_py_acc"] for r in runs])
    all_js = np.array([r["history"]["r1_js_acc"] + r["history"]["r2_js_acc"] for r in runs])

    py_mean, py_std = all_py.mean(0), all_py.std(0)
    js_mean, js_std = all_js.mean(0), all_js.std(0)

    ax.fill_between(all_epochs, py_mean - py_std, py_mean + py_std, alpha=0.15, color="#2ecc71")
    ax.fill_between(all_epochs, js_mean - js_std, js_mean + js_std, alpha=0.15, color="#3498db")
    ax.plot(all_epochs, py_mean, "-", color="#2ecc71", linewidth=2.5, label="Python Group")
    ax.plot(all_epochs, js_mean, "-", color="#3498db", linewidth=2.5, label="JS Group")

    r1_py = py_mean[E - 1]
    r2_py = py_mean[-1]
    fgt = r1_py - r2_py

    ax.annotate("", xy=(E + 1, r2_py), xytext=(E, r1_py),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2.5))
    ax.annotate(f"Forgetting\n{fgt*100:.1f}%",
                xy=(E + 5, (r1_py + r2_py) / 2),
                fontsize=13, color="#e74c3c", fontweight="bold")

    ax.axvline(x=E + 0.5, color="gray", linestyle="--", alpha=0.7)
    ax.text(E / 2, 1.08, "Round 1\n(Python)", ha="center", fontsize=11)
    ax.text(E + E / 2, 1.08, "Round 2\n(JS)", ha="center", fontsize=11)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"Catastrophic Forgetting (mean ± std, n={len(SEEDS)})",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RES_DIR / "forgetting_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_ablation(results, summary):
    """GAT vs MLP comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Forgetting comparison
    ax = axes[0]
    methods = ["GAT (graph)", "MLP (no graph)"]
    gat_fgt = summary["naive_gat"]["forgetting"]
    mlp_fgt = summary["naive_mlp"]["forgetting"]
    means = [gat_fgt["mean"], mlp_fgt["mean"]]
    stds = [gat_fgt["std"], mlp_fgt["std"]]
    colors = ["#e74c3c", "#95a5a6"]
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, edgecolor="white", width=0.5)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.02,
                f"{m:.3f}±{s:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Forgetting Magnitude")
    ax.set_title("Forgetting: GAT vs MLP", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Right: Task accuracies
    ax = axes[1]
    x = np.arange(4)
    width = 0.35
    gat_vals = [summary["naive_gat"]["r1_python_acc"]["mean"],
                summary["naive_gat"]["r2_python_acc"]["mean"],
                summary["naive_gat"]["r2_js_acc"]["mean"],
                summary["joint_gat"]["python_acc"]["mean"]]
    gat_errs = [summary["naive_gat"]["r1_python_acc"]["std"],
                summary["naive_gat"]["r2_python_acc"]["std"],
                summary["naive_gat"]["r2_js_acc"]["std"],
                summary["joint_gat"]["python_acc"]["std"]]
    mlp_vals = [summary["naive_mlp"]["r1_python_acc"]["mean"],
                summary["naive_mlp"]["r2_python_acc"]["mean"],
                summary["naive_mlp"]["r2_js_acc"]["mean"],
                0]  # no joint MLP
    mlp_errs = [summary["naive_mlp"]["r1_python_acc"]["std"],
                summary["naive_mlp"]["r2_python_acc"]["std"],
                summary["naive_mlp"]["r2_js_acc"]["std"],
                0]

    ax.bar(x - width/2, gat_vals, width, yerr=gat_errs, capsize=3,
           label="GAT", color="#e74c3c", edgecolor="white")
    ax.bar(x + width/2, mlp_vals, width, yerr=mlp_errs, capsize=3,
           label="MLP", color="#95a5a6", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(["R1 Python", "R2 Python", "R2 JS", "Joint Python"], fontsize=10)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("GAT vs MLP — Task Accuracies", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(RES_DIR / "ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def _save_text_summary(results, summary):
    """Human-readable text summary."""
    info = results["data_info"]
    lines = [
        "=" * 65,
        "GNN NODE CLASSIFICATION — CATASTROPHIC FORGETTING BASELINE",
        "Research-Grade Results",
        "=" * 65,
        "",
        "CONFIGURATION",
        "-" * 40,
        f"  Model:          GAT (2-layer, {GAT_HEADS} heads, {HIDDEN_DIM} hidden)",
        f"  Features:       TF-IDF ({TFIDF_MAX_FEATURES} dims)",
        f"  Epochs/round:   {EPOCHS_PER_ROUND}",
        f"  Learning rate:  {LEARNING_RATE}",
        f"  Seeds:          {SEEDS}",
        f"  Loss:           CrossEntropy with class weights",
        "",
        "DATA",
        "-" * 40,
        f"  Total nodes:    {info['total_nodes']}",
        f"  Total edges:    {info['total_edges']}",
        f"  Comment counts per subreddit:",
    ]
    for sub, count in info["comment_counts"].items():
        lines.append(f"    {sub:>12}: {count}")
    lines.extend([
        f"  Edge relations:",
        f"    authored : user → comment  ({info['edge_relations'].get('authored', 0)} edges)",
        f"    on_post  : comment → post  ({info['edge_relations'].get('on_post', 0)} edges)",
        f"    reply_to : comment → comment ({info['edge_relations'].get('reply_to', 0)} edges)",
        f"  Class weights: {[f'{w:.3f}' for w in info['class_weights']]}",
        "",
    ])

    for exp, label in [("naive_gat", "NAIVE SEQUENTIAL — GAT"),
                        ("naive_mlp", "NAIVE SEQUENTIAL — MLP (ablation)"),
                        ("joint_gat", "JOINT TRAINING — GAT (upper bound)")]:
        lines.append(label)
        lines.append("-" * 40)
        s = summary[exp]
        for k, v in s.items():
            lines.append(f"  {k:>20}: {v['mean']:.4f} ± {v['std']:.4f}")

        # Per-class mean
        runs = results[exp]
        pc_key = "per_class" if "per_class" in runs[0] else "r2_per_class"
        if pc_key in runs[0]:
            lines.append(f"  Per-class (final):")
            for sub in SUBREDDITS:
                vals = [r[pc_key][sub] for r in runs]
                lines.append(f"    {sub:>12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        lines.append("")

    lines.extend(["=" * 65])

    with open(RES_DIR / "forgetting_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()