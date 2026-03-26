"""
Experience Replay — Random Buffer Selection
=============================================
During R2, interleave replay of uniformly-random R1 nodes with JS training.

Sweep buffer sizes: [100, 500, 1000, 2000]
Saves to ./res/replay_random/
"""

import json
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# Import shared components from baseline
from main import (
    build_unified_graph, GATClassifier, evaluate, evaluate_per_class,
    train_epoch, SUBREDDITS, LABEL_NAMES, PYTHON_LABELS, JS_LABELS,
    NUM_CLASSES, TFIDF_MAX_FEATURES, HIDDEN_DIM, GAT_HEADS,
    EPOCHS_PER_ROUND, LEARNING_RATE, SEEDS, PROJECT_ROOT,
)

# ─── Config ──────────────────────────────────────────────────────────────────
BUFFER_SIZES = [100, 500, 1000, 2000]
RES_DIR = PROJECT_ROOT / "res" / "replay_random"
RES_DIR.mkdir(parents=True, exist_ok=True)


# ─── Buffer Construction ─────────────────────────────────────────────────────

def build_random_buffer(data, buffer_size, seed):
    """Select buffer_size nodes uniformly at random from R1 training set.
    Returns CPU tensor (moved to device during replay)."""
    r1_labels = torch.tensor(PYTHON_LABELS, device=data.y.device)
    r1_train_mask = data.train_mask & torch.isin(data.y, r1_labels)
    r1_indices = r1_train_mask.nonzero(as_tuple=True)[0].cpu()

    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(len(r1_indices), generator=gen)
    n = min(buffer_size, len(r1_indices))
    selected = r1_indices[perm[:n]]
    return selected


# ─── Replay Training Loop ───────────────────────────────────────────────────

def train_epoch_with_replay(model, data, optimizer, task_labels, class_weights,
                            replay_indices, device):
    """One epoch: standard task loss + replay loss on buffered R1 nodes."""
    model.train()
    data = data.to(device)
    cw = class_weights.to(device)

    # Task loss (JS nodes)
    task_tensor = torch.tensor(task_labels, device=device)
    task_mask = data.train_mask & torch.isin(data.y, task_tensor)

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)

    # JS training loss
    loss_task = F.cross_entropy(out[task_mask], data.y[task_mask], weight=cw)

    # Replay loss on buffered R1 nodes
    buf_idx = replay_indices.to(device)
    loss_replay = F.cross_entropy(out[buf_idx], data.y[buf_idx], weight=cw)

    # Combined loss — equal weighting
    loss = loss_task + loss_replay
    loss.backward()
    optimizer.step()
    return loss.item()


# ─── Experiment Runner ───────────────────────────────────────────────────────

def run_replay_experiment(buffer_size, data, class_weights, device, seed):
    """R1 train → build buffer → R2 train with replay. Returns metrics dict."""
    torch.manual_seed(seed)
    gat_kwargs = dict(in_ch=TFIDF_MAX_FEATURES, hid_ch=HIDDEN_DIM,
                      out_ch=NUM_CLASSES, heads=GAT_HEADS)
    model = GATClassifier(**gat_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    history = {"r1_loss": [], "r1_py_acc": [], "r1_js_acc": [],
               "r2_loss": [], "r2_py_acc": [], "r2_js_acc": []}

    # Round 1 — train on Python subreddits (identical to baseline)
    for _ in range(EPOCHS_PER_ROUND):
        loss = train_epoch(model, data, optimizer, PYTHON_LABELS, class_weights, device)
        history["r1_loss"].append(loss)
        history["r1_py_acc"].append(evaluate(model, data, PYTHON_LABELS, device))
        history["r1_js_acc"].append(evaluate(model, data, JS_LABELS, device))

    r1_per_class = evaluate_per_class(model, data, device)

    # Build replay buffer (on CPU, as required)
    replay_indices = build_random_buffer(data, buffer_size, seed)
    print(f"        Buffer: {len(replay_indices)} nodes "
          f"(requested {buffer_size})")

    # Round 2 — train on JS subreddits with replay
    for _ in range(EPOCHS_PER_ROUND):
        loss = train_epoch_with_replay(
            model, data, optimizer, JS_LABELS, class_weights,
            replay_indices, device)
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
        "buffer_size": buffer_size,
    }


# ─── Aggregation & Saving ───────────────────────────────────────────────────

def aggregate_results(all_results):
    """Compute mean ± std per buffer size."""
    summary = {}
    for buf_size, runs in all_results.items():
        s = {}
        for key in ["r1_python_acc", "r2_python_acc", "r2_js_acc", "forgetting"]:
            vals = [r[key] for r in runs]
            s[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        # Recovery ratio and plasticity
        naive_py = 0.0044  # baseline R2 python acc
        joint_py = 0.9495  # joint training python acc
        joint_js = 0.9422  # joint training JS acc
        s["recovery_ratio"] = {
            "mean": float((s["r2_python_acc"]["mean"] - naive_py) / (joint_py - naive_py)),
            "std": float(s["r2_python_acc"]["std"] / (joint_py - naive_py)),
        }
        s["plasticity"] = {
            "mean": float(s["r2_js_acc"]["mean"] / joint_js),
            "std": float(s["r2_js_acc"]["std"] / joint_js),
        }
        summary[buf_size] = s
    return summary


def save_all(all_results, summary):
    """Save JSON, plots, and text summary."""
    print(f"\nSaving results to {RES_DIR}/\n")

    # 1. JSON
    json_data = {
        "method": "replay_random",
        "config": {
            "buffer_sizes": BUFFER_SIZES,
            "seeds": SEEDS,
            "epochs_per_round": EPOCHS_PER_ROUND,
        },
        "results_by_buffer_size": {
            str(k): v for k, v in all_results.items()
        },
        "summary": {str(k): v for k, v in summary.items()},
    }
    json_path = RES_DIR / "experiment_results.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  ✓ {json_path.name}")

    # 2. Training curves (best buffer size)
    _plot_training_curves(all_results, summary)
    print(f"  ✓ training_curves.png")

    # 3. Accuracy comparison
    _plot_accuracy_comparison(all_results, summary)
    print(f"  ✓ accuracy_comparison.png")

    # 4. Forgetting visualization
    _plot_forgetting_viz(all_results, summary)
    print(f"  ✓ forgetting_visualization.png")

    # 5. Buffer size sweep
    _plot_buffer_sweep(summary)
    print(f"  ✓ buffer_sweep.png")

    # 6. Text summary
    _save_text_summary(summary)
    print(f"  ✓ forgetting_summary.txt")

    print(f"\nAll results saved to {RES_DIR}/")


def _best_buffer(summary):
    """Pick buffer size with highest recovery ratio."""
    return max(summary.keys(), key=lambda k: summary[k]["recovery_ratio"]["mean"])


def _plot_training_curves(all_results, summary):
    """Loss and accuracy curves for the best buffer size."""
    best = _best_buffer(summary)
    runs = all_results[best]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    E = EPOCHS_PER_ROUND
    all_epochs = list(range(1, 2 * E + 1))

    all_loss = np.array([r["history"]["r1_loss"] + r["history"]["r2_loss"] for r in runs])
    all_py = np.array([r["history"]["r1_py_acc"] + r["history"]["r2_py_acc"] for r in runs])
    all_js = np.array([r["history"]["r1_js_acc"] + r["history"]["r2_js_acc"] for r in runs])

    for ax, arr, color, title, ylabel in [
        (axes[0], all_loss, "#e74c3c", "Training Loss", "Loss"),
        (axes[1], all_py, "#2ecc71", "Python Accuracy", "Accuracy"),
        (axes[2], all_js, "#3498db", "JS Accuracy", "Accuracy"),
    ]:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
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

    plt.suptitle(f"Random Replay (buffer={best}) — Training Dynamics "
                 f"(mean ± std, n={len(SEEDS)})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RES_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_accuracy_comparison(all_results, summary):
    """Per-subreddit bar chart for the best buffer size."""
    best = _best_buffer(summary)
    runs = all_results[best]
    subs = list(SUBREDDITS.keys())

    fig, ax = plt.subplots(figsize=(12, 7))
    x_pos = np.arange(len(subs))
    width = 0.35

    r1_means = [np.mean([r["r1_per_class"][s] for r in runs]) for s in subs]
    r1_stds = [np.std([r["r1_per_class"][s] for r in runs]) for s in subs]
    r2_means = [np.mean([r["r2_per_class"][s] for r in runs]) for s in subs]
    r2_stds = [np.std([r["r2_per_class"][s] for r in runs]) for s in subs]

    ax.bar(x_pos - width/2, r1_means, width, yerr=r1_stds, capsize=3,
           label="After Round 1", color="#2ecc71", edgecolor="white")
    ax.bar(x_pos + width/2, r2_means, width, yerr=r2_stds, capsize=3,
           label="After Round 2 (Replay)", color="#e74c3c", edgecolor="white")

    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
    fgt = summary[best]["forgetting"]
    ax.set_title(f"Random Replay (buffer={best}) — Per-Subreddit Accuracy\n"
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


def _plot_forgetting_viz(all_results, summary):
    """Area chart: Python vs JS accuracy over time for best buffer."""
    best = _best_buffer(summary)
    runs = all_results[best]

    fig, ax = plt.subplots(figsize=(10, 6))
    E = EPOCHS_PER_ROUND
    all_epochs = list(range(1, 2 * E + 1))

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
    ax.text(E + E / 2, 1.08, "Round 2\n(JS + Replay)", ha="center", fontsize=11)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"Random Replay (buffer={best}) — Forgetting "
                 f"(mean ± std, n={len(SEEDS)})",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RES_DIR / "forgetting_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_buffer_sweep(summary):
    """Line plot: recovery ratio and JS accuracy vs buffer size."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    sizes = sorted(summary.keys())
    recovery = [summary[s]["recovery_ratio"]["mean"] for s in sizes]
    recovery_std = [summary[s]["recovery_ratio"]["std"] for s in sizes]
    js_acc = [summary[s]["r2_js_acc"]["mean"] for s in sizes]
    js_std = [summary[s]["r2_js_acc"]["std"] for s in sizes]

    ax1.errorbar(sizes, recovery, yerr=recovery_std, marker="o", color="#2ecc71",
                 linewidth=2, capsize=5, label="Recovery Ratio")
    ax1.set_xlabel("Buffer Size", fontsize=12)
    ax1.set_ylabel("Recovery Ratio", fontsize=12, color="#2ecc71")
    ax1.tick_params(axis="y", labelcolor="#2ecc71")

    ax2 = ax1.twinx()
    ax2.errorbar(sizes, js_acc, yerr=js_std, marker="s", color="#3498db",
                 linewidth=2, capsize=5, label="JS Accuracy")
    ax2.set_ylabel("JS Accuracy", fontsize=12, color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("Random Replay — Buffer Size Sweep", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RES_DIR / "buffer_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()


def _save_text_summary(summary):
    """Human-readable text summary."""
    lines = [
        "=" * 65,
        "EXPERIENCE REPLAY — RANDOM BUFFER SELECTION",
        "=" * 65, "",
    ]
    for buf_size in sorted(summary.keys()):
        s = summary[buf_size]
        lines.append(f"Buffer Size: {buf_size}")
        lines.append("-" * 40)
        for key in ["r1_python_acc", "r2_python_acc", "r2_js_acc", "forgetting",
                     "recovery_ratio", "plasticity"]:
            v = s[key]
            lines.append(f"  {key:>20}: {v['mean']:.4f} ± {v['std']:.4f}")
        lines.append("")

    with open(RES_DIR / "forgetting_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Method: Random Experience Replay")
    print(f"Buffer sizes: {BUFFER_SIZES}")
    print(f"Seeds: {SEEDS}")
    print(f"Epochs/round: {EPOCHS_PER_ROUND}\n")

    # Results keyed by buffer size
    all_results = {bs: [] for bs in BUFFER_SIZES}

    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"SEED {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'='*60}")

        data, class_weights, info = build_unified_graph(seed)

        for bs in BUFFER_SIZES:
            print(f"\n  Buffer size: {bs}")
            result = run_replay_experiment(bs, data, class_weights, device, seed)
            all_results[bs].append(result)
            print(f"        R1 Py: {result['r1_python_acc']:.4f}  "
                  f"R2 Py: {result['r2_python_acc']:.4f}  "
                  f"R2 JS: {result['r2_js_acc']:.4f}  "
                  f"Forgetting: {result['forgetting']:.4f}")

    # Aggregate
    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS")
    print(f"{'='*60}\n")
    summary = aggregate_results(all_results)
    for bs in sorted(summary.keys()):
        s = summary[bs]
        print(f"  Buffer {bs}:")
        for k, v in s.items():
            print(f"    {k:>20}: {v['mean']:.4f} ± {v['std']:.4f}")
        print()

    save_all(all_results, summary)


if __name__ == "__main__":
    main()
