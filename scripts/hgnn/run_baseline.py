"""
Run Baseline — HGNN Catastrophic Forgetting Baseline
=====================================================
Phase 1 entry point: establishes the forgetting baseline for HGNNs.

Runs:
  1. Topology audit — analyze hypergraph structure
  2. Naive sequential (Finetune) — demonstrate catastrophic forgetting
  3. Joint training — upper bound reference

Produces:
  - Forgetting metrics (A_{1,1}, A_{2,1}, A_{2,2})
  - Per-class accuracy breakdown
  - Training curves and forgetting visualization
  - Results JSON and text summary

Saves to: res/hgnn/baseline/
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from hypergraph_constructor import build_hypergraph_data, validate_hypergraph
from models.hgnn import create_model
from cl_methods.base import FinetuneCLMethod
from training.trainer import HypergraphCLTrainer
from training.joint_trainer import run_joint_training
from evaluation.topology_audit import run_topology_audit
from evaluation.metrics import compute_all_metrics, aggregate_metrics

# ─── Output Directory ───────────────────────────────────────────────────────
BASELINE_DIR = config.OUTPUT_DIR / "baseline"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)


# ─── Plotting ────────────────────────────────────────────────────────────────

def _plot_training_curves(all_results, save_dir):
    """Training curves for the naive sequential experiment."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    E = config.EPOCHS_PER_TASK
    all_epochs = list(range(1, 2 * E + 1))

    runs = all_results["finetune"]
    all_loss = np.array([r["history"]["loss"] for r in runs])
    all_t1 = np.array([r["history"]["t1_acc"] for r in runs])
    all_t2 = np.array([r["history"]["t2_acc"] for r in runs])

    for ax, arr, color, title, ylabel in [
        (axes[0], all_loss, "#e74c3c", "Training Loss", "Loss"),
        (axes[1], all_t1, "#2ecc71", "T1 (Python) Accuracy", "Accuracy"),
        (axes[2], all_t2, "#3498db", "T2 (JS) Accuracy", "Accuracy"),
    ]:
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(all_epochs, mean, "-", color=color, linewidth=2)
        ax.fill_between(all_epochs, mean - std, mean + std, alpha=0.2, color=color)
        ax.axvline(x=E + 0.5, color="gray", linestyle="--", alpha=0.7, label="Task switch")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
        if "Accuracy" in ylabel:
            ax.set_ylim(0, 1.05)

    plt.suptitle(f"HGNN Baseline — Training Dynamics (mean ± std, n={len(runs)})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_forgetting_viz(all_results, save_dir):
    """Forgetting visualization with curves."""
    runs = all_results["finetune"]
    fig, ax = plt.subplots(figsize=(10, 6))
    E = config.EPOCHS_PER_TASK
    all_epochs = list(range(1, 2 * E + 1))

    all_t1 = np.array([r["history"]["t1_acc"] for r in runs])
    all_t2 = np.array([r["history"]["t2_acc"] for r in runs])
    t1_mean, t1_std = all_t1.mean(0), all_t1.std(0)
    t2_mean, t2_std = all_t2.mean(0), all_t2.std(0)

    ax.fill_between(all_epochs, t1_mean - t1_std, t1_mean + t1_std, alpha=0.15, color="#2ecc71")
    ax.fill_between(all_epochs, t2_mean - t2_std, t2_mean + t2_std, alpha=0.15, color="#3498db")
    ax.plot(all_epochs, t1_mean, "-", color="#2ecc71", linewidth=2.5, label="T1 (Python)")
    ax.plot(all_epochs, t2_mean, "-", color="#3498db", linewidth=2.5, label="T2 (JS)")

    r1_t1 = t1_mean[E - 1]
    r2_t1 = t1_mean[-1]
    fgt = r1_t1 - r2_t1

    ax.annotate("", xy=(E + 1, r2_t1), xytext=(E, r1_t1),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2.5))
    ax.annotate(f"Forgetting\n{fgt*100:.1f}%",
                xy=(E + 5, (r1_t1 + r2_t1) / 2),
                fontsize=13, color="#e74c3c", fontweight="bold")

    ax.axvline(x=E + 0.5, color="gray", linestyle="--", alpha=0.7)
    ax.text(E / 2, 1.08, "Task 1\n(Python)", ha="center", fontsize=11)
    ax.text(E + E / 2, 1.08, "Task 2\n(JS)", ha="center", fontsize=11)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"HGNN Catastrophic Forgetting (mean ± std, n={len(runs)})",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.18)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_dir / "forgetting_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_accuracy_comparison(all_results, save_dir):
    """Per-subreddit bar chart: after T1 vs after T2 vs Joint."""
    runs_ft = all_results["finetune"]
    runs_jt = all_results["joint"]
    subs = list(config.SUBREDDITS.keys())

    fig, ax = plt.subplots(figsize=(12, 7))
    x_pos = np.arange(len(subs))
    width = 0.25

    r1_m = [np.mean([r["r1_per_class"][s] for r in runs_ft]) for s in subs]
    r1_s = [np.std([r["r1_per_class"][s] for r in runs_ft]) for s in subs]
    r2_m = [np.mean([r["r2_per_class"][s] for r in runs_ft]) for s in subs]
    r2_s = [np.std([r["r2_per_class"][s] for r in runs_ft]) for s in subs]
    jt_m = [np.mean([r["per_class"][s] for r in runs_jt]) for s in subs]
    jt_s = [np.std([r["per_class"][s] for r in runs_jt]) for s in subs]

    ax.bar(x_pos - width, r1_m, width, yerr=r1_s, capsize=3,
           label="After T1 (Python)", color="#2ecc71", edgecolor="white")
    ax.bar(x_pos, r2_m, width, yerr=r2_s, capsize=3,
           label="After T2 (JS finetune)", color="#e74c3c", edgecolor="white")
    ax.bar(x_pos + width, jt_m, width, yerr=jt_s, capsize=3,
           label="Joint (upper bound)", color="#3498db", edgecolor="white")

    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("HGNN Baseline — Per-Subreddit Accuracy",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Subreddit")
    ax.set_ylabel("Test Accuracy")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace("learnpython", "learn\npython") for s in subs])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_dir / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def _save_text_summary(all_results, ft_summary, jt_summary, save_dir):
    """Save human-readable text summary."""
    lines = [
        "=" * 65,
        "HGNN NODE CLASSIFICATION — CATASTROPHIC FORGETTING BASELINE",
        "=" * 65, "",
        "CONFIGURATION",
        "-" * 40,
        f"  Model:          {config.MODEL_TYPE}",
        f"  Hidden dim:     {config.HIDDEN_DIM}",
        f"  Features:       Sentence-Transformer ({config.NODE_FEATURE_DIM} dims)",
        f"  Epochs/task:    {config.EPOCHS_PER_TASK}",
        f"  Learning rate:  {config.LEARNING_RATE}",
        f"  Seeds:          {config.SEEDS}", "",
        "NAIVE SEQUENTIAL (FINETUNE)",
        "-" * 40,
    ]
    for k, v in ft_summary.items():
        lines.append(f"  {k:>20}: {v['mean']:.4f} ± {v['std']:.4f}")
    lines.append("")
    lines.append("JOINT TRAINING (UPPER BOUND)")
    lines.append("-" * 40)
    for k, v in jt_summary.items():
        lines.append(f"  {k:>20}: {v['mean']:.4f} ± {v['std']:.4f}")
    lines.append("")
    lines.append("=" * 65)

    with open(save_dir / "forgetting_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device(config.DEVICE)
    print(f"Device: {device}")
    print(f"Model: {config.MODEL_TYPE}")
    print(f"Seeds: {config.SEEDS}")
    print(f"Epochs/task: {config.EPOCHS_PER_TASK}\n")

    all_results = {"finetune": [], "joint": []}
    topology_metrics = None

    for i, seed in enumerate(config.SEEDS):
        print(f"\n{'='*60}")
        print(f"SEED {seed} ({i+1}/{len(config.SEEDS)})")
        print(f"{'='*60}")

        # Build hypergraph
        print("\n  Building hypergraph...")
        data, class_weights, info = build_hypergraph_data(seed)
        validate_hypergraph(data, info)

        # Run topology audit (once)
        if topology_metrics is None:
            topology_metrics, predictions = run_topology_audit(data, info)

        # ─── Naive Sequential (Finetune) ─────────────────────────
        print(f"\n  [1/2] Naive Sequential (Finetune)...")
        torch.manual_seed(seed)
        model_ft = create_model(
            config.MODEL_TYPE, data.x.shape[1], config.HIDDEN_DIM,
            config.NUM_CLASSES, config.DROPOUT
        )
        cl_method = FinetuneCLMethod()
        trainer = HypergraphCLTrainer(model_ft, cl_method, config)
        result_ft = trainer.run_experiment(data, class_weights, seed)
        all_results["finetune"].append(result_ft)

        m = result_ft["metrics"]
        print(f"    → T1 Acc: {result_ft['r2_t1_acc']:.4f}  "
              f"T2 Acc: {result_ft['r2_t2_acc']:.4f}  "
              f"Forgetting: {m['forgetting']:.4f}")

        # ─── Joint Training ──────────────────────────────────────
        print(f"\n  [2/2] Joint Training (upper bound)...")
        torch.manual_seed(seed)
        model_jt = create_model(
            config.MODEL_TYPE, data.x.shape[1], config.HIDDEN_DIM,
            config.NUM_CLASSES, config.DROPOUT
        )
        result_jt = run_joint_training(model_jt, data, class_weights, config, seed)
        all_results["joint"].append(result_jt)
        print(f"    → T1: {result_jt['t1_acc']:.4f}  T2: {result_jt['t2_acc']:.4f}")

    # ─── Aggregate ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}\n")

    ft_metrics_list = [r["metrics"] for r in all_results["finetune"]]
    ft_summary = aggregate_metrics(ft_metrics_list)

    jt_summary = {
        "t1_acc": {"mean": float(np.mean([r["t1_acc"] for r in all_results["joint"]])),
                   "std": float(np.std([r["t1_acc"] for r in all_results["joint"]]))},
        "t2_acc": {"mean": float(np.mean([r["t2_acc"] for r in all_results["joint"]])),
                   "std": float(np.std([r["t2_acc"] for r in all_results["joint"]]))},
    }

    print("  Finetune (No Mitigation):")
    for k, v in ft_summary.items():
        print(f"    {k:>20}: {v['mean']:.4f} ± {v['std']:.4f}")
    print(f"\n  Joint Training:")
    for k, v in jt_summary.items():
        print(f"    {k:>20}: {v['mean']:.4f} ± {v['std']:.4f}")

    # ─── Save ────────────────────────────────────────────────────
    print(f"\nSaving results to {BASELINE_DIR}/\n")

    # JSON
    json_data = {
        "method": "baseline",
        "config": {
            "model_type": config.MODEL_TYPE,
            "hidden_dim": config.HIDDEN_DIM,
            "epochs_per_task": config.EPOCHS_PER_TASK,
            "lr": config.LEARNING_RATE,
            "seeds": config.SEEDS,
            "feature_dim": config.NODE_FEATURE_DIM,
            "hyperedge_min_size": config.HYPEREDGE_MIN_SIZE,
            "hyperedge_max_size": config.HYPEREDGE_MAX_SIZE,
        },
        "topology_metrics": topology_metrics,
        "finetune_summary": ft_summary,
        "joint_summary": jt_summary,
        "finetune_results": all_results["finetune"],
        "joint_results": all_results["joint"],
    }
    with open(BASELINE_DIR / "experiment_results.json", "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  ✓ experiment_results.json")

    # Plots
    _plot_training_curves(all_results, BASELINE_DIR)
    print(f"  ✓ training_curves.png")

    _plot_forgetting_viz(all_results, BASELINE_DIR)
    print(f"  ✓ forgetting_visualization.png")

    _plot_accuracy_comparison(all_results, BASELINE_DIR)
    print(f"  ✓ accuracy_comparison.png")

    _save_text_summary(all_results, ft_summary, jt_summary, BASELINE_DIR)
    print(f"  ✓ forgetting_summary.txt")

    print(f"\nAll baseline results saved to {BASELINE_DIR}/")


if __name__ == "__main__":
    main()
