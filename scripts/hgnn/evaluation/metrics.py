"""
Evaluation Metrics — Continual Learning for HGNNs
===================================================
Functions for computing standard CL metrics:
  - Forgetting: A_{1,1} - A_{2,1}
  - Backward Transfer (BWT): A_{2,1} - A_{1,1} (negative of forgetting)
  - Forward Transfer: A_{1,2} (T2 accuracy before T2 training)
  - Recovery Ratio: A_{2,1} / A_{1,1}
  - Plasticity: A_{2,2} (T2 accuracy after T2 training)

Also includes visualization functions for plotting results across methods.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def forgetting(acc_after_t1_on_t1: float, acc_after_t2_on_t1: float) -> float:
    """Forgetting = A_{1,1} - A_{2,1}.
    Higher = more forgetting. 0 = no forgetting. Negative = improvement."""
    return acc_after_t1_on_t1 - acc_after_t2_on_t1


def backward_transfer(acc_after_t1_on_t1: float, acc_after_t2_on_t1: float) -> float:
    """BWT = A_{2,1} - A_{1,1}.
    Negative = forgetting. 0 = no change. Positive = backward improvement."""
    return acc_after_t2_on_t1 - acc_after_t1_on_t1


def forward_transfer(acc_before_t2_on_t2: float) -> float:
    """Forward Transfer = accuracy on T2 before training on T2.
    Measures knowledge transfer from T1 to unseen T2."""
    return acc_before_t2_on_t2


def recovery_ratio(acc_after_t2_on_t1: float, acc_after_t1_on_t1: float) -> float:
    """Recovery Ratio = A_{2,1} / A_{1,1}.
    1.0 = perfect retention. 0 = complete forgetting."""
    if acc_after_t1_on_t1 == 0:
        return 0.0
    return acc_after_t2_on_t1 / acc_after_t1_on_t1


def plasticity(acc_on_new_task: float) -> float:
    """Plasticity = A_{2,2}, the ability to learn the new task."""
    return acc_on_new_task


def compute_all_metrics(acc_matrix: dict) -> dict:
    """Compute all CL metrics from an accuracy matrix.

    Args:
        acc_matrix: {
            "A_1_1": float,  # T1 accuracy after T1 training
            "A_1_2": float,  # T2 accuracy after T1 training (forward transfer)
            "A_2_1": float,  # T1 accuracy after T2 training
            "A_2_2": float,  # T2 accuracy after T2 training
        }

    Returns:
        dict with all computed metrics
    """
    a11 = acc_matrix["A_1_1"]
    a12 = acc_matrix.get("A_1_2", 0.0)
    a21 = acc_matrix["A_2_1"]
    a22 = acc_matrix["A_2_2"]

    return {
        "forgetting": forgetting(a11, a21),
        "backward_transfer": backward_transfer(a11, a21),
        "forward_transfer": forward_transfer(a12),
        "recovery_ratio": recovery_ratio(a21, a11),
        "plasticity": plasticity(a22),
        "A_1_1": a11,
        "A_1_2": a12,
        "A_2_1": a21,
        "A_2_2": a22,
    }


def _stat(vals: list) -> dict:
    """Compute mean and std for a list of values."""
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def aggregate_metrics(all_runs: List[dict]) -> dict:
    """Aggregate metrics across multiple seeds.

    Args:
        all_runs: list of metric dicts from compute_all_metrics()

    Returns:
        dict with mean and std for each metric
    """
    keys = ["forgetting", "backward_transfer", "forward_transfer",
            "recovery_ratio", "plasticity", "A_1_1", "A_1_2", "A_2_1", "A_2_2"]
    summary = {}
    for key in keys:
        vals = [r[key] for r in all_runs if key in r]
        if vals:
            summary[key] = _stat(vals)
    return summary


def plot_results(results_dict: Dict[str, dict], save_path: Path,
                 title: str = "HGNN CL Methods Comparison") -> None:
    """Bar chart comparing all methods across all metrics.

    Args:
        results_dict: {method_name: aggregated_metrics_dict}
        save_path: where to save the plot
    """
    methods = list(results_dict.keys())
    metrics = ["forgetting", "recovery_ratio", "plasticity", "A_2_1", "A_2_2"]
    metric_labels = ["Forgetting ↓", "Recovery ↑", "Plasticity ↑",
                     "T1 Acc (after T2)", "T2 Acc (after T2)"]

    n_methods = len(methods)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    for i, method_name in enumerate(methods):
        m = results_dict[method_name]
        means = [m.get(metric, {}).get("mean", 0) for metric in metrics]
        stds = [m.get(metric, {}).get("std", 0) for metric in metrics]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               label=method_name, color=colors[i], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_forgetting_curves(histories: Dict[str, dict], save_path: Path,
                           epochs_per_task: int = 50,
                           upper_bound: Optional[float] = None) -> None:
    """Overlay T1 accuracy curves for multiple methods.

    Args:
        histories: {method_name: {"t1_acc": [list], "t2_acc": [list]}}
        save_path: where to save
        epochs_per_task: number of epochs per task
        upper_bound: optional horizontal reference line (e.g., joint-training T1 acc)
    """
    E = epochs_per_task
    all_epochs = list(range(1, 2 * E + 1))

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for i, (name, h) in enumerate(histories.items()):
        t1_curve = h.get("t1_acc", [])
        if len(t1_curve) == 2 * E:
            ax.plot(all_epochs, t1_curve, "-", color=colors[i],
                    linewidth=2, label=name)

    if upper_bound is not None:
        ax.axhline(y=upper_bound, color="gold", linestyle="--", linewidth=2,
                   alpha=0.8, label="Joint upper bound")

    ax.axvline(x=E + 0.5, color="gray", linestyle="--", alpha=0.7)
    ax.text(E / 2, 1.05, "Task 1\n(Python)", ha="center", fontsize=11)
    ax.text(E + E / 2, 1.05, "Task 2\n(JS)", ha="center", fontsize=11)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("T1 Accuracy", fontsize=13)
    ax.set_title("Forgetting Curves — All Methods", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
