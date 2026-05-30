"""
Evaluation Metrics — Continual Learning for HGNNs (3-task protocol)
====================================================================
Functions for computing standard CL metrics:
  - T1/T2 Forgetting: A_{i,i} - A_{3,i}  (per-task accuracy drop after all tasks)
  - Avg Forgetting:   mean of T1 and T2 forgetting
  - Backward Transfer (BWT): −avg_forgetting
  - Forward Transfer: avg accuracy on a task *before* training on it
                      = (A_{1,2} + A_{2,3}) / 2
  - Recovery Ratio:   A_{3,1} / A_{1,1}  (T1 retention after all tasks)
  - Plasticity:       A_{3,3}  (ability to learn the final task)
  - Avg Final Acc:    (A_{3,1} + A_{3,2} + A_{3,3}) / 3

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
    """Compute all CL metrics from a 3×3 accuracy matrix.

    Args:
        acc_matrix: {
            "A_1_1": float,  # T1 acc after T1 training
            "A_1_2": float,  # T2 acc after T1 training  (fwd transfer to T2)
            "A_1_3": float,  # T3 acc after T1 training  (fwd transfer to T3)
            "A_2_1": float,  # T1 acc after T2 training
            "A_2_2": float,  # T2 acc after T2 training
            "A_2_3": float,  # T3 acc after T2 training  (fwd transfer to T3)
            "A_3_1": float,  # T1 acc after T3 training  (final retention)
            "A_3_2": float,  # T2 acc after T3 training  (final retention)
            "A_3_3": float,  # T3 acc after T3 training  (plasticity)
        }

    Returns:
        dict with all computed metrics
    """
    # After T1
    a11 = acc_matrix.get("A_1_1", 0.0)
    a12 = acc_matrix.get("A_1_2", 0.0)
    a13 = acc_matrix.get("A_1_3", 0.0)
    # After T2
    a21 = acc_matrix.get("A_2_1", 0.0)
    a22 = acc_matrix.get("A_2_2", 0.0)
    a23 = acc_matrix.get("A_2_3", 0.0)
    # After T3 (final)
    a31 = acc_matrix.get("A_3_1", 0.0)
    a32 = acc_matrix.get("A_3_2", 0.0)
    a33 = acc_matrix.get("A_3_3", 0.0)

    t1_fgt = a11 - a31
    t2_fgt = a22 - a32
    avg_fgt = (t1_fgt + t2_fgt) / 2
    fwt = (a12 + a23) / 2          # avg accuracy on a task before training on it
    rec = (a31 / a11) if a11 > 0 else 0.0

    return {
        "t1_forgetting":    t1_fgt,
        "t2_forgetting":    t2_fgt,
        "avg_forgetting":   avg_fgt,
        "backward_transfer": -avg_fgt,
        "forward_transfer": fwt,
        "recovery_ratio":   rec,
        "plasticity":       a33,
        "avg_final_acc":    (a31 + a32 + a33) / 3,
        # raw matrix entries
        "A_1_1": a11, "A_1_2": a12, "A_1_3": a13,
        "A_2_1": a21, "A_2_2": a22, "A_2_3": a23,
        "A_3_1": a31, "A_3_2": a32, "A_3_3": a33,
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
    keys = [
        "t1_forgetting", "t2_forgetting", "avg_forgetting",
        "backward_transfer", "forward_transfer",
        "recovery_ratio", "plasticity", "avg_final_acc",
        "A_1_1", "A_1_2", "A_1_3",
        "A_2_1", "A_2_2", "A_2_3",
        "A_3_1", "A_3_2", "A_3_3",
    ]
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
    metrics = ["avg_forgetting", "recovery_ratio", "plasticity",
               "avg_final_acc", "A_3_1", "A_3_2", "A_3_3"]
    metric_labels = ["Avg Fgt ↓", "T1 Recovery ↑", "Plasticity ↑",
                     "Avg Final Acc ↑", "T1 (after T3)", "T2 (after T3)", "T3 (after T3)"]

    n_methods = len(methods)
    fig, ax = plt.subplots(figsize=(18, 7))  # wider for 7 metrics
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_methods, 1)

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
                           upper_bound: Optional[float] = None,
                           target_task: str = "T1") -> None:
    """Overlay accuracy curves for a specific task across multiple methods.

    Args:
        histories: {method_name: {"t1_acc": [list], "t2_acc": [list]}}
        save_path: where to save
        epochs_per_task: number of epochs per task
        upper_bound: optional horizontal reference line (e.g., joint-training T1 acc)
        target_task: 'T1' or 'T2'
    """
    E = epochs_per_task
    all_epochs = list(range(1, 3 * E + 1))

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for i, (name, h) in enumerate(histories.items()):
        if target_task == "T1":
            curve = h.get("t1_acc", [])
        else:
            curve = h.get("t2_acc", [])
            
        if len(curve) == 3 * E:
            ax.plot(all_epochs, curve, "-", color=colors[i],
                    linewidth=2, label=name)

    if upper_bound is not None:
        ax.axhline(y=upper_bound, color="gold", linestyle="--", linewidth=2,
                   alpha=0.8, label="Joint upper bound")

    ax.axvline(x=E + 0.5, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(x=2 * E + 0.5, color="gray", linestyle=":", alpha=0.7)
    ax.text(E / 2, 1.05, "Task 1\n(Python)", ha="center", fontsize=11)
    ax.text(E + E / 2, 1.05, "Task 2\n(JS)", ha="center", fontsize=11)
    ax.text(2 * E + E / 2, 1.05, "Task 3\n(Politics)", ha="center", fontsize=11)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel(f"{target_task} Accuracy", fontsize=13)
    ax.set_title(f"Forgetting Curves ({target_task}) — All Methods", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
