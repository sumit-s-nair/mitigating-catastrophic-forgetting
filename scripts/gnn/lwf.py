"""
Learning without Forgetting (LwF) for GNN
==========================================
Before R2 training, run frozen R1 model on all nodes to get soft logits.
During R2, add knowledge distillation loss: λ * KLDiv(current, saved R1 logits)
for the R1 output heads, with temperature T=2.

Sweep λ: [0.5, 1.0, 2.0, 5.0]
Saves to ./res/lwf/
"""

import json
import warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

from main import (
    build_unified_graph, GATClassifier, evaluate, evaluate_per_class,
    train_epoch, SUBREDDITS, LABEL_NAMES, PYTHON_LABELS, JS_LABELS,
    NUM_CLASSES, TFIDF_MAX_FEATURES, HIDDEN_DIM, GAT_HEADS,
    EPOCHS_PER_ROUND, LEARNING_RATE, SEEDS, PROJECT_ROOT,
)

# ─── Config ──────────────────────────────────────────────────────────────────
LAMBDA_VALUES = [0.5, 1.0, 2.0, 5.0]
TEMPERATURE = 2.0
RES_DIR = PROJECT_ROOT / "res" / "lwf"
RES_DIR.mkdir(parents=True, exist_ok=True)


# ─── LwF Training Loop ──────────────────────────────────────────────────────

@torch.no_grad()
def compute_soft_targets(model, data, device):
    """Run frozen R1 model to capture soft logits for distillation.

    Returns soft logits for ALL nodes (not just R1) so the model's knowledge
    about the feature space is preserved. The KD loss will be applied only
    on R1-labeled nodes via masking.
    """
    model.eval()
    data = data.to(device)
    logits = model(data.x, data.edge_index)
    return logits.detach().cpu()


def train_epoch_lwf(model, data, optimizer, task_labels, class_weights,
                    soft_targets, lam, temperature, device):
    """One epoch: task CE loss + λ * KL-divergence distillation loss.

    KD loss is computed on R1 training nodes only:
    - Teacher: soft_targets (frozen R1 model outputs, softened with T)
    - Student: current model outputs (softened with T)
    - KL(teacher || student) with temperature scaling
    """
    model.train()
    data = data.to(device)
    cw = class_weights.to(device)

    # R1 nodes mask (for distillation)
    r1_labels = torch.tensor(PYTHON_LABELS, device=device)
    r1_mask = data.train_mask & torch.isin(data.y, r1_labels)

    # R2 task mask
    task_tensor = torch.tensor(task_labels, device=device)
    task_mask = data.train_mask & torch.isin(data.y, task_tensor)

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)

    # Standard CE loss on new task (JS)
    loss_task = F.cross_entropy(out[task_mask], data.y[task_mask], weight=cw)

    # Knowledge distillation loss on old task nodes (Python)
    # Soften both teacher and student logits with temperature
    teacher_soft = F.log_softmax(soft_targets.to(device)[r1_mask] / temperature, dim=1)
    student_soft = F.log_softmax(out[r1_mask] / temperature, dim=1)

    # KL divergence: KL(teacher || student)
    # F.kl_div expects log-probs as input, target as regular probs or log-probs
    # Using log_target=True for numerical stability
    loss_kd = F.kl_div(student_soft, teacher_soft, reduction="batchmean",
                       log_target=True) * (temperature ** 2)

    loss = loss_task + lam * loss_kd
    loss.backward()
    optimizer.step()
    return loss.item(), loss_task.item(), loss_kd.item()


# ─── Experiment Runner ───────────────────────────────────────────────────────

def run_lwf_experiment(lam, data, class_weights, device, seed):
    """R1 train → capture soft targets → R2 train with KD loss."""
    torch.manual_seed(seed)
    gat_kwargs = dict(in_ch=TFIDF_MAX_FEATURES, hid_ch=HIDDEN_DIM,
                      out_ch=NUM_CLASSES, heads=GAT_HEADS)
    model = GATClassifier(**gat_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)

    history = {"r1_loss": [], "r1_py_acc": [], "r1_js_acc": [],
               "r2_loss": [], "r2_py_acc": [], "r2_js_acc": [],
               "r2_task_loss": [], "r2_kd_loss": []}

    # Round 1 — identical to baseline
    for _ in range(EPOCHS_PER_ROUND):
        loss = train_epoch(model, data, optimizer, PYTHON_LABELS, class_weights, device)
        history["r1_loss"].append(loss)
        history["r1_py_acc"].append(evaluate(model, data, PYTHON_LABELS, device))
        history["r1_js_acc"].append(evaluate(model, data, JS_LABELS, device))

    r1_per_class = evaluate_per_class(model, data, device)

    # Capture soft targets from frozen R1 model
    soft_targets = compute_soft_targets(model, data, device)

    # Round 2 — train on JS with LwF distillation
    for _ in range(EPOCHS_PER_ROUND):
        loss, loss_task, loss_kd = train_epoch_lwf(
            model, data, optimizer, JS_LABELS, class_weights,
            soft_targets, lam, TEMPERATURE, device)
        history["r2_loss"].append(loss)
        history["r2_task_loss"].append(loss_task)
        history["r2_kd_loss"].append(loss_kd)
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
        "lambda": lam,
    }


# ─── Aggregation & Saving ───────────────────────────────────────────────────

def aggregate_results(all_results):
    summary = {}
    for lam, runs in all_results.items():
        s = {}
        for key in ["r1_python_acc", "r2_python_acc", "r2_js_acc", "forgetting"]:
            vals = [r[key] for r in runs]
            s[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        naive_py, joint_py, joint_js = 0.0044, 0.9495, 0.9422
        s["recovery_ratio"] = {
            "mean": float((s["r2_python_acc"]["mean"] - naive_py) / (joint_py - naive_py)),
            "std": float(s["r2_python_acc"]["std"] / (joint_py - naive_py)),
        }
        s["plasticity"] = {
            "mean": float(s["r2_js_acc"]["mean"] / joint_js),
            "std": float(s["r2_js_acc"]["std"] / joint_js),
        }
        summary[lam] = s
    return summary


def save_all(all_results, summary):
    print(f"\nSaving results to {RES_DIR}/\n")

    json_data = {
        "method": "lwf",
        "config": {"lambda_values": LAMBDA_VALUES, "temperature": TEMPERATURE,
                   "seeds": SEEDS, "epochs_per_round": EPOCHS_PER_ROUND},
        "results_by_lambda": {str(k): v for k, v in all_results.items()},
        "summary": {str(k): v for k, v in summary.items()},
    }
    with open(RES_DIR / "experiment_results.json", "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  ✓ experiment_results.json")

    _plot_training_curves(all_results, summary)
    print(f"  ✓ training_curves.png")

    _plot_accuracy_comparison(all_results, summary)
    print(f"  ✓ accuracy_comparison.png")

    _plot_forgetting_viz(all_results, summary)
    print(f"  ✓ forgetting_visualization.png")

    _plot_lambda_sweep(summary)
    print(f"  ✓ lambda_sweep.png")

    _save_text_summary(summary)
    print(f"  ✓ forgetting_summary.txt")

    print(f"\nAll results saved to {RES_DIR}/")


def _best_lambda(summary):
    """Pick λ with highest recovery ratio."""
    return max(summary.keys(), key=lambda k: summary[k]["recovery_ratio"]["mean"])


def _plot_training_curves(all_results, summary):
    best = _best_lambda(summary)
    runs = all_results[best]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    E = EPOCHS_PER_ROUND
    all_epochs = list(range(1, 2 * E + 1))

    all_loss = np.array([r["history"]["r1_loss"] + r["history"]["r2_loss"] for r in runs])
    all_py = np.array([r["history"]["r1_py_acc"] + r["history"]["r2_py_acc"] for r in runs])
    all_js = np.array([r["history"]["r1_js_acc"] + r["history"]["r2_js_acc"] for r in runs])

    for ax, arr, color, title, ylabel in [
        (axes[0], all_loss, "#e74c3c", "Training Loss (Task + KD)", "Loss"),
        (axes[1], all_py, "#2ecc71", "Python Accuracy", "Accuracy"),
        (axes[2], all_js, "#3498db", "JS Accuracy", "Accuracy"),
    ]:
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(all_epochs, mean, "-", color=color, linewidth=2)
        ax.fill_between(all_epochs, mean - std, mean + std, alpha=0.2, color=color)
        ax.axvline(x=E + 0.5, color="gray", linestyle="--", alpha=0.7, label="Round switch")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
        if "Accuracy" in ylabel: ax.set_ylim(0, 1.05)

    plt.suptitle(f"LwF (λ={best}, T={TEMPERATURE}) — Training Dynamics",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(RES_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_accuracy_comparison(all_results, summary):
    best = _best_lambda(summary)
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
           label="After Round 2 (LwF)", color="#e74c3c", edgecolor="white")

    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
    fgt = summary[best]["forgetting"]
    ax.set_title(f"LwF (λ={best}) — Per-Subreddit Accuracy\n"
                 f"Forgetting = {fgt['mean']:.4f} ± {fgt['std']:.4f}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Subreddit"); ax.set_ylabel("Test Accuracy")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace("learnpython", "learn\npython") for s in subs])
    ax.set_ylim(0, 1.15); ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(RES_DIR / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_forgetting_viz(all_results, summary):
    best = _best_lambda(summary)
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

    r1_py, r2_py = py_mean[E - 1], py_mean[-1]
    fgt = r1_py - r2_py
    ax.annotate("", xy=(E + 1, r2_py), xytext=(E, r1_py),
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2.5))
    ax.annotate(f"Forgetting\n{fgt*100:.1f}%", xy=(E + 5, (r1_py + r2_py) / 2),
                fontsize=13, color="#e74c3c", fontweight="bold")

    ax.axvline(x=E + 0.5, color="gray", linestyle="--", alpha=0.7)
    ax.text(E / 2, 1.08, "Round 1\n(Python)", ha="center", fontsize=11)
    ax.text(E + E / 2, 1.08, "Round 2\n(JS + LwF)", ha="center", fontsize=11)
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"LwF (λ={best}) — Forgetting (mean ± std, n={len(SEEDS)})",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.18); ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RES_DIR / "forgetting_visualization.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_lambda_sweep(summary):
    """Line plot: recovery ratio and JS accuracy vs λ."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    lambdas = sorted(summary.keys())
    recovery = [summary[l]["recovery_ratio"]["mean"] for l in lambdas]
    recovery_std = [summary[l]["recovery_ratio"]["std"] for l in lambdas]
    js_acc = [summary[l]["r2_js_acc"]["mean"] for l in lambdas]
    js_std = [summary[l]["r2_js_acc"]["std"] for l in lambdas]

    ax1.errorbar(lambdas, recovery, yerr=recovery_std, marker="o", color="#2ecc71",
                 linewidth=2, capsize=5, label="Recovery Ratio")
    ax1.set_xlabel("λ (KD weight)", fontsize=12)
    ax1.set_ylabel("Recovery Ratio", fontsize=12, color="#2ecc71")
    ax1.tick_params(axis="y", labelcolor="#2ecc71")
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    ax2.errorbar(lambdas, js_acc, yerr=js_std, marker="s", color="#3498db",
                 linewidth=2, capsize=5, label="JS Accuracy")
    ax2.set_ylabel("JS Accuracy", fontsize=12, color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("LwF — λ Sweep (T=2.0)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RES_DIR / "lambda_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()


def _save_text_summary(summary):
    lines = [
        "=" * 65,
        "LEARNING WITHOUT FORGETTING (LwF)",
        f"Temperature: {TEMPERATURE}",
        "=" * 65, "",
    ]
    for lam in sorted(summary.keys()):
        s = summary[lam]
        lines.append(f"λ = {lam}")
        lines.append("-" * 40)
        for key in ["r1_python_acc", "r2_python_acc", "r2_js_acc", "forgetting",
                     "recovery_ratio", "plasticity"]:
            v = s[key]
            lines.append(f"  {key:>20}: {v['mean']:.4f} ± {v['std']:.4f}")
        lines.append("")

    best = _best_lambda(summary)
    lines.append(f"Best λ: {best} (highest recovery ratio)")
    lines.append("")

    with open(RES_DIR / "forgetting_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Method: Learning without Forgetting (LwF)")
    print(f"λ values: {LAMBDA_VALUES}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Seeds: {SEEDS}")
    print(f"Epochs/round: {EPOCHS_PER_ROUND}\n")

    all_results = {lam: [] for lam in LAMBDA_VALUES}

    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"SEED {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'='*60}")

        data, class_weights, info = build_unified_graph(seed)

        for lam in LAMBDA_VALUES:
            print(f"\n  λ = {lam}")
            result = run_lwf_experiment(lam, data, class_weights, device, seed)
            all_results[lam].append(result)
            print(f"        R1 Py: {result['r1_python_acc']:.4f}  "
                  f"R2 Py: {result['r2_python_acc']:.4f}  "
                  f"R2 JS: {result['r2_js_acc']:.4f}  "
                  f"Forgetting: {result['forgetting']:.4f}")

    print(f"\n{'='*60}")
    print(f"AGGREGATED RESULTS")
    print(f"{'='*60}\n")
    summary = aggregate_results(all_results)
    for lam in sorted(summary.keys()):
        s = summary[lam]
        print(f"  λ = {lam}:")
        for k, v in s.items():
            print(f"    {k:>20}: {v['mean']:.4f} ± {v['std']:.4f}")
        print()

    best = _best_lambda(summary)
    print(f"  Best λ: {best}")

    save_all(all_results, summary)


if __name__ == "__main__":
    main()
