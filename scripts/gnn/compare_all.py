"""
Unified Comparison of All Continual Learning Methods
=====================================================
Loads all ./res/<method>/experiment_results.json files and produces:
  1. Unified comparison table (recovery ratio + plasticity)
  2. Stability-plasticity tradeoff scatter plot
  3. Forgetting curves overlay (all methods, mean ± std)

Saves to ./res/comparison/
"""

import json
import warnings
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RES_DIR = PROJECT_ROOT / "res" / "comparison"
RES_DIR.mkdir(parents=True, exist_ok=True)

# Baseline reference values
NAIVE_PY_ACC = 0.0044
JOINT_PY_ACC = 0.9495
JOINT_JS_ACC = 0.9422
EPOCHS_PER_ROUND = 60


# ─── Method Definitions ─────────────────────────────────────────────────────

# Each method: (dir_name, display_name, how to extract best config results)
# Results JSON structure varies by method type (buffer-sweep vs lambda-sweep)

def load_method_results():
    """Load all method results and extract best-config metrics."""
    methods = OrderedDict()

    # ── Baseline (naive sequential) ──
    baseline_path = PROJECT_ROOT / "res" / "gnn_baseline" / "experiment_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        runs = baseline["naive_gat"]
        methods["Naive (no CL)"] = _extract_naive_metrics(runs)

    # ── Replay methods (buffer-sweep) ──
    for dir_name, display_name in [
        ("replay_random", "ER Random"),
        ("replay_stratified", "ER Stratified"),
        ("replay_herding", "ER Herding"),
        ("tar_high_degree", "TAR High-Degree"),
        ("tar_bridge", "TAR Bridge"),
        ("tar_ego", "TAR Ego"),
    ]:
        path = PROJECT_ROOT / "res" / dir_name / "experiment_results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            best = _find_best_buffer(data)
            methods[f"{display_name} (buf={best})"] = _extract_buffer_metrics(data, best)

    # ── Lambda-sweep methods ──
    for dir_name, display_name in [
        ("lwf", "LwF"),
        ("ewc", "EWC"),
    ]:
        path = PROJECT_ROOT / "res" / dir_name / "experiment_results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            best = _find_best_lambda(data)
            methods[f"{display_name} (λ={best})"] = _extract_lambda_metrics(data, best)

    # ── GraphEWC (alpha-sweep) ──
    path = PROJECT_ROOT / "res" / "graph_ewc" / "experiment_results.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        best = _find_best_alpha(data)
        methods[f"GraphEWC (α={best})"] = _extract_alpha_metrics(data, best)

    # ── Joint training (upper bound) ──
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        runs = baseline["joint_gat"]
        methods["Joint (upper bound)"] = _extract_joint_metrics(runs)

    return methods


def _extract_naive_metrics(runs):
    """Extract metrics from naive sequential runs."""
    py_r2 = [r["r2_python_acc"] for r in runs]
    js_r2 = [r["r2_js_acc"] for r in runs]
    py_r1 = [r["r1_python_acc"] for r in runs]
    fgt = [r["forgetting"] for r in runs]
    histories = [(r["history"]["r1_py_acc"] + r["history"]["r2_py_acc"],
                  r["history"]["r1_js_acc"] + r["history"]["r2_js_acc"]) for r in runs]
    return {
        "r1_py": _stat(py_r1), "r2_py": _stat(py_r2), "r2_js": _stat(js_r2),
        "forgetting": _stat(fgt),
        "recovery": _stat([(p - NAIVE_PY_ACC) / (JOINT_PY_ACC - NAIVE_PY_ACC) for p in py_r2]),
        "plasticity": _stat([j / JOINT_JS_ACC for j in js_r2]),
        "histories": histories,
    }


def _extract_joint_metrics(runs):
    """Extract metrics from joint training runs."""
    py = [r["python_acc"] for r in runs]
    js = [r["js_acc"] for r in runs]
    return {
        "r1_py": _stat(py), "r2_py": _stat(py), "r2_js": _stat(js),
        "forgetting": _stat([0.0] * len(py)),
        "recovery": _stat([1.0] * len(py)),
        "plasticity": _stat([j / JOINT_JS_ACC for j in js]),
        "histories": None,
    }


def _extract_buffer_metrics(data, best_buf):
    """Extract metrics for best buffer size from replay/TAR results."""
    runs = data["results_by_buffer_size"][str(best_buf)]
    py_r2 = [r["r2_python_acc"] for r in runs]
    js_r2 = [r["r2_js_acc"] for r in runs]
    py_r1 = [r["r1_python_acc"] for r in runs]
    fgt = [r["forgetting"] for r in runs]
    histories = [(r["history"]["r1_py_acc"] + r["history"]["r2_py_acc"],
                  r["history"]["r1_js_acc"] + r["history"]["r2_js_acc"]) for r in runs]
    return {
        "r1_py": _stat(py_r1), "r2_py": _stat(py_r2), "r2_js": _stat(js_r2),
        "forgetting": _stat(fgt),
        "recovery": _stat([(p - NAIVE_PY_ACC) / (JOINT_PY_ACC - NAIVE_PY_ACC) for p in py_r2]),
        "plasticity": _stat([j / JOINT_JS_ACC for j in js_r2]),
        "histories": histories,
    }


def _extract_lambda_metrics(data, best_lam):
    """Extract metrics for best λ from LwF/EWC results."""
    runs = data["results_by_lambda"][str(best_lam)]
    py_r2 = [r["r2_python_acc"] for r in runs]
    js_r2 = [r["r2_js_acc"] for r in runs]
    py_r1 = [r["r1_python_acc"] for r in runs]
    fgt = [r["forgetting"] for r in runs]
    histories = [(r["history"]["r1_py_acc"] + r["history"]["r2_py_acc"],
                  r["history"]["r1_js_acc"] + r["history"]["r2_js_acc"]) for r in runs]
    return {
        "r1_py": _stat(py_r1), "r2_py": _stat(py_r2), "r2_js": _stat(js_r2),
        "forgetting": _stat(fgt),
        "recovery": _stat([(p - NAIVE_PY_ACC) / (JOINT_PY_ACC - NAIVE_PY_ACC) for p in py_r2]),
        "plasticity": _stat([j / JOINT_JS_ACC for j in js_r2]),
        "histories": histories,
    }


def _extract_alpha_metrics(data, best_alpha):
    """Extract metrics for best α from GraphEWC results."""
    runs = data["results_by_alpha"][str(best_alpha)]
    py_r2 = [r["r2_python_acc"] for r in runs]
    js_r2 = [r["r2_js_acc"] for r in runs]
    py_r1 = [r["r1_python_acc"] for r in runs]
    fgt = [r["forgetting"] for r in runs]
    histories = [(r["history"]["r1_py_acc"] + r["history"]["r2_py_acc"],
                  r["history"]["r1_js_acc"] + r["history"]["r2_js_acc"]) for r in runs]
    return {
        "r1_py": _stat(py_r1), "r2_py": _stat(py_r2), "r2_js": _stat(js_r2),
        "forgetting": _stat(fgt),
        "recovery": _stat([(p - NAIVE_PY_ACC) / (JOINT_PY_ACC - NAIVE_PY_ACC) for p in py_r2]),
        "plasticity": _stat([j / JOINT_JS_ACC for j in js_r2]),
        "histories": histories,
    }


def _stat(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def _find_best_buffer(data):
    """Find buffer size with best recovery ratio."""
    summary = data.get("summary", {})
    best_buf = None
    best_score = -1
    for buf_str, s in summary.items():
        score = s.get("recovery_ratio", {}).get("mean", 0)
        if score > best_score:
            best_score = score
            best_buf = buf_str
    return best_buf


def _find_best_lambda(data):
    """Find λ with best recovery ratio."""
    summary = data.get("summary", {})
    best_lam = None
    best_score = -1
    for lam_str, s in summary.items():
        score = s.get("recovery_ratio", {}).get("mean", 0)
        if score > best_score:
            best_score = score
            best_lam = lam_str
    return best_lam


def _find_best_alpha(data):
    """Find α with best recovery+plasticity sum."""
    summary = data.get("summary", {})
    best_alpha = None
    best_score = -1
    for a_str, s in summary.items():
        score = s.get("recovery_ratio", {}).get("mean", 0) + s.get("plasticity", {}).get("mean", 0)
        if score > best_score:
            best_score = score
            best_alpha = a_str
    return best_alpha


# ─── Outputs ─────────────────────────────────────────────────────────────────

def print_comparison_table(methods):
    """Print and save unified comparison table."""
    print(f"\n{'='*100}")
    print(f"UNIFIED COMPARISON TABLE")
    print(f"{'='*100}")
    header = f"{'Method':<30} {'R1 Py Acc':>12} {'R2 Py Acc':>12} {'R2 JS Acc':>12} {'Forgetting':>12} {'Recovery':>12} {'Plasticity':>12}"
    print(header)
    print("-" * 100)

    lines = [header, "-" * 100]
    for name, m in methods.items():
        row = (f"{name:<30} "
               f"{m['r1_py']['mean']:.4f}±{m['r1_py']['std']:.4f} "
               f"{m['r2_py']['mean']:.4f}±{m['r2_py']['std']:.4f} "
               f"{m['r2_js']['mean']:.4f}±{m['r2_js']['std']:.4f} "
               f"{m['forgetting']['mean']:.4f}±{m['forgetting']['std']:.4f} "
               f"{m['recovery']['mean']:.4f}±{m['recovery']['std']:.4f} "
               f"{m['plasticity']['mean']:.4f}±{m['plasticity']['std']:.4f}")
        print(row)
        lines.append(row)

    # Save
    with open(RES_DIR / "comparison_table.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  ✓ comparison_table.txt")
    return lines


def plot_stability_plasticity_scatter(methods):
    """Scatter plot: x=plasticity, y=recovery ratio, one point per method."""
    fig, ax = plt.subplots(figsize=(12, 9))

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'X', 'P']

    for i, (name, m) in enumerate(methods.items()):
        ax.errorbar(
            m["plasticity"]["mean"], m["recovery"]["mean"],
            xerr=m["plasticity"]["std"], yerr=m["recovery"]["std"],
            marker=markers[i % len(markers)], color=colors[i],
            markersize=12, capsize=5, linewidth=2,
            label=name, markeredgecolor="white", markeredgewidth=1.5,
        )

    # Reference lines
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Perfect recovery")
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5, label="Perfect plasticity")

    # Ideal region
    ax.annotate("Ideal\n(top-right)", xy=(0.98, 1.02), fontsize=11,
                color="green", fontweight="bold", ha="center")

    ax.set_xlabel("Plasticity (JS Acc / Joint JS Acc)", fontsize=13)
    ax.set_ylabel("Recovery Ratio ((Py Acc − Naive) / (Joint − Naive))", fontsize=13)
    ax.set_title("Stability-Plasticity Tradeoff", fontsize=15, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)

    plt.tight_layout()
    fig.savefig(RES_DIR / "stability_plasticity_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ stability_plasticity_scatter.png")


def plot_forgetting_curves_overlay(methods):
    """Overlay all methods' Python accuracy curves (mean ± std)."""
    fig, ax = plt.subplots(figsize=(14, 8))
    E = EPOCHS_PER_ROUND
    all_epochs = list(range(1, 2 * E + 1))

    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for i, (name, m) in enumerate(methods.items()):
        if m["histories"] is None:
            continue  # skip joint (no sequential history)
        py_curves = np.array([h[0] for h in m["histories"]])
        mean = py_curves.mean(axis=0)
        std = py_curves.std(axis=0)
        ax.plot(all_epochs, mean, "-", color=colors[i], linewidth=2, label=name)
        ax.fill_between(all_epochs, mean - std, mean + std, alpha=0.1, color=colors[i])

    # Joint upper bound line
    for name, m in methods.items():
        if "Joint" in name:
            ax.axhline(y=m["r2_py"]["mean"], color="gold", linestyle="--",
                       linewidth=2, alpha=0.7, label="Joint upper bound")
            break

    ax.axvline(x=E + 0.5, color="gray", linestyle="--", alpha=0.7)
    ax.text(E / 2, 1.05, "Round 1 (Python)", ha="center", fontsize=12)
    ax.text(E + E / 2, 1.05, "Round 2 (JS)", ha="center", fontsize=12)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Python Accuracy", fontsize=13)
    ax.set_title("Forgetting Curves — All Methods Overlay", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RES_DIR / "forgetting_curves_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ forgetting_curves_overlay.png")


def save_comparison_json(methods):
    """Save full comparison data as JSON."""
    # Strip histories for JSON (too large)
    clean = {}
    for name, m in methods.items():
        clean[name] = {k: v for k, v in m.items() if k != "histories"}
    with open(RES_DIR / "comparison_results.json", "w") as f:
        json.dump(clean, f, indent=2)
    print(f"  ✓ comparison_results.json")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading all method results...")
    methods = load_method_results()
    print(f"Found {len(methods)} methods\n")

    print_comparison_table(methods)
    plot_stability_plasticity_scatter(methods)
    plot_forgetting_curves_overlay(methods)
    save_comparison_json(methods)

    print(f"\nAll comparison outputs saved to {RES_DIR}/")


if __name__ == "__main__":
    main()
