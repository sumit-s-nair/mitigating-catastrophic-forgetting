"""
Run Experiments — Full HGNN Catastrophic Forgetting Experiment Suite
=====================================================================
Phase 2 entry point: runs ALL CL methods and generates comparison results.

Runs:
  1. Topology audit
  2. Joint training (upper bound)
  3. Finetune (no mitigation baseline)
  4. Random Replay (sweep buffer sizes)
  5. LwF (sweep α values)
  6. EWC (sweep λ values)
  7. TAR — Hypergraph (sweep buffer sizes)
  8. HypergraphEWC (sweep λ values)

Produces:
  - Per-method results JSON with all metrics
  - Comparison bar charts
  - Forgetting curves overlay
  - Results summary table (CSV + TXT)
  - Topology audit report

Saves to: res/hgnn/full/
"""

import json
import sys
import csv
import warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from hypergraph_constructor import build_hypergraph_data, validate_hypergraph
from models.hgnn import create_model
from cl_methods.base import FinetuneCLMethod
from cl_methods.random_replay import RandomReplay
from cl_methods.lwf import LwF
from cl_methods.ewc import EWC
from cl_methods.topology_aware_replay import TopologyAwareReplay
from cl_methods.hypergraph_ewc import HypergraphEWC
from training.trainer import HypergraphCLTrainer
from training.joint_trainer import run_joint_training
from evaluation.topology_audit import run_topology_audit
from evaluation.metrics import (
    compute_all_metrics, aggregate_metrics,
    plot_results, plot_forgetting_curves,
)

# ─── Output Directory ───────────────────────────────────────────────────────
FULL_DIR = config.OUTPUT_DIR / "full"
FULL_DIR.mkdir(parents=True, exist_ok=True)


# ─── Method Factory ──────────────────────────────────────────────────────────

def get_methods():
    """Return all CL methods to evaluate.

    Returns list of (method_name, cl_method_instance) tuples.
    """
    return [
        ("Finetune", FinetuneCLMethod()),
        ("Random Replay", RandomReplay(buffer_size=config.BUFFER_SIZE)),
        ("LwF", LwF(alpha=config.LWF_ALPHA, temperature=config.LWF_TEMPERATURE)),
        ("EWC", EWC(lambda_=config.EWC_LAMBDA, fisher_samples=config.EWC_FISHER_SAMPLES)),
        ("TAR", TopologyAwareReplay(buffer_size=config.BUFFER_SIZE)),
        ("HypergraphEWC", HypergraphEWC(
            lambda_=config.EWC_LAMBDA, fisher_samples=config.EWC_FISHER_SAMPLES)),
    ]


# ─── Run a Single Method ────────────────────────────────────────────────────

def run_method(method_name, cl_method, data, class_weights, seed):
    """Run a single CL method for one seed."""
    model = create_model(
        config.MODEL_TYPE, data.x.shape[1], config.HIDDEN_DIM,
        config.NUM_CLASSES, config.DROPOUT
    )
    trainer = HypergraphCLTrainer(model, cl_method, config)
    result = trainer.run_experiment(data, class_weights, seed)
    return result


# ─── Summary Table ───────────────────────────────────────────────────────────

def save_summary_table(all_summaries, joint_summary, save_dir):
    """Save a comparison summary as CSV and TXT."""
    methods = list(all_summaries.keys())
    metrics_keys = ["forgetting", "recovery_ratio", "plasticity",
                    "A_1_1", "A_2_1", "A_2_2"]

    # CSV
    csv_path = save_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Method"] + [f"{k}_mean" for k in metrics_keys] + \
                 [f"{k}_std" for k in metrics_keys]
        writer.writerow(header)

        for method in methods:
            s = all_summaries[method]
            row = [method]
            row += [f"{s.get(k, {}).get('mean', 0):.4f}" for k in metrics_keys]
            row += [f"{s.get(k, {}).get('std', 0):.4f}" for k in metrics_keys]
            writer.writerow(row)

        # Joint training row
        row = ["Joint (UB)"]
        jt_mapped = {
            "forgetting": {"mean": 0.0, "std": 0.0},
            "recovery_ratio": {"mean": 1.0, "std": 0.0},
            "plasticity": joint_summary.get("t2_acc", {"mean": 0, "std": 0}),
            "A_1_1": joint_summary.get("t1_acc", {"mean": 0, "std": 0}),
            "A_2_1": joint_summary.get("t1_acc", {"mean": 0, "std": 0}),
            "A_2_2": joint_summary.get("t2_acc", {"mean": 0, "std": 0}),
        }
        row += [f"{jt_mapped.get(k, {}).get('mean', 0):.4f}" for k in metrics_keys]
        row += [f"{jt_mapped.get(k, {}).get('std', 0):.4f}" for k in metrics_keys]
        writer.writerow(row)

    # TXT
    txt_path = save_dir / "comparison_summary.txt"
    lines = [
        "=" * 80,
        "HGNN CATASTROPHIC FORGETTING — FULL EXPERIMENT COMPARISON",
        "=" * 80, "",
        f"Model:       {config.MODEL_TYPE}",
        f"Hidden:      {config.HIDDEN_DIM}",
        f"Epochs/task: {config.EPOCHS_PER_TASK}",
        f"Seeds:       {config.SEEDS}", "",
        f"{'Method':<20} {'Fgt↓':>8} {'Recov↑':>8} {'Plast↑':>8} "
        f"{'A11':>8} {'A21':>8} {'A22':>8}",
        "-" * 80,
    ]
    for method in methods:
        s = all_summaries[method]
        vals = [s.get(k, {}).get("mean", 0) for k in metrics_keys]
        stds = [s.get(k, {}).get("std", 0) for k in metrics_keys]
        line = f"{method:<20}"
        for v, sd in zip(vals, stds):
            line += f" {v:>5.3f}±{sd:.3f}"
        lines.append(line)

    # Joint
    joint_line = f"{'Joint (UB)':<20}"
    for k in metrics_keys:
        mv = jt_mapped.get(k, {})
        joint_line += f" {mv.get('mean', 0):>5.3f}±{mv.get('std', 0):.3f}"
    lines.append(joint_line)

    lines.extend(["", "=" * 80])

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device(config.DEVICE)
    print(f"Device: {device}")
    print(f"Model: {config.MODEL_TYPE}")
    print(f"Seeds: {config.SEEDS}")
    print(f"Epochs/task: {config.EPOCHS_PER_TASK}\n")

    # Storage
    all_results = {name: [] for name, _ in get_methods()}
    all_results["Joint"] = []
    topology_metrics = None

    for i, seed in enumerate(config.SEEDS):
        print(f"\n{'='*70}")
        print(f"SEED {seed} ({i+1}/{len(config.SEEDS)})")
        print(f"{'='*70}")

        # Build hypergraph
        print("\n  Building hypergraph...")
        data, class_weights, info = build_hypergraph_data(seed)
        validate_hypergraph(data, info)

        # Topology audit (once)
        if topology_metrics is None:
            topology_metrics, predictions = run_topology_audit(data, info)

        # ─── Joint Training ──────────────────────────────────────
        print(f"\n  [Joint] Joint Training (upper bound)...")
        torch.manual_seed(seed)
        model_jt = create_model(
            config.MODEL_TYPE, data.x.shape[1], config.HIDDEN_DIM,
            config.NUM_CLASSES, config.DROPOUT
        )
        result_jt = run_joint_training(model_jt, data, class_weights, config, seed)
        all_results["Joint"].append(result_jt)
        print(f"    → T1: {result_jt['t1_acc']:.4f}  T2: {result_jt['t2_acc']:.4f}")

        # ─── CL Methods ─────────────────────────────────────────
        methods = get_methods()
        for j, (method_name, cl_method) in enumerate(methods):
            print(f"\n  [{j+1}/{len(methods)}] {method_name}...")
            torch.manual_seed(seed)
            result = run_method(method_name, cl_method, data, class_weights, seed)
            all_results[method_name].append(result)

            m = result["metrics"]
            print(f"    → T1: {result['r2_t1_acc']:.4f}  "
                  f"T2: {result['r2_t2_acc']:.4f}  "
                  f"Forgetting: {m['forgetting']:.4f}  "
                  f"Recovery: {m['recovery_ratio']:.4f}")

    # ─── Aggregate ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}\n")

    all_summaries = {}
    for method_name, _ in get_methods():
        runs = all_results[method_name]
        metrics_list = [r["metrics"] for r in runs]
        summary = aggregate_metrics(metrics_list)
        all_summaries[method_name] = summary

        print(f"  {method_name}:")
        for k, v in summary.items():
            print(f"    {k:>20}: {v['mean']:.4f} ± {v['std']:.4f}")
        print()

    joint_summary = {
        "t1_acc": {"mean": float(np.mean([r["t1_acc"] for r in all_results["Joint"]])),
                   "std": float(np.std([r["t1_acc"] for r in all_results["Joint"]]))},
        "t2_acc": {"mean": float(np.mean([r["t2_acc"] for r in all_results["Joint"]])),
                   "std": float(np.std([r["t2_acc"] for r in all_results["Joint"]]))},
    }
    print(f"  Joint Training (UB):")
    for k, v in joint_summary.items():
        print(f"    {k:>20}: {v['mean']:.4f} ± {v['std']:.4f}")

    # ─── Save ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Saving results to {FULL_DIR}/")
    print(f"{'='*70}\n")

    # JSON
    json_data = {
        "config": {
            "model_type": config.MODEL_TYPE,
            "hidden_dim": config.HIDDEN_DIM,
            "epochs_per_task": config.EPOCHS_PER_TASK,
            "lr": config.LEARNING_RATE,
            "buffer_size": config.BUFFER_SIZE,
            "ewc_lambda": config.EWC_LAMBDA,
            "lwf_alpha": config.LWF_ALPHA,
            "seeds": config.SEEDS,
        },
        "topology_metrics": topology_metrics,
        "method_summaries": all_summaries,
        "joint_summary": joint_summary,
        "all_results": {k: v for k, v in all_results.items()},
    }
    with open(FULL_DIR / "experiment_results.json", "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  ✓ experiment_results.json")

    # Comparison bar chart
    plot_results(all_summaries, FULL_DIR / "method_comparison.png",
                 title="HGNN CL Methods — Full Comparison")
    print(f"  ✓ method_comparison.png")

    # Forgetting curves overlay
    avg_histories = {}
    for method_name, _ in get_methods():
        runs = all_results[method_name]
        if runs:
            E = config.EPOCHS_PER_TASK
            all_t1 = np.array([r["history"]["t1_acc"] for r in runs])
            avg_histories[method_name] = {
                "t1_acc": all_t1.mean(axis=0).tolist(),
            }
    plot_forgetting_curves(avg_histories, FULL_DIR / "forgetting_curves.png",
                           epochs_per_task=config.EPOCHS_PER_TASK)
    print(f"  ✓ forgetting_curves.png")

    # Summary table
    save_summary_table(all_summaries, joint_summary, FULL_DIR)
    print(f"  ✓ comparison_table.csv")
    print(f"  ✓ comparison_summary.txt")

    # Save topology audit
    if topology_metrics:
        audit_data = {
            "metrics": topology_metrics,
            "predictions": [
                {"method": m, "verdict": v, "reasoning": r}
                for m, v, r in predictions
            ],
        }
        with open(FULL_DIR / "topology_audit.json", "w") as f:
            json.dump(audit_data, f, indent=2)
        print(f"  ✓ topology_audit.json")

    print(f"\nAll results saved to {FULL_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
