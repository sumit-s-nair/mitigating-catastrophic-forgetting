import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts" / "hgnn"))
import config
from evaluation.metrics import plot_forgetting_curves
from run_baseline import _plot_accuracy_comparison

def regenerate_plots():
    print("Regenerating baseline accuracy comparison...")
    baseline_json = Path("res/hgnn/baseline/experiment_results.json")
    if baseline_json.exists():
        data = json.loads(baseline_json.read_text())
        formatted_data = {"finetune": data["finetune_results"], "joint": data["joint_results"]}
        _plot_accuracy_comparison(formatted_data, Path("res/hgnn/baseline"))
        print("✓ Saved res/hgnn/baseline/accuracy_comparison.png")
        
    print("\nRegenerating full training forgetting curves...")
    full_json = Path("res/hgnn/full/experiment_results.json")
    if full_json.exists():
        data = json.loads(full_json.read_text())
        all_results = data["all_results"]
        avg_histories = {}
        for method_name, runs in all_results.items():
            if method_name == "HypergraphEWC": continue
            if runs:
                all_t1 = np.array([r["history"]["t1_acc"] for r in runs])
                all_t2 = np.array([r["history"]["t2_acc"] for r in runs])
                avg_histories[method_name] = {
                    "t1_acc": all_t1.mean(axis=0).tolist(),
                    "t2_acc": all_t2.mean(axis=0).tolist(),
                }
        jt_t1_mean = data.get("joint_summary", {}).get("t1_acc", {}).get("mean")
        jt_t2_mean = data.get("joint_summary", {}).get("t2_acc", {}).get("mean")
        
        plot_forgetting_curves(avg_histories, Path("res/hgnn/full/forgetting_curves_t1.png"), 
                               upper_bound=jt_t1_mean, target_task="T1")
        print("✓ Saved res/hgnn/full/forgetting_curves_t1.png")
        
        plot_forgetting_curves(avg_histories, Path("res/hgnn/full/forgetting_curves_t2.png"), 
                               upper_bound=jt_t2_mean, target_task="T2")
        print("✓ Saved res/hgnn/full/forgetting_curves_t2.png")

if __name__ == "__main__":
    regenerate_plots()
