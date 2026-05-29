"""
Trainer — Core CL Training Loop for HGNNs
===========================================
HypergraphCLTrainer handles the full T1→T2→T3 continual learning protocol:
  1. Train on T1 (Python subreddits) for EPOCHS_PER_TASK epochs
  2. Call CL method’s after_task("T1") hook
  3. Call CL method’s before_task("T2") hook for T2
  4. Train on T2 (JS subreddits) for EPOCHS_PER_TASK epochs
  5. Call CL method’s after_task("T2") hook
  6. Call CL method’s before_task("T3") hook for T3
  7. Train on T3 (Politics subreddits) for EPOCHS_PER_TASK epochs
  8. Evaluate on all three tasks after each task

Hypergraph-specific notes:
  - Training uses hyperedge_index instead of edge_index
  - The full graph structure is preserved during training (no subgraph
    sampling) since hyperedge convolutions need the full incidence matrix
  - Evaluation masks select per-task nodes just like in the GNN setup
"""

import json
import warnings
from pathlib import Path
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

warnings.filterwarnings("ignore", category=FutureWarning)


class HypergraphCLTrainer:
    """Core continual learning trainer for hypergraph neural networks.

    Manages the sequential task training protocol and evaluates forgetting.
    Each CL method injects its loss through the compute_loss() hook.
    """

    def __init__(self, model: nn.Module, cl_method, config_module):
        """
        Args:
            model: HGNN backbone (HGNNBackbone or AllDeepSetsBackbone)
            cl_method: BaseCLMethod instance
            config_module: config module with all constants
        """
        self.model = model
        self.cl_method = cl_method
        self.config = config_module
        self.device = torch.device(config_module.DEVICE)
        self.model.to(self.device)

    def train_task(self, task_id: str, data: Data, class_weights: torch.Tensor,
                   epochs: int = None) -> dict:
        """Train the model on a single task.

        Args:
            task_id: "T1", "T2", or "T3"
            data: full hypergraph Data object
            class_weights: [C] class weight tensor
            epochs: override for epochs per task

        Returns:
            dict with per-epoch history
        """
        epochs = epochs or self.config.EPOCHS_PER_TASK
        task_labels = torch.tensor(self.config.TASK_SPLIT[task_id], device=self.device)
        cw = class_weights.to(self.device)
        data = data.to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Task training mask
        task_mask = data.train_mask & torch.isin(data.y, task_labels)

        history = {"loss": [], "t1_acc": [], "t2_acc": [], "t3_acc": []}

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            # Forward pass
            logits = self.model(data.x, data.hyperedge_index,
                                getattr(data, 'hyperedge_weight', None))

            # CL method computes the total loss
            loss = self.cl_method.compute_loss(
                logits, data.y, self.model, data, task_mask, cw
            )

            loss.backward()
            optimizer.step()

            # Evaluate all three tasks every epoch
            t1_acc = self.evaluate(data, "T1")
            t2_acc = self.evaluate(data, "T2")
            t3_acc = self.evaluate(data, "T3")

            history["loss"].append(loss.item())
            history["t1_acc"].append(t1_acc)
            history["t2_acc"].append(t2_acc)
            history["t3_acc"].append(t3_acc)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:3d}/{epochs}  Loss: {loss.item():.4f}  "
                      f"T1: {t1_acc:.4f}  T2: {t2_acc:.4f}  T3: {t3_acc:.4f}")

        return history

    @torch.no_grad()
    def evaluate(self, data: Data, task_id: str) -> float:
        """Evaluate accuracy on a specific task's test nodes.

        Args:
            data: full hypergraph Data object
            task_id: "T1", "T2", or "T3"

        Returns:
            accuracy as float
        """
        self.model.eval()
        data = data.to(self.device)

        task_labels = torch.tensor(self.config.TASK_SPLIT[task_id], device=self.device)
        mask = data.test_mask & torch.isin(data.y, task_labels)

        if mask.sum() == 0:
            return 0.0

        logits = self.model(data.x, data.hyperedge_index,
                            getattr(data, 'hyperedge_weight', None))
        pred = logits[mask].argmax(dim=-1)
        return (pred == data.y[mask]).float().mean().item()

    @torch.no_grad()
    def evaluate_per_class(self, data: Data) -> dict:
        """Evaluate per-class accuracy across all classes.

        Returns:
            dict mapping class_name -> accuracy
        """
        self.model.eval()
        data = data.to(self.device)
        results = {}

        for label_id in range(self.config.NUM_CLASSES):
            name = self.config.LABEL_NAMES.get(label_id, f"class_{label_id}")
            t = torch.tensor([label_id], device=self.device)
            mask = data.test_mask & torch.isin(data.y, t)

            if mask.sum() == 0:
                results[name] = 0.0
                continue

            logits = self.model(data.x, data.hyperedge_index,
                                getattr(data, 'hyperedge_weight', None))
            pred = logits[mask].argmax(dim=-1)
            results[name] = (pred == data.y[mask]).float().mean().item()

        return results

    def run_experiment(self, data: Data, class_weights: torch.Tensor,
                       seed: int) -> dict:
        """Run the full T1→T2→T3 continual learning experiment.

        Protocol:
          1. Train T1 → record A_{1,1}, A_{1,2}, A_{1,3}
          2. after_task("T1") → CL method stores what it needs
          3. before_task("T2") + Train T2 → record A_{2,1}, A_{2,2}, A_{2,3}
          4. after_task("T2") → CL method stores what it needs
          5. before_task("T3") + Train T3 → record A_{3,1}, A_{3,2}, A_{3,3}
          6. Compute all metrics from the 3×3 acc matrix

        Returns:
            dict with full results including history and metrics
        """
        torch.manual_seed(seed)

        # ─── Task 1 ────────────────────────────────────────────
        print(f"\n  Training Task 1 (Python subreddits)...")
        h1 = self.train_task("T1", data, class_weights)

        a11 = self.evaluate(data, "T1")
        a12 = self.evaluate(data, "T2")
        a13 = self.evaluate(data, "T3")
        r1_per_class = self.evaluate_per_class(data)
        print(f"  After T1: T1={a11:.4f}  T2={a12:.4f}  T3={a13:.4f}")

        self.cl_method.after_task("T1", data, self.model)

        # ─── Task 2 ────────────────────────────────────────────
        print(f"\n  Training Task 2 (JS subreddits)...")
        self.cl_method.before_task("T2", data, self.model)
        h2 = self.train_task("T2", data, class_weights)

        a21 = self.evaluate(data, "T1")
        a22 = self.evaluate(data, "T2")
        a23 = self.evaluate(data, "T3")
        r2_per_class = self.evaluate_per_class(data)
        print(f"  After T2: T1={a21:.4f}  T2={a22:.4f}  T3={a23:.4f}")

        self.cl_method.after_task("T2", data, self.model)

        # ─── Task 3 ────────────────────────────────────────────
        print(f"\n  Training Task 3 (Politics subreddits)...")
        self.cl_method.before_task("T3", data, self.model)
        h3 = self.train_task("T3", data, class_weights)

        a31 = self.evaluate(data, "T1")
        a32 = self.evaluate(data, "T2")
        a33 = self.evaluate(data, "T3")
        r3_per_class = self.evaluate_per_class(data)
        print(f"  After T3: T1={a31:.4f}  T2={a32:.4f}  T3={a33:.4f}")

        # ─── Compile results ──────────────────────────────────────
        from evaluation.metrics import compute_all_metrics

        acc_matrix = {
            "A_1_1": a11, "A_1_2": a12, "A_1_3": a13,
            "A_2_1": a21, "A_2_2": a22, "A_2_3": a23,
            "A_3_1": a31, "A_3_2": a32, "A_3_3": a33,
        }
        cl_metrics = compute_all_metrics(acc_matrix)

        combined_history = {
            "t1_acc": h1["t1_acc"] + h2["t1_acc"] + h3["t1_acc"],
            "t2_acc": h1["t2_acc"] + h2["t2_acc"] + h3["t2_acc"],
            "t3_acc": h1["t3_acc"] + h2["t3_acc"] + h3["t3_acc"],
            "loss":   h1["loss"]   + h2["loss"]   + h3["loss"],
        }

        return {
            "method": self.cl_method.name,
            "seed": seed,
            "history": combined_history,
            # 3×3 accuracy matrix (row = after task i, col = task j)
            "r1_t1_acc": a11, "r1_t2_acc": a12, "r1_t3_acc": a13,
            "r2_t1_acc": a21, "r2_t2_acc": a22, "r2_t3_acc": a23,
            "r3_t1_acc": a31, "r3_t2_acc": a32, "r3_t3_acc": a33,
            "r1_per_class": r1_per_class,
            "r2_per_class": r2_per_class,
            "r3_per_class": r3_per_class,
            "metrics": cl_metrics,
        }
