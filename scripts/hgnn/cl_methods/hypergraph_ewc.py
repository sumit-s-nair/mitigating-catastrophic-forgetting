"""
HypergraphEWC — Novel Structure-Aware EWC for Hypergraphs
==========================================================
**This is the novel research contribution.**

Extends standard EWC with structure-aware Fisher Information computation.
Instead of treating all nodes equally when computing the Fisher diagonal,
we weight each node's gradient contribution by its hyperedge participation
pattern.

Key insight:
  Nodes that participate in many small hyperedges are more structurally
  important than nodes in a few large hyperedges, because:
  1. They serve as bridges between multiple distinct groups
  2. Their learned representations encode information about diverse contexts
  3. Forgetting their representations has cascading effects on many hyperedges

Fisher computation:
  Standard EWC:  F_i = (1/N) Σ_n (∂ℓ_n/∂θ_i)^2

  HypergraphEWC: F_i = (1/N) Σ_n w_n * (∂ℓ_n/∂θ_i)^2

  where w_n = Σ_{e: n∈e} |e|^{-1}

  This gives higher weight to nodes in many small hyperedges and lower
  weight to nodes in few large hyperedges (mega-threads). The size
  normalization |e|^{-1} prevents mega-threads from dominating the
  Fisher computation.

Comparison with GraphEWC (GNN version):
  - GraphEWC used:       F_graph = F_standard * (1 + α * mean_degree)
    This is a global scaling — all Fisher entries get the same weight.
  - HypergraphEWC uses:  Per-node weighting based on local hyperedge structure.
    This is a local, node-specific weighting — different nodes contribute
    differently based on their structural role.

Hypothesis:
  HypergraphEWC should outperform standard EWC because:
  1. It preserves parameters that are important for structurally central nodes
  2. The size normalization prevents mega-threads from skewing Fisher
  3. Unlike GraphEWC's global scaling, per-node weighting captures local
     structural importance

  However, if the hypergraph has very uniform structure (all nodes have
  similar participation patterns), HypergraphEWC may degrade to standard EWC.

λ = 5000 (penalty weight, same as standard EWC for fair comparison)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from typing import Dict, Optional

from cl_methods.base import BaseCLMethod


class HypergraphEWC(BaseCLMethod):
    """Structure-aware EWC for hypergraph continual learning.

    Novel contribution: weights Fisher Information by hyperedge participation,
    normalized by hyperedge sizes to avoid mega-thread dominance.
    """

    def __init__(self, lambda_: float = 5000, fisher_samples: int = 2000):
        super().__init__(name="HypergraphEWC")
        self.lambda_ = lambda_
        self.fisher_samples = fisher_samples
        self.fisher: Optional[Dict[str, Tensor]] = None
        self.optimal_params: Optional[Dict[str, Tensor]] = None

    def before_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        pass

    def after_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        """Compute structure-aware Fisher and store optimal parameters."""
        import config

        device = next(model.parameters()).device

        # Store optimal parameters
        self.optimal_params = {
            n: p.data.clone() for n, p in model.named_parameters()
        }

        # Compute hypergraph-weighted Fisher
        print(f"        Computing hypergraph-weighted Fisher (λ={self.lambda_:.0f})...")
        self.fisher = self._compute_hypergraph_fisher(model, data, config, device)

    def _compute_node_weights(self, data: Data, device) -> Tensor:
        """Compute per-node structural importance weights.

        w_n = Σ_{e: n∈e} |e|^{-1}

        Nodes in many small hyperedges get high weight.
        Nodes in few large hyperedges get low weight.
        """
        he_index = data.hyperedge_index
        num_nodes = data.x.shape[0]

        if he_index.shape[1] == 0:
            return torch.ones(num_nodes, device=device)

        node_idx = he_index[0]
        he_idx = he_index[1]
        num_he = int(he_idx.max()) + 1

        # Compute hyperedge sizes
        he_sizes = torch.zeros(num_he, device=device, dtype=torch.float)
        he_sizes.scatter_add_(0, he_idx.to(device),
                              torch.ones(len(he_idx), device=device))
        he_sizes = he_sizes.clamp(min=1)

        # For each incidence (node, hyperedge), compute 1/|e|
        inv_sizes = 1.0 / he_sizes[he_idx.to(device)]

        # Sum inv_sizes per node: w_n = Σ_{e: n∈e} |e|^{-1}
        node_weights = torch.zeros(num_nodes, device=device)
        node_weights.scatter_add_(0, node_idx.to(device), inv_sizes)

        # Normalize to mean=1 to keep Fisher scale consistent with standard EWC
        node_weights = node_weights / node_weights.mean().clamp(min=1e-8)

        return node_weights

    def _compute_hypergraph_fisher(self, model: nn.Module, data: Data,
                                    config_module, device) -> Dict[str, Tensor]:
        """Compute structure-aware Fisher Information Matrix.

        F_i = (1/N) Σ_n w_n * (∂ℓ_n/∂θ_i)^2

        where w_n = Σ_{e: n∈e} |e|^{-1} is the structural importance weight.
        """
        model.eval()
        data = data.to(device)

        # Get node structural weights
        node_weights = self._compute_node_weights(data, device)

        # Get T1 training nodes
        task_labels = torch.tensor(config_module.T1_LABELS, device=device)
        task_mask = data.train_mask & torch.isin(data.y, task_labels)
        task_indices = task_mask.nonzero(as_tuple=True)[0]

        # Subsample
        if self.fisher_samples < len(task_indices):
            perm = torch.randperm(len(task_indices), device=device)[:self.fisher_samples]
            task_indices = task_indices[perm]

        # Initialize Fisher
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

        # Forward pass
        logits = model(data.x, data.hyperedge_index,
                       getattr(data, 'hyperedge_weight', None))

        # Weighted per-sample gradient accumulation
        for idx in task_indices:
            model.zero_grad()
            log_prob = F.log_softmax(logits[idx], dim=0)
            loss = -log_prob[data.y[idx]]
            loss.backward(retain_graph=True)

            w = node_weights[idx].item()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += w * (p.grad.data.clone() ** 2)

        # Average
        n_samples = len(task_indices)
        for key in fisher:
            fisher[key] /= n_samples

        # Print weight statistics
        task_weights = node_weights[task_indices]
        print(f"        Node weights — mean: {task_weights.mean():.3f}, "
              f"std: {task_weights.std():.3f}, "
              f"range: [{task_weights.min():.3f}, {task_weights.max():.3f}]")

        return fisher

    def compute_loss(self, logits: Tensor, labels: Tensor, model: nn.Module,
                     data: Data, task_mask: Tensor,
                     class_weights: Tensor) -> Tensor:
        """CE on current task + λ * structure-weighted EWC penalty."""
        device = logits.device

        # Current task CE
        if task_mask.sum() == 0:
            loss_task = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss_task = F.cross_entropy(logits[task_mask], labels[task_mask],
                                        weight=class_weights)

        # Structure-aware EWC penalty
        if self.fisher is not None and self.optimal_params is not None:
            ewc_loss = torch.tensor(0.0, device=device)
            for n, p in model.named_parameters():
                ewc_loss += (
                    self.fisher[n].to(device) *
                    (p - self.optimal_params[n].to(device)) ** 2
                ).sum()
            ewc_loss = self.lambda_ * ewc_loss
        else:
            ewc_loss = torch.tensor(0.0, device=device)

        return loss_task + ewc_loss
