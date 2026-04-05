"""
Elastic Weight Consolidation (EWC) — For HGNNs
================================================
Standard EWC with diagonal Fisher Information approximation.

Algorithm:
  1. After T1 training, compute Fisher Information Matrix (diagonal)
  2. Store optimal T1 parameters θ*
  3. During T2 training:
     loss = CE(T2) + λ * Σ_i F_i * (θ_i - θ*_i)^2

Fisher computation:
  - Diagonal approximation via squared gradients of the log-likelihood
  - Computed on T1 training data (subsample for efficiency)
  - Each parameter's importance weighted by how much the loss changes
    when that parameter is perturbed

Hypergraph-specific notes:
  - EWC operates on the PARAMETER SPACE — topology-agnostic at the
    regularization level
  - However, the Fisher Information IS influenced by the topology because
    the gradients flow through the hypergraph convolution layers
  - In HGNNs, parameters in the first layer aggregate over hyperedge
    neighborhoods, meaning Fisher entries for those parameters implicitly
    capture hypergraph structural importance
  - This is a subtle but important distinction from GNN EWC

Original GNN assumption:
  - None fundamentally broken. EWC works on parameters regardless of
    the model architecture.
  - The Fisher diagonal may have different characteristics because HGNN
    gradients involve higher-order aggregations.

Hypothesis:
  Standard EWC should work similarly on HGNNs as on GNNs. The Fisher
  diagonal may be smoother (less sparse) because hypergraph convolutions
  average over larger neighborhoods, potentially making EWC slightly
  more effective at preserving overall knowledge.

λ = 5000 (penalty weight)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from typing import Dict, Optional

from cl_methods.base import BaseCLMethod


class EWC(BaseCLMethod):
    """Elastic Weight Consolidation for hypergraph continual learning.

    Penalizes changes to parameters that were important for T1,
    as measured by the diagonal Fisher Information Matrix.
    """

    def __init__(self, lambda_: float = 5000, fisher_samples: int = 2000):
        super().__init__(name="EWC")
        self.lambda_ = lambda_
        self.fisher_samples = fisher_samples
        self.fisher: Optional[Dict[str, Tensor]] = None
        self.optimal_params: Optional[Dict[str, Tensor]] = None

    def before_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        """No preparation needed."""
        pass

    def after_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        """Compute Fisher Information and store optimal parameters."""
        import config

        device = next(model.parameters()).device

        # Store optimal parameters
        self.optimal_params = {
            n: p.data.clone() for n, p in model.named_parameters()
        }

        # Compute Fisher
        print(f"        Computing Fisher information (λ={self.lambda_:.0f})...")
        self.fisher = self._compute_fisher(model, data, config, device)

    def _compute_fisher(self, model: nn.Module, data: Data,
                        config_module, device) -> Dict[str, Tensor]:
        """Compute diagonal Fisher Information Matrix.

        Fisher_i = E[∂L/∂θ_i)^2] ≈ (1/N) Σ_n (∂ℓ_n/∂θ_i)^2

        where ℓ_n = -log p(y_n | x_n, θ) is the negative log-likelihood
        for node n.
        """
        model.eval()
        data = data.to(device)

        task_labels = torch.tensor(config_module.T1_LABELS, device=device)
        task_mask = data.train_mask & torch.isin(data.y, task_labels)
        task_indices = task_mask.nonzero(as_tuple=True)[0]

        # Subsample for efficiency
        if self.fisher_samples < len(task_indices):
            perm = torch.randperm(len(task_indices), device=device)[:self.fisher_samples]
            task_indices = task_indices[perm]

        # Initialize Fisher
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

        # Forward pass (need gradients through the graph)
        logits = model(data.x, data.hyperedge_index,
                       getattr(data, 'hyperedge_weight', None))

        # Per-sample gradient accumulation
        for idx in task_indices:
            model.zero_grad()
            log_prob = F.log_softmax(logits[idx], dim=0)
            loss = -log_prob[data.y[idx]]
            loss.backward(retain_graph=True)

            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.clone() ** 2

        # Average
        n_samples = len(task_indices)
        for key in fisher:
            fisher[key] /= n_samples

        return fisher

    def compute_loss(self, logits: Tensor, labels: Tensor, model: nn.Module,
                     data: Data, task_mask: Tensor,
                     class_weights: Tensor) -> Tensor:
        """CE on current task + λ * EWC penalty.

        EWC penalty: λ * Σ_i F_i * (θ_i - θ*_i)^2
        """
        device = logits.device

        # Current task CE
        if task_mask.sum() == 0:
            loss_task = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss_task = F.cross_entropy(logits[task_mask], labels[task_mask],
                                        weight=class_weights)

        # EWC penalty
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
