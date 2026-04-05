"""
CL Methods — Base Class and Finetune Baseline
===============================================
Abstract base class for all continual learning methods in the HGNN pipeline.

All CL methods must implement these hooks:
  - before_task(): prepare for a new task (e.g., save old model for LwF)
  - after_task(): post-task processing (e.g., compute Fisher for EWC)
  - compute_loss(): return the total loss including any CL penalty
  - update_buffer(): for replay methods, update the exemplar buffer

The FinetuneCLMethod is the "no mitigation" baseline — it applies no CL
strategy, demonstrating pure catastrophic forgetting.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data


class BaseCLMethod(ABC):
    """Abstract base class for continual learning methods on hypergraphs.

    Each method modifies the training loop by injecting additional losses
    or constraints. The trainer calls these hooks at the appropriate points.

    Hypergraph-specific note:
      Standard CL methods (LwF, EWC) operate either on the output space
      or the parameter space and are topology-agnostic. Topology-aware
      methods (TAR, HypergraphEWC) must be adapted for the hypergraph
      structure. Each subclass documents what assumptions might break.
    """

    def __init__(self, name: str = "BaseCL"):
        self.name = name

    @abstractmethod
    def before_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        """Called before training on a new task.

        Typical uses:
          - LwF: capture old model's predictions for distillation
          - Replay: no-op (buffer already built from previous after_task)
        """
        pass

    @abstractmethod
    def after_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        """Called after training on a task completes.

        Typical uses:
          - EWC: compute Fisher Information Matrix
          - Replay: fill/update the exemplar buffer
        """
        pass

    @abstractmethod
    def compute_loss(self, logits: Tensor, labels: Tensor, model: nn.Module,
                     data: Data, task_mask: Tensor,
                     class_weights: Tensor) -> Tensor:
        """Compute the total training loss including any CL penalty.

        Args:
            logits: [N, C] model output for all nodes
            labels: [N] ground truth labels
            model: current model (needed for EWC penalty on params)
            data: full Data object (needed for replay node access)
            task_mask: [N] boolean mask for current task's training nodes
            class_weights: [C] class weights for CE loss

        Returns:
            Total loss scalar
        """
        pass

    def update_buffer(self, task_id: str, data: Data, model: nn.Module) -> None:
        """For replay-based methods: update the exemplar buffer.
        Default: no-op for non-replay methods."""
        pass


class FinetuneCLMethod(BaseCLMethod):
    """No Mitigation Baseline — pure sequential finetuning.

    Simply trains on the new task with standard cross-entropy loss.
    This demonstrates catastrophic forgetting without any mitigation.

    Hypothesis for hypergraphs:
      We expect similar catastrophic forgetting as in the GNN baseline
      (~97% T1 accuracy dropping to near-zero after T2 training).
      The higher-order structure of hypergraphs is unlikely to prevent
      forgetting since the model weights are still overwritten by T2 gradients.
    """

    def __init__(self):
        super().__init__(name="Finetune (No Mitigation)")

    def before_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        pass  # no-op

    def after_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        pass  # no-op

    def compute_loss(self, logits: Tensor, labels: Tensor, model: nn.Module,
                     data: Data, task_mask: Tensor,
                     class_weights: Tensor) -> Tensor:
        """Standard cross-entropy loss on current task nodes only."""
        if task_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return F.cross_entropy(logits[task_mask], labels[task_mask],
                               weight=class_weights)
