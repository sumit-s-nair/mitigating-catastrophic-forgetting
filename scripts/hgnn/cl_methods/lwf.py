"""
Learning without Forgetting (LwF) — For HGNNs
================================================
Standard LwF adapted (minimally) for hypergraph neural networks.

Algorithm:
  1. After T1 training, capture the frozen model's soft logit predictions
  2. During T2 training:
     loss = CE(new_logits_T2, labels_T2) + α * KL(new_logits_T1 || old_logits_T1)
  3. The KL divergence is computed with temperature softening (T=2.0)

Hypergraph-specific notes:
  - LwF is TOPOLOGY-AGNOSTIC: it operates entirely on the output logit space
  - The distillation loss doesn't depend on graph structure at all
  - Works identically to the GNN version since it constrains output predictions
  - The only difference is that the underlying model is an HGNN, so the
    logits are computed via hypergraph convolutions instead of pairwise ones

Original GNN assumption:
  - None broken. LwF makes no structural assumptions.
  - The method constrains the function mapping (x → logits), not the path
    through the graph structure.

Hypothesis:
  LwF should provide similar mitigation on hypergraphs as on GNNs, since
  it regularizes the output space regardless of the underlying topology.
  If HGNN representations are smoother (due to higher-order aggregation),
  LwF might be slightly more effective because the output landscape is
  more stable.

α = 1.0 (distillation weight)
T = 2.0 (temperature for softening)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional

from cl_methods.base import BaseCLMethod


class LwF(BaseCLMethod):
    """Learning without Forgetting for hypergraph continual learning.

    Stores soft predictions from the T1-trained model and uses knowledge
    distillation during T2 training to preserve T1 performance.
    """

    def __init__(self, alpha: float = 1.0, temperature: float = 2.0):
        super().__init__(name="LwF")
        self.alpha = alpha
        self.temperature = temperature
        self.soft_targets: Optional[Tensor] = None
        self.old_task_labels: Optional[list] = None

    def before_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        """No preparation needed — soft targets captured in after_task."""
        pass

    def after_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        """Capture soft logit predictions from the frozen model.

        Stores predictions for ALL nodes (not just T1), since the model's
        knowledge about the feature space should be preserved globally.
        The KD loss will be masked to T1 nodes during T2 training.
        """
        import config

        model.eval()
        device = next(model.parameters()).device
        data = data.to(device)

        with torch.no_grad():
            logits = model(data.x, data.hyperedge_index,
                           getattr(data, 'hyperedge_weight', None))
            self.soft_targets = logits.detach().cpu()

        self.old_task_labels = config.TASK_SPLIT[task_id]
        print(f"        LwF: captured soft targets (α={self.alpha}, T={self.temperature})")

    def compute_loss(self, logits: Tensor, labels: Tensor, model: nn.Module,
                     data: Data, task_mask: Tensor,
                     class_weights: Tensor) -> Tensor:
        """CE on current task + α * KL-divergence distillation on old task.

        The distillation loss is computed only on old task (T1) training nodes
        to preserve the model's predictions on those nodes.
        """
        import config

        device = logits.device

        # Current task CE loss
        if task_mask.sum() == 0:
            loss_task = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss_task = F.cross_entropy(logits[task_mask], labels[task_mask],
                                        weight=class_weights)

        # Knowledge distillation loss
        if self.soft_targets is not None and self.old_task_labels is not None:
            old_labels = torch.tensor(self.old_task_labels, device=device)
            old_mask = data.train_mask & torch.isin(data.y, old_labels)

            if old_mask.sum() > 0:
                T = self.temperature
                teacher_soft = F.log_softmax(
                    self.soft_targets.to(device)[old_mask] / T, dim=1)
                student_soft = F.log_softmax(logits[old_mask] / T, dim=1)

                loss_kd = F.kl_div(
                    student_soft, teacher_soft,
                    reduction="batchmean", log_target=True
                ) * (T ** 2)
            else:
                loss_kd = torch.tensor(0.0, device=device)
        else:
            loss_kd = torch.tensor(0.0, device=device)

        return loss_task + self.alpha * loss_kd
