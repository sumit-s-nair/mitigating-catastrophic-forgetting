"""
Random Experience Replay — Adapted for Hypergraphs
====================================================
Buffer stores (node_idx, task_id) tuples. At training time, replays buffer
nodes by including their full hyperedge neighborhoods in the loss computation.

Hypergraph-specific adaptation:
  - In standard GNN replay, replaying a node means including its 1-hop
    neighbors via the edge_index. In hypergraphs, replaying a node means
    including ALL hyperedges it belongs to and ALL members of those hyperedges.
  - This is a strictly more inclusive replay context than pairwise GNNs,
    since hyperedges encode higher-order group interactions.
  - The full graph structure is always present (no subgraph extraction needed
    since HGNN forward pass uses the full incidence matrix), but the replay
    loss is computed only on buffer node predictions.

Original GNN assumption that might break:
  - GNN replay assumes 1-hop neighborhood context is sufficient.
  - In hypergraphs, a node's representation depends on ALL its hyperedge
    members. Since we always pass the full incidence matrix, this is
    automatically handled — the HGNN convolution already sees the full
    hyperedge context.

Hypothesis:
  Random replay should provide a reasonable baseline mitigation for hypergraph
  forgetting, comparable to GNN replay performance, since it operates on
  the output/loss level rather than the structural level.

Buffer update: reservoir sampling to maintain a fixed-size buffer.
Loss: CE(current_task_nodes) + CE(buffer_nodes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional

from cl_methods.base import BaseCLMethod


class RandomReplay(BaseCLMethod):
    """Random Experience Replay for hypergraph continual learning.

    Stores a random subset of T1 training nodes in a buffer.
    During T2 training, replays buffer nodes alongside T2 nodes.
    Uses reservoir sampling for buffer updates.
    """

    def __init__(self, buffer_size: int = 100):
        super().__init__(name="Random Replay")
        self.buffer_size = buffer_size
        self.buffer_indices: Optional[Tensor] = None
        self.buffer_task_id: Optional[str] = None

    def before_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        """No preparation needed — buffer is built in after_task."""
        pass

    def after_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        """Build the replay buffer from the completed task."""
        self.update_buffer(task_id, data, model)

    def update_buffer(self, task_id: str, data: Data, model: nn.Module) -> None:
        """Fill buffer with randomly sampled training nodes from this task.

        Uses reservoir sampling: uniform random selection of buffer_size
        nodes from the task's training set.
        """
        import config
        task_labels = torch.tensor(config.TASK_SPLIT[task_id], device=data.y.device)
        train_mask = data.train_mask & torch.isin(data.y, task_labels)
        task_indices = train_mask.nonzero(as_tuple=True)[0].cpu()

        # Reservoir sampling
        n = min(self.buffer_size, len(task_indices))
        perm = torch.randperm(len(task_indices))[:n]
        self.buffer_indices = task_indices[perm]
        self.buffer_task_id = task_id

        print(f"        Buffer: {len(self.buffer_indices)} nodes "
              f"(requested {self.buffer_size})")

    def compute_loss(self, logits: Tensor, labels: Tensor, model: nn.Module,
                     data: Data, task_mask: Tensor,
                     class_weights: Tensor) -> Tensor:
        """CE on current task + CE on replay buffer nodes.

        Both losses use the same class weights for consistency.
        The full hyperedge context is automatically included since
        the HGNN forward pass uses the complete incidence matrix.
        """
        # Current task loss
        if task_mask.sum() == 0:
            loss_task = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            loss_task = F.cross_entropy(logits[task_mask], labels[task_mask],
                                        weight=class_weights)

        # Replay loss
        if self.buffer_indices is not None and len(self.buffer_indices) > 0:
            buf_idx = self.buffer_indices.to(logits.device)
            loss_replay = F.cross_entropy(logits[buf_idx], labels[buf_idx],
                                          weight=class_weights)
        else:
            loss_replay = torch.tensor(0.0, device=logits.device)

        return loss_task + loss_replay
