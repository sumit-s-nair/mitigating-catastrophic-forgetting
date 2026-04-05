"""
Topology-Aware Replay (TAR) — Adapted for Hypergraphs
======================================================
Selects replay buffer nodes based on hypergraph centrality rather than
standard graph degree centrality.

Hypergraph centrality: number of hyperedges a node participates in
(aka hyperedge degree). Selects top-k nodes by this measure as exemplars.

Hypergraph-specific adaptation:
  - In standard TAR on GNNs: hub nodes are selected by pairwise degree
  - For hypergraphs: "degree" = number of hyperedges containing the node
  - This is equivalent to the sum of column entries in the incidence matrix
  - Nodes in many hyperedges are assumed to be structurally important
    because they participate in many group interactions

Original GNN assumption that BREAKS:
  - Standard TAR assumes a skewed degree distribution where hub nodes are
    highly connected and structurally important.
  - In the Reddit bipartite graph, degree distribution is relatively flat
    because most users comment on a similar number of posts.
  - This caused TAR to fail on the standard GNN setup — it should similarly
    fail on the hypergraph version.

Hypothesis:
  TAR is EXPECTED TO PERFORM POORLY on this hypergraph because:
  1. User participation counts (hyperedge degrees) are relatively uniform
  2. Without clear hubs, "top-k by degree" is ~equivalent to random selection
  3. A warning is logged if degree variance is too low (std < 1.5)
  
  This is implemented faithfully to confirm the hypothesis experimentally.
"""

import warnings as _warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from typing import Optional

from cl_methods.base import BaseCLMethod


class TopologyAwareReplay(BaseCLMethod):
    """Topology-Aware Replay using hyperedge degree centrality.

    Selects buffer nodes by their hyperedge participation count —
    nodes that appear in the most hyperedges are considered most
    structurally important and prioritized for replay.

    Expected to fail on flat degree distributions.
    """

    def __init__(self, buffer_size: int = 100):
        super().__init__(name="TAR (Hypergraph)")
        self.buffer_size = buffer_size
        self.buffer_indices: Optional[Tensor] = None

    def before_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        pass

    def after_task(self, task_id: str, data: Data, model: nn.Module) -> None:
        """Select top-k T1 nodes by hyperedge degree."""
        self.update_buffer(task_id, data, model)

    def update_buffer(self, task_id: str, data: Data, model: nn.Module) -> None:
        """Build buffer using hyperedge degree centrality.

        Hyperedge degree = number of hyperedges each node belongs to.
        Top-k nodes by this measure are selected as exemplars.
        """
        import config

        he_index = data.hyperedge_index
        num_nodes = data.x.shape[0]

        # Compute hyperedge degree for each node
        node_idx = he_index[0]
        he_degrees = torch.zeros(num_nodes, dtype=torch.float)
        if he_index.shape[1] > 0:
            he_degrees.scatter_add_(
                0, node_idx.cpu(),
                torch.ones(he_index.shape[1], dtype=torch.float)
            )

        # Check degree variance — warn if too flat
        task_labels = torch.tensor(config.TASK_SPLIT[task_id], device=data.y.device)
        train_mask = data.train_mask & torch.isin(data.y, task_labels)
        task_indices = train_mask.nonzero(as_tuple=True)[0].cpu()

        if len(task_indices) == 0:
            self.buffer_indices = torch.tensor([], dtype=torch.long)
            return

        task_degrees = he_degrees[task_indices]
        degree_std = float(task_degrees.std())

        if degree_std < 1.5:
            _warnings.warn(
                f"[TAR WARNING] Hyperedge degree std = {degree_std:.2f} < 1.5. "
                f"Flat centrality distribution — TAR is unlikely to perform "
                f"better than random replay. Mean degree: {task_degrees.mean():.2f}, "
                f"Range: [{task_degrees.min():.0f}, {task_degrees.max():.0f}]"
            )

        # Select top-k by hyperedge degree
        n = min(self.buffer_size, len(task_indices))
        _, top_idx = task_degrees.topk(n, largest=True)
        self.buffer_indices = task_indices[top_idx]

        print(f"        TAR Buffer: {len(self.buffer_indices)} nodes "
              f"(degree std={degree_std:.2f}, "
              f"range=[{task_degrees.min():.0f}-{task_degrees.max():.0f}])")

    def compute_loss(self, logits: Tensor, labels: Tensor, model: nn.Module,
                     data: Data, task_mask: Tensor,
                     class_weights: Tensor) -> Tensor:
        """CE on current task + CE on TAR buffer nodes."""
        device = logits.device

        # Current task loss
        if task_mask.sum() == 0:
            loss_task = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss_task = F.cross_entropy(logits[task_mask], labels[task_mask],
                                        weight=class_weights)

        # Replay loss on TAR buffer
        if self.buffer_indices is not None and len(self.buffer_indices) > 0:
            buf_idx = self.buffer_indices.to(device)
            loss_replay = F.cross_entropy(logits[buf_idx], labels[buf_idx],
                                          weight=class_weights)
        else:
            loss_replay = torch.tensor(0.0, device=device)

        return loss_task + loss_replay
