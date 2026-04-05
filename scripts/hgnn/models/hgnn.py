"""
HGNN Model Backbones — Hypergraph Neural Networks
===================================================
Two HGNN architectures selectable via config.MODEL_TYPE:

Option A: HGNN (Feng et al. 2019)
  Uses the hypergraph Laplacian:
    Δ = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
  Two-layer HGNN with ReLU activation and dropout.
  This is the spectral approach — message passing through the Laplacian.

Option B: AllDeepSets
  Set-based message passing over hyperedges:
    1. Each hyperedge aggregates member node features with a φ network (MLP)
    2. Each node aggregates signals from its hyperedges with a ρ network (MLP)
  This is the spatial approach — explicit set aggregation.

Both have identical interface for drop-in replacement:
  forward(x, hyperedge_index, hyperedge_weight) -> logits
  get_embeddings(x, hyperedge_index) -> hidden representations

Hypergraph-specific notes:
  - Unlike standard GNNs (e.g., GAT) which pass messages along pairwise edges,
    HGNNs pass messages through higher-order hyperedges.
  - The hypergraph Laplacian captures multi-way relationships that are lost
    in clique-expansion approaches.
  - For continual learning: the key question is whether preserving higher-order
    structural information helps prevent forgetting compared to pairwise GNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HGNNConv(nn.Module):
    """Single Hypergraph Convolutional Layer (Feng et al. 2019).

    Implements: Y = σ(D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X Θ)

    Where:
      H: incidence matrix [N, M]
      W: diagonal hyperedge weight matrix [M, M]
      D_v: node degree matrix (D_v[i,i] = Σ_e H[i,e] * W[e])
      D_e: hyperedge degree matrix (D_e[e,e] = Σ_i H[i,e])
      Θ: learnable weight matrix [in_ch, out_ch]

    We compute this efficiently using sparse scatter operations on the COO
    hyperedge_index without materializing the full incidence matrix.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Tensor = None) -> Tensor:
        """
        Args:
            x: [N, in_ch] node features
            hyperedge_index: [2, num_incidences] COO (node_idx, hyperedge_idx)
            hyperedge_weight: [M] optional weights (default: ones)

        Returns:
            [N, out_ch] transformed node features
        """
        num_nodes = x.size(0)
        node_idx = hyperedge_index[0]  # [num_incidences]
        he_idx = hyperedge_index[1]    # [num_incidences]

        if hyperedge_index.shape[1] == 0:
            # No incidences, just apply linear
            return self.linear(x)

        num_he = int(he_idx.max()) + 1

        # Default weights
        if hyperedge_weight is None:
            hyperedge_weight = torch.ones(num_he, device=x.device, dtype=x.dtype)

        # D_e = hyperedge degrees (number of nodes in each hyperedge)
        d_e = torch.zeros(num_he, device=x.device, dtype=x.dtype)
        d_e.scatter_add_(0, he_idx, torch.ones_like(he_idx, dtype=x.dtype))
        d_e = d_e.clamp(min=1)  # avoid div by zero

        # D_v = node degrees (sum of weights of hyperedges each node belongs to)
        w_per_incidence = hyperedge_weight[he_idx]  # weight for each incidence
        d_v = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        d_v.scatter_add_(0, node_idx, w_per_incidence)
        d_v = d_v.clamp(min=1)

        # D_v^{-1/2}
        d_v_inv_sqrt = d_v.pow(-0.5)

        # Step 1: X' = D_v^{-1/2} X
        x_prime = x * d_v_inv_sqrt.unsqueeze(1)

        # Step 2: Aggregate node features to hyperedges: H^T X'
        # For each hyperedge, sum the features of its member nodes
        he_feat = torch.zeros(num_he, x.size(1), device=x.device, dtype=x.dtype)
        he_feat.scatter_add_(0, he_idx.unsqueeze(1).expand(-1, x.size(1)),
                             x_prime[node_idx])

        # Step 3: W D_e^{-1} * hyperedge features
        he_feat = he_feat * (hyperedge_weight / d_e).unsqueeze(1)

        # Step 4: Distribute back to nodes: H * he_feat
        out = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, node_idx.unsqueeze(1).expand(-1, x.size(1)),
                         he_feat[he_idx])

        # Step 5: D_v^{-1/2} * out
        out = out * d_v_inv_sqrt.unsqueeze(1)

        # Step 6: Linear transformation
        out = self.linear(out)
        return out


class HGNNBackbone(nn.Module):
    """Two-layer HGNN backbone (Feng et al. 2019).

    Architecture: x → HGNNConv1 → ReLU → Dropout → HGNNConv2 → logits

    For CL methods: get_embeddings() returns the intermediate representation
    after the first layer, useful for LwF distillation and feature analysis.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = HGNNConv(in_channels, hidden_channels)
        self.conv2 = HGNNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Tensor = None) -> Tensor:
        """Full forward pass returning logits."""
        x = self.conv1(x, hyperedge_index, hyperedge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, hyperedge_index, hyperedge_weight)
        return x

    def get_embeddings(self, x: Tensor, hyperedge_index: Tensor,
                       hyperedge_weight: Tensor = None) -> Tensor:
        """Return intermediate embeddings (after first conv + ReLU).
        Used for LwF distillation and representation analysis."""
        x = self.conv1(x, hyperedge_index, hyperedge_weight)
        x = F.relu(x)
        return x


class DeepSetsConv(nn.Module):
    """Single DeepSets-style hypergraph convolution layer.

    Message passing in two phases:
      1. φ (node → hyperedge): aggregate member node features into each hyperedge
         using mean pooling followed by an MLP.
      2. ρ (hyperedge → node): aggregate hyperedge signals back to each node
         using sum followed by an MLP.

    This is a spatial approach that explicitly models set operations,
    unlike the spectral HGNN approach.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # φ: processes aggregated node features per hyperedge
        self.phi = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        # ρ: processes aggregated hyperedge signals per node
        self.rho = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Tensor = None) -> Tensor:
        num_nodes = x.size(0)
        node_idx = hyperedge_index[0]
        he_idx = hyperedge_index[1]

        if hyperedge_index.shape[1] == 0:
            return self.rho(self.phi(x))

        num_he = int(he_idx.max()) + 1

        # Phase 1: φ — aggregate nodes into hyperedges (mean pool)
        he_sum = torch.zeros(num_he, x.size(1), device=x.device, dtype=x.dtype)
        he_sum.scatter_add_(0, he_idx.unsqueeze(1).expand(-1, x.size(1)),
                            x[node_idx])
        he_count = torch.zeros(num_he, 1, device=x.device, dtype=x.dtype)
        he_count.scatter_add_(0, he_idx.unsqueeze(1),
                              torch.ones(len(he_idx), 1, device=x.device, dtype=x.dtype))
        he_count = he_count.clamp(min=1)
        he_mean = he_sum / he_count
        he_feat = self.phi(he_mean)

        # Apply hyperedge weights if provided
        if hyperedge_weight is not None:
            he_feat = he_feat * hyperedge_weight.unsqueeze(1)

        # Phase 2: ρ — aggregate hyperedge signals back to nodes (sum)
        node_signal = torch.zeros(num_nodes, he_feat.size(1), device=x.device, dtype=x.dtype)
        node_signal.scatter_add_(0, node_idx.unsqueeze(1).expand(-1, he_feat.size(1)),
                                 he_feat[he_idx])
        out = self.rho(node_signal)
        return out


class AllDeepSetsBackbone(nn.Module):
    """Two-layer AllDeepSets backbone for hypergraph learning.

    Architecture: x → DeepSetsConv1 → ReLU → Dropout → Linear → logits

    Uses set-based message passing where each hyperedge independently
    aggregates its member features, then distributes signals back.

    For CL: the hypothesis is that explicit set-based aggregation may
    preserve class-discriminative features differently from the spectral
    HGNN approach, potentially affecting forgetting patterns.
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = DeepSetsConv(in_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Tensor = None) -> Tensor:
        x = self.conv1(x, hyperedge_index, hyperedge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

    def get_embeddings(self, x: Tensor, hyperedge_index: Tensor,
                       hyperedge_weight: Tensor = None) -> Tensor:
        x = self.conv1(x, hyperedge_index, hyperedge_weight)
        x = F.relu(x)
        return x


def create_model(model_type: str, in_channels: int, hidden_channels: int,
                 out_channels: int, dropout: float = 0.5) -> nn.Module:
    """Factory function to create the appropriate HGNN backbone.

    Args:
        model_type: "HGNN" or "AllDeepSets"
    """
    if model_type == "HGNN":
        return HGNNBackbone(in_channels, hidden_channels, out_channels, dropout)
    elif model_type == "AllDeepSets":
        return AllDeepSetsBackbone(in_channels, hidden_channels, out_channels, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'HGNN' or 'AllDeepSets'.")
