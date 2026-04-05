"""
Topology Audit — Pre-run Hypergraph Analysis
==============================================
Analyzes the hypergraph topology to predict which CL methods are likely
to succeed or fail. This is a key research contribution: understanding
WHY certain methods work or fail on specific graph structures.

Metrics computed:
  - Node degree statistics (hyperedge participation count)
  - Hyperedge size statistics
  - Label-neighbor density (homophily through hyperedges)
  - Bipartite score (how close to bipartite the incidence is)

Predictions:
  - Flat degree distribution → TAR likely to fail
  - High label density → Replay methods should maintain diversity
  - Near-bipartite → Topology-agnostic methods preferred
  - HypergraphEWC always worth trying (novel baseline)
"""

import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import Dict, Tuple


def compute_hypergraph_metrics(data: Data) -> Dict[str, float]:
    """Compute structural metrics for the hypergraph.

    Args:
        data: Data object with hyperedge_index, y, and features

    Returns:
        dict with all topology metrics
    """
    he_index = data.hyperedge_index
    num_nodes = data.x.shape[0]

    if he_index.shape[1] == 0:
        return {
            "node_degree_mean": 0, "node_degree_std": 0,
            "hyperedge_size_mean": 0, "hyperedge_size_std": 0,
            "label_neighbor_density": 0, "bipartite_score": 0,
        }

    node_idx = he_index[0]
    he_idx = he_index[1]
    num_he = int(he_idx.max()) + 1

    # ─── Node degree (hyperedge participation count) ──────────────
    node_degrees = torch.zeros(num_nodes, dtype=torch.float)
    node_degrees.scatter_add_(0, node_idx, torch.ones(len(node_idx), dtype=torch.float))

    # ─── Hyperedge size ──────────────────────────────────────────
    he_sizes = torch.zeros(num_he, dtype=torch.float)
    he_sizes.scatter_add_(0, he_idx, torch.ones(len(he_idx), dtype=torch.float))

    # ─── Label-neighbor density ──────────────────────────────────
    # Per-type and overall: fraction of same-label pairs within hyperedges.
    # Thread HEs are trivially same-subreddit; user HEs are the interesting
    # case (a user commenting in multiple subreddits would create mixed labels).
    labels = data.y

    def _compute_density_for_range(start_he, end_he):
        """Compute label-neighbor density for a range of hyperedge IDs."""
        same, total = 0, 0
        sample_ids = list(range(start_he, end_he))
        if len(sample_ids) > 500:
            import random
            random.shuffle(sample_ids)
            sample_ids = sample_ids[:500]
        for he_i in sample_ids:
            members = node_idx[he_idx == he_i]
            if len(members) < 2:
                continue
            member_labels = labels[members]
            valid = member_labels >= 0
            member_labels = member_labels[valid]
            if len(member_labels) < 2:
                continue
            for i in range(len(member_labels)):
                for j in range(i + 1, len(member_labels)):
                    total += 1
                    if member_labels[i] == member_labels[j]:
                        same += 1
        return same / max(total, 1), total

    # Compute per-type density if type offsets are available
    # (passed via data attributes or computed from hyperedge IDs)
    overall_density, overall_pairs = _compute_density_for_range(0, num_he)
    label_neighbor_density = overall_density

    # ─── Bipartite score ─────────────────────────────────────────
    # How close the incidence structure is to bipartite:
    # If most nodes belong to exactly 1 hyperedge, the structure is star-like.
    # Bipartite score = fraction of nodes with degree 1.
    degree_1_frac = (node_degrees == 1).float().mean().item()
    # Also consider: if most hyperedges have size 2, it's nearly a graph
    size_2_frac = (he_sizes == 2).float().mean().item()
    bipartite_score = 0.5 * degree_1_frac + 0.5 * size_2_frac

    metrics = {
        "node_degree_mean": float(node_degrees.mean()),
        "node_degree_std": float(node_degrees.std()),
        "node_degree_min": int(node_degrees.min()),
        "node_degree_max": int(node_degrees.max()),
        "hyperedge_size_mean": float(he_sizes.mean()),
        "hyperedge_size_std": float(he_sizes.std()),
        "hyperedge_size_min": int(he_sizes.min()),
        "hyperedge_size_max": int(he_sizes.max()),
        "label_neighbor_density": label_neighbor_density,
        "bipartite_score": bipartite_score,
        "num_nodes": num_nodes,
        "num_hyperedges": num_he,
        "num_incidences": int(he_index.shape[1]),
    }

    return metrics


def predict_method_suitability(metrics: Dict[str, float]) -> list:
    """Predict which CL methods are likely to work based on topology metrics.

    Returns a ranked list of (method, prediction, reasoning) tuples.

    This is the research contribution: connecting structural properties
    to expected CL method performance before running experiments.
    """
    predictions = []

    degree_std = metrics.get("node_degree_std", 0)
    label_density = metrics.get("label_neighbor_density", 0)
    bipartite_score = metrics.get("bipartite_score", 0)

    # ─── TAR Assessment ──────────────────────────────────────────
    if degree_std < 1.5:
        predictions.append((
            "TAR (Hypergraph)",
            "LIKELY TO FAIL",
            f"Flat centrality distribution (degree std={degree_std:.2f} < 1.5). "
            f"Cannot meaningfully distinguish hub nodes from periphery. "
            f"This mirrors the bipartite Reddit graph result where TAR failed "
            f"because degree distribution was too uniform."
        ))
    else:
        predictions.append((
            "TAR (Hypergraph)",
            "MAY SUCCEED",
            f"Degree variance is sufficient (std={degree_std:.2f}) for "
            f"meaningful hub selection."
        ))

    # ─── Replay Assessment ────────────────────────────────────────
    if label_density > 0.4:
        predictions.append((
            "Random Replay",
            "LIKELY TO MAINTAIN CLASS DIVERSITY",
            f"High label-neighbor density ({label_density:.2f} > 0.4) — "
            f"nodes sharing hyperedges tend to share labels, so replaying "
            f"nodes preserves community structure."
        ))
    else:
        predictions.append((
            "Random Replay",
            "CLASS DIVERSITY AT RISK",
            f"Low label-neighbor density ({label_density:.2f}) — "
            f"hyperedges mix labels, random replay may not preserve "
            f"class coherence."
        ))

    # ─── Topology-agnostic Assessment ────────────────────────────
    if bipartite_score > 0.8:
        predictions.append((
            "LwF / Standard EWC",
            "PREFERRED",
            f"Near-bipartite structure (score={bipartite_score:.2f} > 0.8). "
            f"Topology provides little additional signal for CL; "
            f"topology-agnostic methods (LwF, EWC) avoid structural noise."
        ))
    else:
        predictions.append((
            "LwF / Standard EWC",
            "REASONABLE BASELINE",
            f"Non-trivial hypergraph structure (bipartite score={bipartite_score:.2f}). "
            f"Topology-agnostic methods serve as baseline; "
            f"structure-aware methods may outperform."
        ))

    # ─── HypergraphEWC — always worth trying ────────────────────
    predictions.append((
        "HypergraphEWC",
        "WORTH TRYING (NOVEL)",
        f"Novel contribution: structure-aware Fisher weighting by hyperedge "
        f"participation. Even with flat degree distribution, the size-normalized "
        f"weighting (|e|^{{-1}}) may capture important structural importance "
        f"signals that standard EWC misses."
    ))

    return predictions


def run_topology_audit(data: Data, info: dict = None) -> Tuple[dict, list]:
    """Run the full topology audit and print results.

    Args:
        data: hypergraph Data object
        info: optional dict from build_hypergraph_data with he_type_offsets

    Returns:
        metrics: dict of computed topology metrics
        predictions: list of method suitability predictions
    """
    metrics = compute_hypergraph_metrics(data)

    # Compute per-type density if type info is available
    if info and "he_type_offsets" in info:
        he_index = data.hyperedge_index
        node_idx = he_index[0]
        he_idx = he_index[1]
        labels = data.y

        def _density_for_range(start, end):
            same, total = 0, 0
            sample_ids = list(range(start, end))
            if len(sample_ids) > 500:
                import random
                random.shuffle(sample_ids)
                sample_ids = sample_ids[:500]
            for he_i in sample_ids:
                members = node_idx[he_idx == he_i]
                if len(members) < 2:
                    continue
                ml = labels[members]
                ml = ml[ml >= 0]
                if len(ml) < 2:
                    continue
                for i in range(len(ml)):
                    for j in range(i + 1, len(ml)):
                        total += 1
                        if ml[i] == ml[j]:
                            same += 1
            return same / max(total, 1)

        per_type_density = {}
        for htype, (start, end) in info["he_type_offsets"].items():
            if end > start:
                per_type_density[htype] = _density_for_range(start, end)
        metrics["per_type_density"] = per_type_density

    predictions = predict_method_suitability(metrics)

    print(f"\n{'='*65}")
    print("HYPERGRAPH TOPOLOGY AUDIT")
    print(f"{'='*65}")
    print(f"\n  Structure:")
    print(f"    Nodes:        {metrics['num_nodes']}")
    print(f"    Hyperedges:   {metrics['num_hyperedges']}")
    print(f"    Incidences:   {metrics['num_incidences']}")
    print(f"\n  Node Degree (hyperedge participation):")
    print(f"    Mean: {metrics['node_degree_mean']:.2f}  Std: {metrics['node_degree_std']:.2f}")
    print(f"    Range: [{metrics.get('node_degree_min', 0)}, {metrics.get('node_degree_max', 0)}]")
    print(f"\n  Hyperedge Size:")
    print(f"    Mean: {metrics['hyperedge_size_mean']:.2f}  Std: {metrics['hyperedge_size_std']:.2f}")
    print(f"    Range: [{metrics.get('hyperedge_size_min', 0)}, {metrics.get('hyperedge_size_max', 0)}]")
    print(f"\n  Label-Neighbor Density (overall): {metrics['label_neighbor_density']:.4f}")
    if "per_type_density" in metrics:
        for htype, density in metrics["per_type_density"].items():
            print(f"    {htype:>10}: {density:.4f}")
    print(f"  Bipartite Score:       {metrics['bipartite_score']:.4f}")

    print(f"\n{'─'*65}")
    print("METHOD SUITABILITY PREDICTIONS")
    print(f"{'─'*65}")
    for method, verdict, reason in predictions:
        print(f"\n  {method}: {verdict}")
        print(f"    {reason}")
    print(f"\n{'='*65}\n")

    return metrics, predictions

