from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def load_graph(nodes_path: Path, edges_path: Path) -> tuple[nx.Graph, pd.DataFrame]:
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    required_node_cols = {"node_id", "node_type", "node_key"}
    required_edge_cols = {"src", "dst", "relation"}

    missing_node_cols = required_node_cols - set(nodes_df.columns)
    missing_edge_cols = required_edge_cols - set(edges_df.columns)
    if missing_node_cols:
        raise ValueError(f"Missing node columns: {sorted(missing_node_cols)}")
    if missing_edge_cols:
        raise ValueError(f"Missing edge columns: {sorted(missing_edge_cols)}")

    graph = nx.Graph()
    for row in nodes_df.itertuples(index=False):
        graph.add_node(
            int(row.node_id),
            node_type=str(row.node_type),
            node_key=str(row.node_key),
            label=str(getattr(row, "text", "") or ""),
        )

    for row in edges_df.itertuples(index=False):
        src = int(row.src)
        dst = int(row.dst)
        if src in graph and dst in graph:
            graph.add_edge(src, dst, relation=str(row.relation))

    return graph, nodes_df


def sample_graph_by_degree(graph: nx.Graph, max_nodes: int) -> nx.Graph:
    if graph.number_of_nodes() <= max_nodes:
        return graph

    degrees = sorted(graph.degree, key=lambda item: item[1], reverse=True)
    keep_nodes = {node_id for node_id, _ in degrees[:max_nodes]}
    return graph.subgraph(keep_nodes).copy()


def draw_graph(graph: nx.Graph, out_file: Path, original_nodes: int, sampled: bool) -> None:
    color_by_type = {
        "user": "#1f77b4",
        "post": "#ff7f0e",
        "comment": "#2ca02c",
    }

    node_types = nx.get_node_attributes(graph, "node_type")
    node_colors = [color_by_type.get(node_types.get(node, ""), "#7f7f7f") for node in graph.nodes]

    node_sizes = []
    for node in graph.nodes:
        node_type = node_types.get(node, "")
        if node_type == "post":
            node_sizes.append(50)
        elif node_type == "user":
            node_sizes.append(35)
        else:
            node_sizes.append(22)

    plt.figure(figsize=(20, 14), dpi=120)
    pos = nx.spring_layout(graph, seed=42, k=0.25, iterations=120)

    nx.draw_networkx_edges(graph, pos, alpha=0.10, width=0.5, edge_color="#999999")
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.88,
        linewidths=0.2,
        edgecolors="#ffffff",
    )

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=color, label=label)
        for label, color in [("user", "#1f77b4"), ("post", "#ff7f0e"), ("comment", "#2ca02c")]
    ]
    plt.legend(handles=handles, loc="upper right", frameon=True, title="Node Type")

    sample_note = "sampled" if sampled else "full"
    plt.title(
        (
            "Reddit Graph Visualization "
            f"({sample_note}, rendered_nodes={graph.number_of_nodes()}, total_nodes={original_nodes}, "
            f"edges={graph.number_of_edges()})"
        ),
        fontsize=14,
    )
    plt.axis("off")
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize users, posts, and comments graph from generated node/edge CSVs."
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path("output/combined/python_gnn_nodes.csv"),
        help="Path to nodes CSV file.",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("output/combined/python_gnn_edges.csv"),
        help="Path to edges CSV file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/combined/python_gnn_visualization.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=4000,
        help="Maximum number of nodes to render (high-degree sampling if exceeded). Use 0 for full graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    graph, _ = load_graph(args.nodes, args.edges)
    original_nodes = graph.number_of_nodes()
    original_edges = graph.number_of_edges()

    if args.max_nodes <= 0:
        sampled = False
    else:
        graph = sample_graph_by_degree(graph, max_nodes=max(100, args.max_nodes))
        sampled = graph.number_of_nodes() < original_nodes

    draw_graph(graph, args.out, original_nodes=original_nodes, sampled=sampled)

    print(f"Loaded graph with {original_nodes} nodes and {original_edges} edges.")
    print(f"Rendered graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    if sampled:
        print("Rendering mode: sampled")
    else:
        print("Rendering mode: full")
    print(f"Saved visualization: {args.out}")


if __name__ == "__main__":
    main()
