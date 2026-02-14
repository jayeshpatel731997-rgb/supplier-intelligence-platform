"""
Graph Centrality & Network Analysis for Supply Chain Risk
==========================================================

Computes graph-theoretic metrics to identify critical "nexus" nodes
in multi-tier supply networks. High-centrality nodes represent
systemic risk — their failure cascades widely.

Metrics computed:
1. Degree Centrality — number of direct connections
2. Betweenness Centrality — controls flow between other nodes
3. Eigenvector Centrality — connected to other well-connected nodes
4. Composite Criticality Score — weighted combination

References:
- Brintrup et al. 2021: GNN for supply chain link prediction
- Xie et al. 2023: Centrality-weighted risk scores
- Acemoglu et al. 2012: Network origins of aggregate fluctuations
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Optional
@dataclass
class NodeCentrality:
    """Centrality metrics for a single node."""
    node_id: str
    name: str
    tier: int
    degree_centrality: float
    in_degree_centrality: float
    out_degree_centrality: float
    betweenness_centrality: float
    eigenvector_centrality: float
    closeness_centrality: float
    pagerank: float
    criticality_score: float
    risk_amplification_factor: float
def compute_centrality_metrics(
    G: nx.DiGraph,
    weights: Optional[dict] = None,
) -> list[NodeCentrality]:
    """
    Compute comprehensive centrality metrics for all nodes.

    Args:
        G: NetworkX directed graph
        weights: Optional dict with keys 'degree', 'betweenness',
                 'eigenvector', 'pagerank' summing to 1.0

    Returns:
        List of NodeCentrality objects, sorted by criticality_score desc
    """
    if weights is None:
        weights = {
            "degree": 0.15,
            "betweenness": 0.35,
            "eigenvector": 0.20,
            "pagerank": 0.30,
        }

    n = G.number_of_nodes()
    if n == 0:
        return []

    # ─── COMPUTE RAW METRICS ─────────────────────────────────────

    # Degree centrality (normalized)
    degree = nx.degree_centrality(G)
    in_degree = nx.in_degree_centrality(G)
    out_degree = nx.out_degree_centrality(G)

    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

    # Eigenvector centrality (on undirected version for convergence)
    try:
        G_undirected = G.to_undirected()
        eigenvector = nx.eigenvector_centrality(
            G_undirected, max_iter=1000, weight="weight"
        )
    except nx.PowerIterationFailedConvergence:
        eigenvector = {node: 1.0 / n for node in G.nodes()}

    # Closeness centrality
    closeness = nx.closeness_centrality(G)

    # PageRank (supply flows as "importance" propagation)
    pagerank = nx.pagerank(G, weight="weight", alpha=0.85)

    # ─── NORMALIZE TO [0, 1] ─────────────────────────────────────

    def normalize(d: dict) -> dict:
        values = list(d.values())
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return {k: 0.5 for k in d}
        return {k: (v - min_v) / (max_v - min_v) for k, v in d.items()}

    norm_degree = normalize(degree)
    norm_between = normalize(betweenness)
    norm_eigen = normalize(eigenvector)
    norm_pagerank = normalize(pagerank)

    # ─── COMPOSITE CRITICALITY SCORE ─────────────────────────────

    results = []
    for node_id in G.nodes():
        node_data = G.nodes[node_id]

        criticality = (
            weights["degree"] * norm_degree[node_id] +
            weights["betweenness"] * norm_between[node_id] +
            weights["eigenvector"] * norm_eigen[node_id] +
            weights["pagerank"] * norm_pagerank[node_id]
        )

        # Risk amplification: how much does this node's failure
        # amplify disruption to the focal firm?
        # Based on shortest path to OEM and edge weights
        tier = node_data.get("tier", 0)
        risk_amp = _compute_amplification(G, node_id, tier)

        results.append(NodeCentrality(
            node_id=node_id,
            name=node_data.get("name", node_id),
            tier=node_data.get("tier", 0),
            degree_centrality=degree[node_id],
            in_degree_centrality=in_degree[node_id],
            out_degree_centrality=out_degree[node_id],
            betweenness_centrality=betweenness[node_id],
            eigenvector_centrality=eigenvector[node_id],
            closeness_centrality=closeness[node_id],
            pagerank=pagerank[node_id],
            criticality_score=criticality,
            risk_amplification_factor=risk_amp,
        ))

    return sorted(results, key=lambda x: x.criticality_score, reverse=True)
def _compute_amplification(G: nx.DiGraph, node_id: str, tier: int) -> float:
    """
    Compute how much a node's disruption amplifies toward the focal firm.

    Higher amplification = node is on critical paths to OEM.
    """
    # Find focal firm (tier 0 or node named "OEM")
    focal_nodes = [
        n for n in G.nodes()
        if G.nodes[n].get("tier", -1) == 0 or G.nodes[n].get("type") == "focal"
    ]

    if not focal_nodes or node_id in focal_nodes:
        return 1.0

    # Count paths from this node to focal firm
    max_amp = 0.0
    for focal in focal_nodes:
        try:
            paths = list(nx.all_simple_paths(G, node_id, focal, cutoff=5))
            if paths:
                # More paths = higher amplification
                path_factor = min(len(paths) / 3.0, 2.0)
                # Shorter paths = higher amplification
                shortest = min(len(p) for p in paths)
                distance_factor = 1.0 / shortest
                max_amp = max(max_amp, path_factor * distance_factor)
        except nx.NetworkXError:
            pass

    # If no direct paths, try reversed graph (supply flows upstream)
    if max_amp == 0:
        G_rev = G.reverse()
        for focal in focal_nodes:
            try:
                paths = list(nx.all_simple_paths(G_rev, focal, node_id, cutoff=5))
                if paths:
                    path_factor = min(len(paths) / 3.0, 2.0)
                    shortest = min(len(p) for p in paths)
                    max_amp = max(max_amp, path_factor * (1.0 / shortest))
            except nx.NetworkXError:
                pass

    return max(max_amp, 0.1)  # Minimum 0.1
# ─── VULNERABILITY ANALYSIS ─────────────────────────────────────

def identify_single_points_of_failure(G: nx.DiGraph) -> list[dict]:
    """
    Find nodes whose removal disconnects supply to the focal firm.
    These are the most critical vulnerabilities.
    """
    focal_nodes = [
        n for n in G.nodes()
        if G.nodes[n].get("tier", -1) == 0 or G.nodes[n].get("type") == "focal"
    ]

    if not focal_nodes:
        return []

    spofs = []
    G_undirected = G.to_undirected()

    # Check articulation points (nodes whose removal disconnects the graph)
    try:
        artic_points = list(nx.articulation_points(G_undirected))
    except:
        artic_points = []

    for node_id in artic_points:
        if node_id not in focal_nodes:
            node_data = G.nodes[node_id]
            # How many nodes become disconnected?
            G_removed = G_undirected.copy()
            G_removed.remove_node(node_id)
            components = list(nx.connected_components(G_removed))

            disconnected = 0
            for focal in focal_nodes:
                focal_component = next(
                    (c for c in components if focal in c), set()
                )
                disconnected = sum(
                    1 for c in components
                    if c != focal_component
                    for n in c if G.nodes[n].get("type") == "supplier"
                )

            spofs.append({
                "node_id": node_id,
                "name": node_data.get("name", node_id),
                "tier": node_data.get("tier", 0),
                "suppliers_disconnected": disconnected,
                "severity": "CRITICAL" if disconnected > 2 else "HIGH",
            })

    return sorted(spofs, key=lambda x: x["suppliers_disconnected"], reverse=True)
def compute_network_resilience_score(G: nx.DiGraph) -> dict:
    """
    Compute overall network resilience metrics.
    Higher score = more resilient supply network.
    """
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return {"resilience_score": 0}

    # Density: higher = more redundant paths
    density = nx.density(G)

    # Average clustering (undirected)
    try:
        avg_clustering = nx.average_clustering(G.to_undirected())
    except:
        avg_clustering = 0

    # Number of articulation points (fewer = more resilient)
    try:
        n_artic = len(list(nx.articulation_points(G.to_undirected())))
    except:
        n_artic = 0

    artic_ratio = 1 - (n_artic / max(n_nodes, 1))

    # Edge connectivity (minimum edges to remove to disconnect)
    try:
        edge_conn = nx.edge_connectivity(G.to_undirected())
    except:
        edge_conn = 0

    # Composite resilience score (0-100)
    resilience = (
        density * 25 +
        avg_clustering * 25 +
        artic_ratio * 25 +
        min(edge_conn / 3, 1) * 25
    )

    return {
        "resilience_score": resilience,
        "density": density,
        "avg_clustering": avg_clustering,
        "articulation_points": n_artic,
        "edge_connectivity": edge_conn,
        "total_nodes": n_nodes,
        "total_edges": n_edges,
        "interpretation": (
            "HIGH" if resilience > 60 else
            "MEDIUM" if resilience > 35 else
            "LOW"
        ),
    }
