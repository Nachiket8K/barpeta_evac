"""
OSMnx / road graph utilities as used in the notebook:
- download graph from bbox
- filter by highway classes
- largest connected component
- save nodes/edges to a GeoPackage (as in notebook)
"""

from __future__ import annotations

from pathlib import Path
from typing import Set, Tuple

from .config import ox, nx, gpd, pd


def download_graph_from_bbox(
    bbox: Tuple[float, float, float, float],
    *,
    network_type: str = "drive",
    simplify: bool = True,
    retain_all: bool = True,
    timeout: int = 180,
):
    """
    bbox is (north, south, east, west) as used by osmnx.graph_from_bbox.
    """
    ox.settings.timeout = timeout
    return ox.graph_from_bbox(
        bbox=bbox,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
    )


def graph_to_gdfs(G):
    """Convert to (nodes_gdf, edges_gdf)."""
    return ox.graph_to_gdfs(G, nodes=True, edges=True)


def _edge_has_highway(data: dict, keep: Set[str]) -> bool:
    hw = data.get("highway")
    if hw is None:
        return False
    if isinstance(hw, (list, tuple, set)):
        return any(h in keep for h in hw)
    return hw in keep


def filter_graph_by_highway(G, keep_highways: Set[str]):
    """Edge-induced subgraph retaining only edges with highway in keep_highways."""
    edges_keep = [
        (u, v, k)
        for u, v, k, data in G.edges(keys=True, data=True)
        if _edge_has_highway(data, keep_highways)
    ]
    return G.edge_subgraph(edges_keep).copy()


def largest_connected_component(G):
    """Largest connected component (undirected connectivity)."""
    Gu = G.to_undirected()
    lcc_nodes = max(nx.connected_components(Gu), key=len)
    return G.subgraph(lcc_nodes).copy()


def save_graph_gpkg(G, out_path: Path, *, nodes_layer: str = "nodes", edges_layer: str = "edges") -> Path:
    """
    Save graph nodes and edges to a GeoPackage (matches your notebook cell).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nodes, edges = graph_to_gdfs(G)
    edges.to_file(out_path, layer=edges_layer, driver="GPKG")
    nodes.to_file(out_path, layer=nodes_layer, driver="GPKG")
    return out_path


def save_graphml(G, out_path: Path) -> Path:
    """Save graph to GraphML."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, filepath=str(out_path))
    return out_path
