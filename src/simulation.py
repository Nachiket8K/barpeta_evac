# src/simulation.py
"""
Evacuation simulation primitives (v0)

This module implements the simulation components utilized in
`04_simulation_outline.ipynb`. It is scoped to:

- road damage realization based on per-edge hazard features (by month)
- truck route computation from a source node to exit nodes under damage
- people placement into buildings (uniform or weighted)
- accessibility scoring: distance from buildings to nearest evac-path geometry
- Monte Carlo wrapper producing per-run KPI rows

It does NOT yet implement:
- dynamic congestion / time-stepping / loading & capacity constraints
- pickup / dropoff logistics
- agent-to-road assignment dynamics
- calibration against observed flood closures

These can be incorporated later but are not going to be done at this stage. 
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import math

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import unary_union


EdgeKey = Tuple[Any, Any, Any]  # (u, v, key) for MultiDiGraph


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FailureModelParams:
    """
    Parameters controlling the edge failure probability model.

    The recommended initial rule set:
    - If p_hazard (centerline wet fraction) >= water_over_threshold -> fail with p=1
    - Else if p_buffer_hazard available -> p_fail = clip(p_buffer_hazard, 0, 1)
    - Else -> p_background (unknown risk)
    """
    water_over_threshold: float = 0.95
    p_background: float = 0.01


@dataclass(frozen=True)
class SimulationParams:
    """
    High-level simulation knobs.
    """
    month: str = "jan"
    n_trucks: int = 10
    n_people: int = 50_000
    access_radius_m: float = 500.0
    seed: int = 42

    # Routing
    route_weight: str = "length"  # edge attribute name used for shortest path

    # Failure model
    failure: FailureModelParams = FailureModelParams()


# -----------------------------------------------------------------------------
# Utilities: attaching hazard features to a graph
# -----------------------------------------------------------------------------

def attach_edge_features_from_gdf(
    G: nx.MultiDiGraph,
    edges_gdf: gpd.GeoDataFrame,
    cols: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> nx.MultiDiGraph:
    """
    Attach selected columns from an edges GeoDataFrame to the corresponding edges in G.

    Matching is done by (u, v, key) columns in edges_gdf. Your edges parquet has:
    ['u','v','key', ... hazard cols ..., 'geometry']

    Parameters
    ----------
    G:
        NetworkX graph (MultiDiGraph recommended).
    edges_gdf:
        GeoDataFrame containing u/v/key and feature columns.
    cols:
        Which columns to attach (defaults to all non-index, non-geometry columns excluding u/v/key).
    strict:
        If True, raise if an (u,v,key) from edges_gdf is not found in G.

    Returns
    -------
    G (mutated in place and returned for convenience).
    """
    required = {"u", "v", "key"}
    if not required.issubset(edges_gdf.columns):
        raise ValueError(f"edges_gdf must contain columns {required}, got {set(edges_gdf.columns)}")

    if cols is None:
        cols = [c for c in edges_gdf.columns if c not in ("u", "v", "key", "geometry")]

    # Iterate rows; 2,389 edges is small enough for this direct approach.
    missing = 0
    for row in edges_gdf.itertuples(index=False):
        u = getattr(row, "u")
        v = getattr(row, "v")
        k = getattr(row, "key")
        if not G.has_edge(u, v, k):
            missing += 1
            if strict:
                raise KeyError(f"Graph does not have edge ({u}, {v}, {k})")
            continue

        data = G.edges[u, v, k]
        for c in cols:
            data[c] = getattr(row, c)

        # Attach geometry if present and graph edge lacks it
        if hasattr(row, "geometry") and row.geometry is not None:
            data.setdefault("geometry", row.geometry)

    if missing and not strict:
        # We don't print here (library code). Caller can log if needed.
        pass
    return G


# -----------------------------------------------------------------------------
# Failure model
# -----------------------------------------------------------------------------

def edge_failure_probability(
    edge_attrs: Dict[str, Any],
    month: str,
    params: FailureModelParams = FailureModelParams(),
    line_prefix: str = "p_hazard_",
    buf_prefix: str = "p_buffer_hazard_",
) -> float:
    """
    Compute p_fail for a single edge based on its attributes.

    Rules:
    1) If p_hazard_month >= water_over_threshold -> 1.0
    2) Else if p_buffer_hazard_month is finite -> clip to [0,1]
    3) Else -> p_background
    """
    p_line = edge_attrs.get(f"{line_prefix}{month}", None)
    p_buf = edge_attrs.get(f"{buf_prefix}{month}", None)

    # Rule 1: deterministic failure if "over water" proxy is very high
    if p_line is not None and np.isfinite(p_line) and float(p_line) >= params.water_over_threshold:
        return 1.0

    # Rule 2: buffer hazard probability
    if p_buf is not None and np.isfinite(p_buf):
        p = float(p_buf)
        return float(np.clip(p, 0.0, 1.0))

    # Rule 3: background risk for unknown
    return float(params.p_background)


def sample_failed_edges(
    G: nx.MultiDiGraph,
    month: str,
    params: FailureModelParams,
    rng: np.random.Generator,
) -> set[EdgeKey]:
    """
    Monte-Carlo sample edge failures.

    Returns a set of failed edge keys (u, v, k).
    """
    failed: set[EdgeKey] = set()

    # Works for MultiDiGraph; if Graph, keys=True will error.
    if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
        edge_iter = G.edges(keys=True, data=True)
        for u, v, k, data in edge_iter:
            p = edge_failure_probability(data, month=month, params=params)
            if rng.random() < p:
                failed.add((u, v, k))
    else:
        # Simple Graph fallback: represent key as None
        for u, v, data in G.edges(data=True):
            p = edge_failure_probability(data, month=month, params=params)
            if rng.random() < p:
                failed.add((u, v, None))

    return failed


def graph_without_failed_edges(
    G: nx.MultiDiGraph,
    failed: set[EdgeKey],
    copy_graph: bool = True,
) -> nx.MultiDiGraph:
    """
    Return a graph with failed edges removed.

    For small graphs, copying is fine. For larger graphs, you might want a view.
    """
    H = G.copy() if copy_graph else G
    for u, v, k in failed:
        if k is None:
            # simple Graph case
            if H.has_edge(u, v):
                H.remove_edge(u, v)
        else:
            if H.has_edge(u, v, k):
                H.remove_edge(u, v, k)
    return H


# -----------------------------------------------------------------------------
# Routing (trucks)
# -----------------------------------------------------------------------------

def shortest_path_safe(
    G: nx.Graph,
    source: Any,
    target: Any,
    weight: str = "length",
) -> Optional[List[Any]]:
    """
    Compute shortest path; return None if no path exists.
    """
    try:
        return nx.shortest_path(G, source, target, weight=weight)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def choose_exit_nodes_boundary_bbox(
    G: nx.Graph,
    bbox: Tuple[float, float, float, float],
    k: int = 20,
    epsilon_deg: float = 0.002,
) -> List[Any]:
    """
    Choose candidate exit nodes near the bbox boundary.

    bbox = (left, bottom, right, top) in EPSG:4326 degrees.
    epsilon_deg ~ 0.002 ≈ ~200m latitude-wise (roughly; varies with latitude).

    Returns up to k node IDs.
    """
    left, bottom, right, top = bbox

    candidates = []
    for n, d in G.nodes(data=True):
        x = d.get("x", None)
        y = d.get("y", None)
        if x is None or y is None:
            continue

        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            continue

        near_left = abs(x - left) <= epsilon_deg
        near_right = abs(x - right) <= epsilon_deg
        near_bottom = abs(y - bottom) <= epsilon_deg
        near_top = abs(y - top) <= epsilon_deg

        if near_left or near_right or near_bottom or near_top:
            candidates.append(n)

    # If too many, subsample evenly
    if not candidates:
        return []
    if len(candidates) <= k:
        return candidates

    idx = np.linspace(0, len(candidates) - 1, num=k).astype(int)
    return [candidates[i] for i in idx]


def routes_for_trucks(
    G: nx.Graph,
    source_node: Any,
    exit_nodes: Sequence[Any],
    n_trucks: int,
    weight: str = "length",
    rng: Optional[np.random.Generator] = None,
    strategy: str = "round_robin",
) -> List[Optional[List[Any]]]:
    """
    Compute one route per truck from source -> chosen exit.

    strategy:
    - "round_robin": cycles exits
    - "random": random exit for each truck (requires rng)
    """
    if not exit_nodes:
        return [None] * n_trucks

    routes: List[Optional[List[Any]]] = []
    for i in range(n_trucks):
        if strategy == "random":
            if rng is None:
                raise ValueError("rng is required for strategy='random'")
            target = exit_nodes[int(rng.integers(0, len(exit_nodes)))]
        else:
            target = exit_nodes[i % len(exit_nodes)]

        path = shortest_path_safe(G, source_node, target, weight=weight)
        routes.append(path)
    return routes


def node_paths_to_edge_keys(
    G: nx.MultiDiGraph,
    paths: Sequence[Optional[List[Any]]],
) -> set[EdgeKey]:
    """
    Convert node paths into a set of edge keys used by any path.

    For MultiDiGraph parallel edges: choose the edge with minimum weight ('length' if present).
    """
    evac_edges: set[EdgeKey] = set()

    for path in paths:
        if not path or len(path) < 2:
            continue
        for a, b in zip(path[:-1], path[1:]):
            if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
                # choose best key among parallel edges
                edict = G.get_edge_data(a, b, default={})
                if not edict:
                    continue
                best_k = None
                best_w = float("inf")
                for k, data in edict.items():
                    w = data.get("length", 1.0)
                    try:
                        w = float(w)
                    except Exception:
                        w = 1.0
                    if w < best_w:
                        best_w = w
                        best_k = k
                if best_k is not None:
                    evac_edges.add((a, b, best_k))
            else:
                evac_edges.add((a, b, None))

    return evac_edges


# -----------------------------------------------------------------------------
# Geometry extraction for evac paths
# -----------------------------------------------------------------------------

def edge_geometry(
    G: nx.MultiDiGraph,
    u: Any,
    v: Any,
    k: Any,
) -> Optional[LineString]:
    """
    Extract a LineString for an edge.

    Prefers stored 'geometry'. If missing, builds a straight line from node coords.
    """
    data = G.edges[u, v, k] if k is not None else G.edges[u, v]
    geom = data.get("geometry", None)
    if geom is not None and isinstance(geom, LineString):
        return geom

    # Build from node coordinates
    ux = G.nodes[u].get("x", None)
    uy = G.nodes[u].get("y", None)
    vx = G.nodes[v].get("x", None)
    vy = G.nodes[v].get("y", None)
    if None in (ux, uy, vx, vy):
        return None
    return LineString([(ux, uy), (vx, vy)])


def evac_edges_to_geoseries(
    G: nx.MultiDiGraph,
    evac_edges: Iterable[EdgeKey],
    crs: str = "EPSG:4326",
) -> gpd.GeoSeries:
    """
    Convert evac edge keys into a GeoSeries of LineStrings.
    """
    geoms = []
    for u, v, k in evac_edges:
        if k is None:
            continue
        try:
            geom = edge_geometry(G, u, v, k)
        except Exception:
            geom = None
        if geom is not None:
            geoms.append(geom)

    return gpd.GeoSeries(geoms, crs=crs)


# -----------------------------------------------------------------------------
# People placement
# -----------------------------------------------------------------------------

def assign_people_to_buildings(
    buildings: gpd.GeoDataFrame,
    n_people: int,
    rng: np.random.Generator,
    weight: str = "uniform",
    weight_col: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Assign people counts to building polygons.

    weight options:
    - "uniform": each person equally likely to be placed in any building
    - "area": weighted by polygon area (requires projected CRS; will project internally)
    - "column": weighted by `weight_col` (must be nonnegative)

    Returns a copy with a new integer column `people`.
    """
    b = buildings.copy()

    if len(b) == 0:
        b["people"] = 0
        return b

    if weight == "uniform":
        probs = np.ones(len(b), dtype=float)
        probs /= probs.sum()

    elif weight == "area":
        # area requires metric CRS; estimate UTM CRS from geometry
        metric_crs = b.estimate_utm_crs()
        area = b.to_crs(metric_crs).geometry.area.values.astype(float)
        area[~np.isfinite(area)] = 0.0
        if area.sum() <= 0:
            probs = np.ones(len(b), dtype=float) / len(b)
        else:
            probs = area / area.sum()

    elif weight == "column":
        if not weight_col:
            raise ValueError("weight_col is required when weight='column'")
        w = b[weight_col].to_numpy(dtype=float)
        w[~np.isfinite(w)] = 0.0
        w[w < 0] = 0.0
        if w.sum() <= 0:
            probs = np.ones(len(b), dtype=float) / len(b)
        else:
            probs = w / w.sum()

    else:
        raise ValueError(f"Unknown weight mode: {weight}")

    counts = rng.multinomial(n_people, probs)
    b["people"] = counts.astype(int)
    return b

def weighted_distance_percentiles_people(
    b_dist: gpd.GeoDataFrame,
    dist_col: str = "dist_to_evac_m",
    w_col: str = "people",
    ps=(0.50, 0.75, 0.90, 0.95),
) -> Dict[str, float]:
    """
    Weighted percentiles for b_dist[dist_col] using b_dist[w_col] as weights.
    Specific to your pipeline: dist_to_evac_m and people.
    """
    df = b_dist[[dist_col, w_col]].copy()
    df = df[df[w_col] > 0]

    if len(df) == 0:
        return {f"dist_p{int(p*100)}_m": float("nan") for p in ps}

    df = df.sort_values(dist_col)
    w = df[w_col].astype(float).to_numpy()
    x = df[dist_col].astype(float).to_numpy()

    cumw = w.cumsum()
    total = float(cumw[-1])

    out = {}
    for p in ps:
        cutoff = p * total
        idx = int((cumw >= cutoff).argmax())
        out[f"dist_p{int(p*100)}_m"] = float(x[idx])

    return out


def weighted_percentiles_people_distance(
    b_dist: gpd.GeoDataFrame,
    dist_col: str = "dist_to_evac_m",
    w_col: str = "people",
    ps=(0.50, 0.75, 0.90, 0.95),
) -> Dict[str, float]:
    """
    Weighted percentiles of building->evac distance using 'people' as weights.
    Assumes b_dist contains:
      - dist_to_evac_m (float, no NaNs)
      - people (>=0)
    Returns e.g. {'dist_p50_m': ..., 'dist_p75_m': ...}
    """
    # Keep only positive weights; zero-people buildings don't contribute
    df = b_dist[[dist_col, w_col]].copy()
    df = df[df[w_col] > 0]

    if len(df) == 0:
        # no people assigned anywhere (degenerate run)
        return {f"dist_p{int(p*100)}_m": float("nan") for p in ps}

    df = df.sort_values(dist_col)
    w = df[w_col].astype(float).to_numpy()
    x = df[dist_col].astype(float).to_numpy()

    cumw = w.cumsum()
    total = cumw[-1]

    out = {}
    for p in ps:
        cutoff = p * total
        idx = int((cumw >= cutoff).argmax())
        out[f"dist_p{int(p*100)}_m"] = float(x[idx])

    return out



# -----------------------------------------------------------------------------
# Accessibility scoring
# -----------------------------------------------------------------------------



def compute_building_distance_to_evac_paths(
    buildings: gpd.GeoDataFrame,
    evac_lines: gpd.GeoSeries,
    metric_crs: Optional[str] = None,
    use_centroids: bool = True,
) -> gpd.GeoDataFrame:
    """
    Compute distance (meters) from each building to the nearest evac line.

    Uses a unary_union of evac_lines and GeoPandas distance. This is fine for
    a few thousand lines; for very large line sets, swap to a spatial index
    nearest-neighbor approach.

    Returns a copy with `dist_to_evac_m`.
    """
    b = buildings.copy()
    if len(evac_lines) == 0:
        b["dist_to_evac_m"] = np.nan
        return b

    if metric_crs is None:
        metric_crs = b.estimate_utm_crs()

    b_m = b.to_crs(metric_crs)
    l_m = evac_lines.to_crs(metric_crs)

    target_geom = unary_union(list(l_m.values))
    pts = b_m.geometry.centroid if use_centroids else b_m.geometry
    b["dist_to_evac_m"] = pts.distance(target_geom).astype(float).values
    return b


def accessibility_metrics(
    buildings_with_dist: gpd.GeoDataFrame,
    radius_m: float = 500.0,
    people_col: str = "people",
) -> Dict[str, Optional[float]]:
    """
    People-weighted accessibility metrics.
    """
    b = buildings_with_dist.copy()
    if "dist_to_evac_m" not in b.columns:
        raise ValueError("Expected column dist_to_evac_m")

    b = b[np.isfinite(b["dist_to_evac_m"].to_numpy(dtype=float))].copy()
    if len(b) == 0:
        return {"access_frac": None, "mean_dist": None, "median_dist": None}

    if people_col not in b.columns:
        b[people_col] = 1

    people = b[people_col].to_numpy(dtype=float)
    total = float(people.sum())
    if total <= 0:
        return {"access_frac": None, "mean_dist": None, "median_dist": None}

    dist = b["dist_to_evac_m"].to_numpy(dtype=float)
    accessible = dist <= radius_m

    access_frac = float(people[accessible].sum() / total)
    mean_dist = float((dist * people).sum() / total)

    # median distance (unweighted median of buildings with people > 0)
    median_dist = float(np.nanmedian(dist[people > 0])) if np.any(people > 0) else float(np.nanmedian(dist))

    return {"access_frac": access_frac, "mean_dist": mean_dist, "median_dist": median_dist}


# -----------------------------------------------------------------------------
# One realization + Monte Carlo runner
# -----------------------------------------------------------------------------

def run_one_realization(
    G: nx.MultiDiGraph,
    buildings: gpd.GeoDataFrame,
    source_node: Any,
    exit_nodes: Sequence[Any],
    params: SimulationParams,
    rng: np.random.Generator,
    building_weight_mode: str = "uniform",
    building_weight_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one Monte Carlo realization:
    - sample road failures
    - route trucks on damaged graph
    - build evac edge set
    - assign people to buildings
    - compute distance to evac paths and accessibility KPIs
    + (added) evac corridor summary + weighted distance percentiles
    """
    failed = sample_failed_edges(G, params.month, params.failure, rng)
    H = graph_without_failed_edges(G, failed, copy_graph=True)

    routes = routes_for_trucks(
        H,
        source_node=source_node,
        exit_nodes=exit_nodes,
        n_trucks=params.n_trucks,
        weight=params.route_weight,
        rng=rng,
        strategy="round_robin",
    )

    # Existing
    evac_edges = node_paths_to_edge_keys(H, routes)
    evac_lines = evac_edges_to_geoseries(H, evac_edges, crs="EPSG:4326")

    # --- NEW: corridor summary (based on graph 'length' attribute) ---
    evac_edges_count = int(len(evac_edges))

    evac_total_length_m = 0.0
    for (u, v, k) in evac_edges:
        try:
            L = H.edges[u, v, k].get("length", 0.0)
        except Exception:
            L = 0.0
        try:
            evac_total_length_m += float(L)
        except Exception:
            # if length is somehow non-castable (shouldn't after your fix)
            pass

    n_routes_found = int(sum(1 for r in routes if r is not None))
    frac_routes_found = float(n_routes_found / params.n_trucks) if params.n_trucks else 0.0

    # exits reached: count unique exits that are actually the end of a found route
    reached_exits = set()
    for r in routes:
        if r is None or len(r) == 0:
            continue
        reached_exits.add(r[-1])
    n_exits = int(len(exit_nodes))
    n_exits_reached = int(sum(1 for e in exit_nodes if e in reached_exits))
    frac_exits_reached = float(n_exits_reached / n_exits) if n_exits else 0.0

    # People assignment + accessibility
    b_people = assign_people_to_buildings(
        buildings,
        n_people=params.n_people,
        rng=rng,
        weight=building_weight_mode,
        weight_col=building_weight_col,
    )

    b_dist = compute_building_distance_to_evac_paths(
        b_people,
        evac_lines,
        metric_crs=None,
        use_centroids=True
    )
    mets = accessibility_metrics(b_dist, radius_m=params.access_radius_m, people_col="people")

    # ---  weighted percentiles of dist_to_evac_m ---
    dist_pct = weighted_percentiles_people_distance(
        b_dist,
        dist_col="dist_to_evac_m",
        w_col="people",
        ps=(0.50, 0.75, 0.90, 0.95),
    )

    return {
        "month": params.month,
        "n_edges_total": int(H.number_of_edges()),
        "n_failed_edges": int(len(failed)),
        "n_trucks": int(params.n_trucks),
        "n_routes_found": n_routes_found,
        "frac_routes_found": frac_routes_found,

        # Corridor summary
        "evac_edges_count": evac_edges_count,
        "evac_total_length_m": float(evac_total_length_m),
        "n_exits": n_exits,
        "n_exits_reached": n_exits_reached,
        "frac_exits_reached": frac_exits_reached,

        # Existing accessibility metrics
        **mets,

        # Percentiles
        **dist_pct,
    }


def run_monte_carlo(
    G: nx.MultiDiGraph,
    buildings: gpd.GeoDataFrame,
    source_node: Any,
    exit_nodes: Sequence[Any],
    params: SimulationParams,
    n_runs: int = 100,
    building_weight_mode: str = "uniform",
    building_weight_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run multiple realizations and return a results DataFrame.
    """
    rng = np.random.default_rng(params.seed)
    rows: List[Dict[str, Any]] = []
    for _ in range(int(n_runs)):
        rows.append(
            run_one_realization(
                G=G,
                buildings=buildings,
                source_node=source_node,
                exit_nodes=exit_nodes,
                params=params,
                rng=rng,
                building_weight_mode=building_weight_mode,
                building_weight_col=building_weight_col,
            )
        )
    return pd.DataFrame(rows)
