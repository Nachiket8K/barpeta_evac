"""
Scenario export helpers for static GitHub Pages playback.

This module writes detailed artifacts:
  - docs/scenarios/index.json
  - docs/scenarios/<scenario_id>/manifest.json
  - docs/scenarios/<scenario_id>/frames/<layer>/t####.png
  - scenario detail tables (CSV/JSON)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import shutil
import math

import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image
from shapely import wkt
from shapely.geometry import box


DEFAULT_MAX_EXPORT_FILE_MB: float = 95.0
GITHUB_HARD_FILE_LIMIT_MB: float = 100.0


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_dir(path: Path) -> Path:
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path


def write_csv(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_geojson(path: Path, gdf: gpd.GeoDataFrame) -> Path:
    """
    Write GeoDataFrame as GeoJSON using in-process JSON serialization.
    (Avoids dependence on external Fiona/GDAL write drivers.)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = gdf.to_json(drop_id=True)
    path.write_text(text, encoding="utf-8")
    return path


def list_files_over_size(root: Path, max_size_mb: float) -> List[Tuple[Path, int]]:
    """
    Return files under root larger than max_size_mb as (path, bytes), sorted largest-first.
    """
    if not root.exists():
        return []

    max_bytes = int(float(max_size_mb) * 1024.0 * 1024.0)
    out: List[Tuple[Path, int]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            sz = int(p.stat().st_size)
        except Exception:
            continue
        if sz > max_bytes:
            out.append((p, sz))

    out.sort(key=lambda x: x[1], reverse=True)
    return out


def _feature_json_size_estimate(
    xs: np.ndarray,
    ys: np.ndarray,
    props_df: pd.DataFrame,
    *,
    dist_prop_name: str = "dist_to_evac_m",
    include_legacy_dist_alias: bool = False,
    sample_n: int = 2000,
    coord_precision: int = 6,
) -> float:
    """
    Estimate average serialized bytes per GeoJSON feature from a sample.
    """
    n = len(xs)
    if n == 0:
        return 128.0

    s_n = int(min(sample_n, n))
    idx = np.linspace(0, n - 1, num=s_n).astype(int)

    sizes: List[int] = []
    for i in idx:
        x = xs[i]
        y = ys[i]
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(x), coord_precision), round(float(y), coord_precision)],
            },
            "properties": {
                "building_id": int(props_df.iloc[i]["building_id"]),
                "people": int(props_df.iloc[i]["people"]),
                dist_prop_name: float(props_df.iloc[i]["dist_value"]),
            },
        }
        if include_legacy_dist_alias and dist_prop_name != "dist_to_evac_m":
            feat["properties"]["dist_to_evac_m"] = float(props_df.iloc[i]["dist_value"])
        sizes.append(len(json.dumps(feat, separators=(",", ":"), ensure_ascii=False)))

    return float(np.mean(sizes)) if sizes else 128.0


def export_buildings_access_geojson_chunks(
    scenario_dir: Path,
    buildings: gpd.GeoDataFrame,
    *,
    people_col: str = "people",
    dist_col: str = "dist_to_evac_m",
    building_id_col: str = "building_id",
    include_zero_people: bool = False,
    max_chunk_size_mb: float = 95.0,
    min_chunks_if_exceeds: int = 2,
    coord_precision: int = 6,
) -> Dict[str, Any]:
    """
    Export building accessibility as chunked GeoJSON Point layers for web rendering.

    - Geometry is converted to representative points in EPSG:4326.
    - Only required properties are retained: building_id, people, dist_to_evac_m.
    - If estimated payload exceeds max_chunk_size_mb, split into multiple files
      (at least `min_chunks_if_exceeds`, default 2).
    """
    ensure_dir(scenario_dir)

    if "geometry" not in buildings.columns:
        raise ValueError("buildings must contain a geometry column")

    b = buildings.copy()
    if building_id_col not in b.columns:
        b[building_id_col] = np.arange(len(b), dtype=int)

    if people_col not in b.columns:
        b[people_col] = 0
    if dist_col not in b.columns:
        b[dist_col] = np.nan

    if not include_zero_people:
        b = b[pd.to_numeric(b[people_col], errors="coerce").fillna(0).astype(float) > 0].copy()

    if b.crs is None:
        b = b.set_crs("EPSG:4326")
    else:
        b = b.to_crs("EPSG:4326")

    b = b[np.isfinite(pd.to_numeric(b[dist_col], errors="coerce"))].copy()

    out_files: List[str] = []
    if len(b) == 0:
        empty_name = "buildings_access_part1.geojson"
        payload = {"type": "FeatureCollection", "features": []}
        (scenario_dir / empty_name).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return {
            "files": [empty_name],
            "feature_count": 0,
            "people_total": 0,
            "chunk_count": 1,
            "include_zero_people": bool(include_zero_people),
        }

    # Build minimal props and lightweight display points.
    # Using polygon bbox centers is much faster than representative_point() at this scale
    # and is sufficient for point-based thematic rendering in the web viewer.
    bb = b.geometry.bounds
    xs = np.round(((bb.minx.to_numpy(dtype=float) + bb.maxx.to_numpy(dtype=float)) * 0.5), coord_precision)
    ys = np.round(((bb.miny.to_numpy(dtype=float) + bb.maxy.to_numpy(dtype=float)) * 0.5), coord_precision)
    dist_prop_name = str(dist_col) if isinstance(dist_col, str) and dist_col else "dist_to_evac_m"
    include_legacy_dist_alias = dist_prop_name != "dist_to_evac_m"

    props = pd.DataFrame(
        {
            "building_id": pd.to_numeric(b[building_id_col], errors="coerce").fillna(0).astype(np.int64),
            "people": pd.to_numeric(b[people_col], errors="coerce").fillna(0).astype(np.int64),
            "dist_value": pd.to_numeric(b[dist_col], errors="coerce").fillna(np.nan).astype(float),
        }
    )

    n = len(props)
    avg_feature_bytes = _feature_json_size_estimate(
        xs,
        ys,
        props,
        dist_prop_name=dist_prop_name,
        include_legacy_dist_alias=include_legacy_dist_alias,
        coord_precision=coord_precision,
    )
    estimated_total_bytes = avg_feature_bytes * float(n) + 64.0
    max_bytes = float(max_chunk_size_mb) * 1024.0 * 1024.0

    if estimated_total_bytes > max_bytes:
        chunk_count = max(int(math.ceil(estimated_total_bytes / max_bytes)), int(min_chunks_if_exceeds))
    else:
        chunk_count = 1

    while True:
        # Clear existing parts from previous iteration.
        for old in scenario_dir.glob("buildings_access_part*.geojson"):
            old.unlink(missing_ok=True)

        out_files = []
        chunk_sizes: List[int] = []
        rows_per_chunk = int(math.ceil(n / max(chunk_count, 1)))

        # Pre-materialize arrays once (avoid DataFrame .iloc in tight loops).
        ids = props["building_id"].to_numpy(dtype=np.int64, copy=False)
        ppl = props["people"].to_numpy(dtype=np.int64, copy=False)
        dists = props["dist_value"].to_numpy(dtype=float, copy=False)

        for i in range(chunk_count):
            start = i * rows_per_chunk
            end = min((i + 1) * rows_per_chunk, n)
            if start >= end:
                break
            name = f"buildings_access_part{i + 1}.geojson"
            out_path = scenario_dir / name

            # Stream JSON directly to avoid constructing very large intermediate lists.
            with out_path.open("w", encoding="utf-8") as f:
                f.write('{"type":"FeatureCollection","features":[')
                first = True
                for j in range(start, end):
                    if not first:
                        f.write(",")
                    else:
                        first = False

                    # Manual compact serialization is significantly faster here than
                    # allocating dicts + calling json.dumps for every feature.
                    if include_legacy_dist_alias:
                        f.write(
                            f'{{"type":"Feature","geometry":{{"type":"Point","coordinates":[{xs[j]:.{coord_precision}f},{ys[j]:.{coord_precision}f}]}}'
                            f',"properties":{{"building_id":{int(ids[j])},"people":{int(ppl[j])},"{dist_prop_name}":{float(dists[j]):.3f},"dist_to_evac_m":{float(dists[j]):.3f}}}}}'
                        )
                    else:
                        f.write(
                            f'{{"type":"Feature","geometry":{{"type":"Point","coordinates":[{xs[j]:.{coord_precision}f},{ys[j]:.{coord_precision}f}]}}'
                            f',"properties":{{"building_id":{int(ids[j])},"people":{int(ppl[j])},"{dist_prop_name}":{float(dists[j]):.3f}}}}}'
                        )

                f.write("]}")

            out_files.append(name)
            chunk_sizes.append(int(out_path.stat().st_size))

        if not chunk_sizes:
            break

        if max(chunk_sizes) <= max_bytes or chunk_count >= n:
            break

        # Increase chunk count and rewrite until each part is under threshold.
        chunk_count = min(int(chunk_count * 2), int(n))

    return {
        "files": out_files,
        "feature_count": int(n),
        "people_total": int(props["people"].sum()),
        "chunk_count": int(len(out_files)),
        "include_zero_people": bool(include_zero_people),
        "estimated_total_mb": float(estimated_total_bytes / (1024.0 * 1024.0)),
        "max_chunk_size_mb": float(max_chunk_size_mb),
        "max_chunk_written_mb": float(max(chunk_sizes) / (1024.0 * 1024.0)) if 'chunk_sizes' in locals() and chunk_sizes else 0.0,
    }


def export_accessibility_zones_geojson(
    scenario_dir: Path,
    buildings: gpd.GeoDataFrame,
    *,
    out_name: str = "accessibility_zones.geojson",
    people_col: str = "people",
    dist_col: str = "dist_to_connected_road_m",
    accessible_col: str = "is_accessible_network",
    default_threshold_m: float = 1000.0,
    cell_size_m: float = 150.0,
) -> Dict[str, Any]:
    """
    Export rough low-zoom safe/unsafe accessibility zones as grid polygons.

    Zones are people-weighted aggregates from buildings and are intended for
    low-zoom rendering when individual houses are not shown.
    """
    ensure_dir(scenario_dir)

    b = buildings.copy()
    if len(b) == 0 or "geometry" not in b.columns:
        payload = {"type": "FeatureCollection", "features": []}
        out_path = scenario_dir / out_name
        out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return {
            "file": out_name,
            "zone_count": 0,
            "people_total": 0,
            "cell_size_m": float(cell_size_m),
        }

    if b.crs is None:
        b = b.set_crs("EPSG:4326")

    if people_col not in b.columns:
        b[people_col] = 0
    if dist_col not in b.columns:
        b[dist_col] = np.nan

    b[people_col] = pd.to_numeric(b[people_col], errors="coerce").fillna(0).astype(float)
    b[dist_col] = pd.to_numeric(b[dist_col], errors="coerce").astype(float)
    b = b[(b[people_col] > 0) & np.isfinite(b[dist_col])].copy()

    if len(b) == 0:
        payload = {"type": "FeatureCollection", "features": []}
        out_path = scenario_dir / out_name
        out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return {
            "file": out_name,
            "zone_count": 0,
            "people_total": 0,
            "cell_size_m": float(cell_size_m),
        }

    metric_crs = b.estimate_utm_crs()
    b_m = b.to_crs(metric_crs)

    # Use representative points for robust zone assignment.
    pts = b_m.geometry.centroid
    x = pts.x.to_numpy(dtype=float)
    y = pts.y.to_numpy(dtype=float)
    ppl = b_m[people_col].to_numpy(dtype=float)
    dist = b_m[dist_col].to_numpy(dtype=float)

    cell = float(max(cell_size_m, 50.0))
    gx = np.floor(x / cell).astype(np.int64)
    gy = np.floor(y / cell).astype(np.int64)

    safe_mask = None
    if accessible_col in b_m.columns:
        safe_mask = b_m[accessible_col].astype(bool).to_numpy()
    else:
        safe_mask = dist <= float(default_threshold_m)

    df = pd.DataFrame(
        {
            "gx": gx,
            "gy": gy,
            "people": ppl,
            "dist_wsum": dist * ppl,
            "safe_people": np.where(safe_mask, ppl, 0.0),
        }
    )

    agg = (
        df.groupby(["gx", "gy"], as_index=False)
        .agg(
            people_total=("people", "sum"),
            dist_wsum=("dist_wsum", "sum"),
            accessible_people=("safe_people", "sum"),
        )
    )
    if len(agg) == 0:
        payload = {"type": "FeatureCollection", "features": []}
        out_path = scenario_dir / out_name
        out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return {
            "file": out_name,
            "zone_count": 0,
            "people_total": 0,
            "cell_size_m": float(cell_size_m),
        }

    agg["dist_mean_m"] = np.where(agg["people_total"] > 0, agg["dist_wsum"] / agg["people_total"], np.nan)
    agg["affected_people"] = np.maximum(agg["people_total"] - agg["accessible_people"], 0.0)
    agg["access_frac"] = np.where(agg["people_total"] > 0, agg["accessible_people"] / agg["people_total"], np.nan)
    agg["zone_status"] = np.where(agg["dist_mean_m"] <= float(default_threshold_m), "safe", "unsafe")

    geoms = []
    for r in agg.itertuples(index=False):
        minx = float(r.gx) * cell
        miny = float(r.gy) * cell
        geoms.append(box(minx, miny, minx + cell, miny + cell))

    zones_m = gpd.GeoDataFrame(agg, geometry=geoms, crs=metric_crs)
    zones = zones_m.to_crs("EPSG:4326")
    zones["zone_id"] = np.arange(len(zones), dtype=int)

    keep_cols = [
        "zone_id",
        "zone_status",
        "people_total",
        "accessible_people",
        "affected_people",
        "access_frac",
        "dist_mean_m",
        "geometry",
    ]
    zones_out = zones[keep_cols].copy()

    out_path = scenario_dir / out_name
    write_geojson(out_path, zones_out)

    return {
        "file": out_name,
        "zone_count": int(len(zones_out)),
        "people_total": int(round(float(zones_out["people_total"].sum()))),
        "cell_size_m": float(cell),
        "default_threshold_m": float(default_threshold_m),
    }


def summarize_stranded_people_grid(
    buildings: gpd.GeoDataFrame,
    *,
    people_col: str = "people",
    dist_col: str = "dist_to_connected_road_m",
    accessible_col: str = "is_accessible_network",
    default_threshold_m: float = 1000.0,
    cell_size_m: float = 30.0,
) -> Dict[str, Any]:
    """
    Summarize stranded people into a fixed-size metric grid for one realization.

    Returns a compact table keyed by (gx, gy) with per-cell totals:
      - people_total
      - stranded_people
    """
    b = buildings.copy()
    if len(b) == 0 or "geometry" not in b.columns:
        return {
            "grid": pd.DataFrame(columns=["gx", "gy", "people_total", "stranded_people"]),
            "metric_crs": None,
            "cell_size_m": float(cell_size_m),
        }

    if b.crs is None:
        b = b.set_crs("EPSG:4326")

    if people_col not in b.columns:
        b[people_col] = 0
    if dist_col not in b.columns:
        b[dist_col] = np.nan

    b[people_col] = pd.to_numeric(b[people_col], errors="coerce").fillna(0).astype(float)
    b[dist_col] = pd.to_numeric(b[dist_col], errors="coerce").astype(float)
    b = b[b[people_col] > 0].copy()
    if len(b) == 0:
        return {
            "grid": pd.DataFrame(columns=["gx", "gy", "people_total", "stranded_people"]),
            "metric_crs": None,
            "cell_size_m": float(cell_size_m),
        }

    metric_crs = b.estimate_utm_crs()
    b_m = b.to_crs(metric_crs)

    pts = b_m.geometry.centroid
    x = pts.x.to_numpy(dtype=float)
    y = pts.y.to_numpy(dtype=float)
    ppl = b_m[people_col].to_numpy(dtype=float)

    cell = float(max(float(cell_size_m), 5.0))
    gx = np.floor(x / cell).astype(np.int64)
    gy = np.floor(y / cell).astype(np.int64)

    if accessible_col in b_m.columns:
        stranded_mask = ~b_m[accessible_col].astype(bool).to_numpy()
    else:
        dist = b_m[dist_col].to_numpy(dtype=float)
        stranded_mask = np.isfinite(dist) & (dist > float(default_threshold_m))

    df = pd.DataFrame(
        {
            "gx": gx,
            "gy": gy,
            "people_total": ppl,
            "stranded_people": np.where(stranded_mask, ppl, 0.0),
        }
    )
    out = (
        df.groupby(["gx", "gy"], as_index=False)
        .agg(
            people_total=("people_total", "sum"),
            stranded_people=("stranded_people", "sum"),
        )
        .reset_index(drop=True)
    )

    return {
        "grid": out,
        "metric_crs": metric_crs,
        "cell_size_m": float(cell),
    }


def export_seed_stranded_aggregate_geojson(
    out_path: Path,
    seed_grids: Sequence[pd.DataFrame],
    *,
    metric_crs: Any,
    cell_size_m: float = 30.0,
    max_chunk_size_mb: float = DEFAULT_MAX_EXPORT_FILE_MB,
    min_chunks_if_exceeds: int = 2,
) -> Dict[str, Any]:
    """
    Export cross-seed stranded-grid aggregation for one threshold setting.

    Output properties per cell:
      - seed_count
      - stranded_seed_hits
      - stranded_seed_frac
      - mean_stranded_people
      - max_stranded_people
      - mean_people_total
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    valid = [g for g in seed_grids if isinstance(g, pd.DataFrame) and len(g)]
    seed_count = int(len(seed_grids))
    if seed_count <= 0 or not valid or metric_crs is None:
        payload = {"type": "FeatureCollection", "features": []}
        out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return {
            "file": out_path.name,
            "files": [out_path.name],
            "cell_count": 0,
            "seed_count": int(seed_count),
            "cell_size_m": float(cell_size_m),
        }

    parts: List[pd.DataFrame] = []
    for i, g in enumerate(seed_grids):
        d = g.copy()
        if len(d) == 0:
            continue
        cols = {"gx", "gy", "people_total", "stranded_people"}
        if not cols.issubset(set(d.columns)):
            continue
        d = d[["gx", "gy", "people_total", "stranded_people"]].copy()
        d["seed_idx"] = int(i)
        parts.append(d)

    if not parts:
        payload = {"type": "FeatureCollection", "features": []}
        out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return {
            "file": out_path.name,
            "files": [out_path.name],
            "cell_count": 0,
            "seed_count": int(seed_count),
            "cell_size_m": float(cell_size_m),
        }

    cat = pd.concat(parts, ignore_index=True)
    cat["stranded_hit"] = (pd.to_numeric(cat["stranded_people"], errors="coerce").fillna(0) > 0).astype(int)

    agg = (
        cat.groupby(["gx", "gy"], as_index=False)
        .agg(
            stranded_sum=("stranded_people", "sum"),
            stranded_max=("stranded_people", "max"),
            people_sum=("people_total", "sum"),
            stranded_seed_hits=("stranded_hit", "sum"),
        )
        .reset_index(drop=True)
    )

    denom = max(int(seed_count), 1)
    agg["seed_count"] = int(seed_count)
    agg["stranded_seed_frac"] = agg["stranded_seed_hits"] / float(denom)
    agg["mean_stranded_people"] = agg["stranded_sum"] / float(denom)
    agg["max_stranded_people"] = agg["stranded_max"]
    agg["mean_people_total"] = agg["people_sum"] / float(denom)

    cell = float(max(float(cell_size_m), 5.0))
    geoms = []
    for r in agg.itertuples(index=False):
        minx = float(r.gx) * cell
        miny = float(r.gy) * cell
        geoms.append(box(minx, miny, minx + cell, miny + cell))

    g_m = gpd.GeoDataFrame(agg, geometry=geoms, crs=metric_crs)
    g_wgs = g_m.to_crs("EPSG:4326")
    g_wgs["cell_id"] = np.arange(len(g_wgs), dtype=int)

    keep_cols = [
        "cell_id",
        "seed_count",
        "stranded_seed_hits",
        "stranded_seed_frac",
        "mean_stranded_people",
        "max_stranded_people",
        "mean_people_total",
        "geometry",
    ]
    g_out = g_wgs[keep_cols].copy()

    n = int(len(g_out))
    max_bytes = int(float(max_chunk_size_mb) * 1024.0 * 1024.0)
    chunk_count = 1

    # Iterative split-until-under-threshold approach.
    while True:
        # Remove previous chunk artifacts for this basename.
        for old in out_path.parent.glob(f"{out_path.stem}_part*.geojson"):
            old.unlink(missing_ok=True)

        files: List[str] = []
        chunk_sizes: List[int] = []

        if chunk_count <= 1:
            write_geojson(out_path, g_out)
            files = [out_path.name]
            chunk_sizes = [int(out_path.stat().st_size)]
        else:
            out_path.unlink(missing_ok=True)
            rows_per_chunk = int(max(math.ceil(n / float(chunk_count)), 1))
            for i in range(chunk_count):
                start = i * rows_per_chunk
                end = min((i + 1) * rows_per_chunk, n)
                if start >= end:
                    break
                part_name = f"{out_path.stem}_part{i + 1}.geojson"
                part_path = out_path.parent / part_name
                write_geojson(part_path, g_out.iloc[start:end].copy())
                files.append(part_name)
                chunk_sizes.append(int(part_path.stat().st_size))

        if not chunk_sizes:
            break

        if max(chunk_sizes) <= max_bytes or chunk_count >= max(n, 1):
            return {
                "file": files[0],
                "files": files,
                "cell_count": int(n),
                "seed_count": int(seed_count),
                "cell_size_m": float(cell),
                "chunk_count": int(len(files)),
                "max_chunk_size_mb": float(max_chunk_size_mb),
                "max_chunk_written_mb": float(max(chunk_sizes) / (1024.0 * 1024.0)),
            }

        chunk_count = max(int(chunk_count * 2), int(min_chunks_if_exceeds))


def write_admin_boundary_geojson(
    scenario_dir: Path,
    *,
    aoi_gdf: Optional[gpd.GeoDataFrame] = None,
    aoi_bounds_wgs84: Optional[Sequence[float]] = None,
    filename: str = "admin_boundary.geojson",
) -> Path:
    """
    Write a jurisdiction/admin boundary overlay as a simple GeoJSON polygon.
    """
    if aoi_gdf is None and aoi_bounds_wgs84 is None:
        raise ValueError("Provide either aoi_gdf or aoi_bounds_wgs84")

    if aoi_gdf is not None:
        g = aoi_gdf.copy()
        if g.crs is None:
            g = g.set_crs("EPSG:4326")
        else:
            g = g.to_crs("EPSG:4326")
        out = g[["geometry"]].copy()
    else:
        b = list(aoi_bounds_wgs84)
        if len(b) != 4:
            raise ValueError("aoi_bounds_wgs84 must be [minx, miny, maxx, maxy]")
        out = gpd.GeoDataFrame({"name": ["aoi_boundary"]}, geometry=[box(float(b[0]), float(b[1]), float(b[2]), float(b[3]))], crs="EPSG:4326")

    return write_geojson(scenario_dir / filename, out)


def load_buildings_access_csv_as_gdf(
    csv_path: Path,
    *,
    crs: str = "EPSG:4326",
    include_zero_people: bool = False,
) -> gpd.GeoDataFrame:
    """
    Load legacy buildings_access.csv (with geometry_wkt) as a GeoDataFrame.
    """
    use_cols = ["building_id", "people", "dist_to_evac_m", "geometry_wkt"]

    if include_zero_people:
        df = pd.read_csv(csv_path, usecols=lambda c: c in use_cols)
    else:
        parts: List[pd.DataFrame] = []
        for chunk in pd.read_csv(csv_path, usecols=lambda c: c in use_cols, chunksize=100_000):
            p = pd.to_numeric(chunk.get("people", 0), errors="coerce").fillna(0)
            keep = chunk[p > 0].copy()
            if len(keep):
                parts.append(keep)
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=use_cols)

    if "geometry_wkt" not in df.columns:
        raise KeyError(f"Expected geometry_wkt column in {csv_path}")

    if "building_id" not in df.columns:
        df["building_id"] = np.arange(len(df), dtype=int)
    if "people" not in df.columns:
        df["people"] = 0
    if "dist_to_evac_m" not in df.columns:
        df["dist_to_evac_m"] = np.nan

    geom = df["geometry_wkt"].fillna("").map(lambda s: wkt.loads(s) if isinstance(s, str) and s else None)
    out = gpd.GeoDataFrame(df.drop(columns=["geometry_wkt"]), geometry=geom, crs=crs)
    return out


def add_buildings_overlay_to_scenario_bundle(
    scenario_dir: Path,
    *,
    max_chunk_size_mb: float = 95.0,
    include_zero_people: bool = False,
) -> Path:
    """
    Convert legacy buildings_access.csv into chunked GeoJSON building overlays and
    enrich the scenario manifest accordingly.
    """
    manifest_path = scenario_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest in {scenario_dir}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assets = manifest.setdefault("assets", {})
    layers = manifest.setdefault("layers", {})

    csv_rel = assets.get("buildings_access", "buildings_access.csv")
    csv_path = scenario_dir / csv_rel
    if not csv_path.exists():
        return manifest_path

    b = load_buildings_access_csv_as_gdf(csv_path, include_zero_people=include_zero_people)
    meta = export_buildings_access_geojson_chunks(
        scenario_dir,
        b,
        include_zero_people=include_zero_people,
        max_chunk_size_mb=max_chunk_size_mb,
    )

    assets["buildings_access_chunks"] = meta["files"]
    assets["buildings_access_format"] = "geojson"
    assets["buildings_access_summary"] = {
        "feature_count": int(meta.get("feature_count", 0)),
        "people_total": int(meta.get("people_total", 0)),
        "chunk_count": int(meta.get("chunk_count", len(meta.get("files", [])))),
        "include_zero_people": bool(meta.get("include_zero_people", False)),
    }

    if "admin_boundary" not in assets:
        aoi_bounds = manifest.get("aoi", {}).get("bounds_wgs84")
        if aoi_bounds:
            write_admin_boundary_geojson(scenario_dir, aoi_bounds_wgs84=aoi_bounds)
            assets["admin_boundary"] = "admin_boundary.geojson"

    layers["buildings_access"] = {
        "type": "circle",
        "asset": "buildings_access_chunks",
        "radius": 2.0,
        "color_ok": "#2ca02c",
        "color_bad": "#d62728",
    }

    return write_json(manifest_path, manifest)


def add_buildings_overlay_to_all_scenarios(
    scenarios_root: Path,
    *,
    max_chunk_size_mb: float = 95.0,
    include_zero_people: bool = False,
) -> List[Path]:
    """
    Enrich all scenario bundles with chunked building overlays.
    """
    updated: List[Path] = []
    for d in sorted(scenarios_root.iterdir()):
        if d.is_dir() and (d / "manifest.json").exists():
            updated.append(
                add_buildings_overlay_to_scenario_bundle(
                    d,
                    max_chunk_size_mb=max_chunk_size_mb,
                    include_zero_people=include_zero_people,
                )
            )
    return updated


def edge_keys_to_geodataframe(
    edges_gdf: gpd.GeoDataFrame,
    keys_df: pd.DataFrame,
    *,
    key_cols: Tuple[str, str, str] = ("u", "v", "key"),
) -> gpd.GeoDataFrame:
    """
    Subset edge GeoDataFrame by (u, v, key) table using type-agnostic matching.
    """
    u_col, v_col, k_col = key_cols
    if any(c not in edges_gdf.columns for c in key_cols):
        raise KeyError(f"edges_gdf must contain {key_cols}")
    if any(c not in keys_df.columns for c in key_cols):
        raise KeyError(f"keys_df must contain {key_cols}")

    e = edges_gdf.copy()
    k = keys_df.copy()
    e["_edge_key"] = e[u_col].astype(str) + "|" + e[v_col].astype(str) + "|" + e[k_col].astype(str)
    k["_edge_key"] = k[u_col].astype(str) + "|" + k[v_col].astype(str) + "|" + k[k_col].astype(str)
    keep = set(k["_edge_key"].tolist())
    out = e[e["_edge_key"].isin(keep)].copy()
    return out.drop(columns=["_edge_key"])


def write_png_mask_rgba(
    mask: np.ndarray,
    out_path: Path,
    rgb: Tuple[int, int, int] = (65, 155, 223),
    alpha: int = 140,
) -> Path:
    """
    Write a boolean/fractional mask as an RGBA transparent PNG.
    """
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("mask must be a 2D array")

    rgba = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
    on = m > 0
    rgba[on, 0] = int(rgb[0])
    rgba[on, 1] = int(rgb[1])
    rgba[on, 2] = int(rgb[2])
    rgba[on, 3] = int(alpha)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_path)
    return out_path


def write_weekly_frame_stack(
    masks: Sequence[np.ndarray],
    out_dir: Path,
    *,
    rgb: Tuple[int, int, int] = (65, 155, 223),
    alpha: int = 140,
    prefix: str = "t",
) -> List[Path]:
    """
    Write frame list to t0000.png, t0001.png, ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i, mask in enumerate(masks):
        p = out_dir / f"{prefix}{i:04d}.png"
        write_png_mask_rgba(mask, p, rgb=rgb, alpha=alpha)
        paths.append(p)
    return paths


def ensure_weekly_frame_stack(
    masks: Sequence[np.ndarray],
    out_dir: Path,
    *,
    rgb: Tuple[int, int, int] = (65, 155, 223),
    alpha: int = 140,
    prefix: str = "t",
) -> List[Path]:
    """
    Ensure weekly frame stack exists on disk; regenerate only if incomplete/missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = int(len(masks))
    existing = sorted(out_dir.glob(f"{prefix}[0-9][0-9][0-9][0-9].png"))
    if len(existing) == expected:
        return existing

    return write_weekly_frame_stack(
        masks,
        out_dir,
        rgb=rgb,
        alpha=alpha,
        prefix=prefix,
    )


def build_manifest(
    *,
    scenario_id: str,
    n_frames: int,
    start_date: str,
    end_date: str,
    parameters: Dict[str, Any],
    assets: Optional[Dict[str, Any]] = None,
    aoi_bounds_wgs84: Optional[Sequence[float]] = None,
    overlay_bounds_wgs84: Optional[Sequence[float]] = None,
    frame_dates: Optional[Sequence[str]] = None,
    vector_layers: Optional[Dict[str, Any]] = None,
    water_frame_path_template: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a Kivu-style manifest payload.
    """
    payload: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "time": {
            "start": start_date,
            "end": end_date,
            "n_frames": int(n_frames),
        },
        "parameters": parameters,
        "layers": {
            "water_mask": {
                "frame_path_template": str(
                    water_frame_path_template
                    if water_frame_path_template is not None
                    else "frames/water_mask/t{frame:04d}.png"
                ),
                "frame_count": int(n_frames),
            }
        },
    }
    if frame_dates is not None:
        payload["time"]["frame_dates"] = list(frame_dates)

    if overlay_bounds_wgs84 is not None:
        payload["layers"]["water_mask"]["bounds_wgs84"] = list(overlay_bounds_wgs84)

    if vector_layers:
        payload["layers"].update(vector_layers)

    if assets:
        payload["assets"] = assets
    if aoi_bounds_wgs84 is not None:
        payload["aoi"] = {"bounds_wgs84": list(aoi_bounds_wgs84)}
    return payload


def update_index_json(
    index_path: Path,
    scenario_entry: Dict[str, Any],
    *,
    options_updates: Optional[Dict[str, Sequence[Any]]] = None,
) -> Path:
    """
    Upsert a scenario entry into docs/scenarios/index.json.
    """
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {"scenarios": [], "options": {}}

    scenarios = payload.setdefault("scenarios", [])
    sid = scenario_entry.get("scenario_id")
    scenarios = [s for s in scenarios if s.get("scenario_id") != sid]
    scenarios.append(scenario_entry)
    payload["scenarios"] = sorted(scenarios, key=lambda x: str(x.get("scenario_id", "")))

    if options_updates:
        opts = payload.setdefault("options", {})
        for k, vals in options_updates.items():
            current = set(opts.get(k, []))
            current.update(list(vals))
            opts[k] = sorted(current)

    return write_json(index_path, payload)


def remove_index_entries_by_path_prefix(index_path: Path, path_prefix: str) -> Path:
    """
    Remove scenario entries whose manifest path starts with a prefix and rebuild options
    from remaining entries.
    """
    if not index_path.exists():
        return index_path

    with index_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    prefix = str(path_prefix).strip().rstrip("/") + "/"
    scenarios = payload.get("scenarios", [])
    kept = [s for s in scenarios if not str(s.get("path", "")).startswith(prefix)]
    payload["scenarios"] = sorted(kept, key=lambda x: str(x.get("scenario_id", "")))

    opt_vals: Dict[str, set] = {}
    for s in kept:
        p = s.get("parameters", {}) or {}
        for k, v in p.items():
            opt_vals.setdefault(str(k), set()).add(v)

    opts: Dict[str, Any] = {}
    for k, vals in opt_vals.items():
        try:
            opts[k] = sorted(vals)
        except Exception:
            opts[k] = sorted(list(vals), key=lambda x: str(x))
    payload["options"] = opts

    return write_json(index_path, payload)


def _build_frame_dates(start_date: str, end_date: str, n_frames: int) -> List[str]:
    """
    Construct frame date labels aligned with weekly cadence when possible.
    """
    if n_frames <= 0:
        return []

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    dates = pd.date_range(start, end, freq="7D")
    if len(dates) == 0 or dates[-1] != end:
        dates = dates.append(pd.DatetimeIndex([end]))

    if len(dates) != int(n_frames):
        dates = pd.date_range(start, end, periods=int(n_frames))

    return [str(pd.Timestamp(d).date()) for d in dates]


def add_vector_overlays_to_scenario_bundle(
    scenario_dir: Path,
    edges_gdf: gpd.GeoDataFrame,
) -> Path:
    """
    Add roads/failed/evac GeoJSON overlays and enrich manifest in an existing scenario bundle.
    """
    manifest_path = scenario_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest in {scenario_dir}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    roads_base = edges_gdf[[c for c in ["u", "v", "key", "geometry"] if c in edges_gdf.columns]].copy()
    if roads_base.crs is None:
        roads_base = roads_base.set_crs("EPSG:4326")
    else:
        roads_base = roads_base.to_crs("EPSG:4326")
    roads_base["status"] = "intact"

    failed_csv = scenario_dir / "failed_edges.csv"
    evac_csv = scenario_dir / "evac_edges.csv"

    failed_df = pd.read_csv(failed_csv) if failed_csv.exists() else pd.DataFrame(columns=["u", "v", "key"])
    evac_df = pd.read_csv(evac_csv) if evac_csv.exists() else pd.DataFrame(columns=["u", "v", "key"])

    failed_keys = failed_df[[c for c in ["u", "v", "key"] if c in failed_df.columns]].copy()
    evac_keys = evac_df[[c for c in ["u", "v", "key"] if c in evac_df.columns]].copy()

    if len(failed_keys) > 0:
        roads_failed = edge_keys_to_geodataframe(roads_base, failed_keys)
    else:
        roads_failed = roads_base.iloc[0:0].copy()
    roads_failed["status"] = "failed"

    if len(evac_keys) > 0:
        evac_paths = edge_keys_to_geodataframe(roads_base, evac_keys)
    else:
        evac_paths = roads_base.iloc[0:0].copy()
    evac_paths["status"] = "evac"

    write_geojson(scenario_dir / "roads_base.geojson", roads_base[["u", "v", "key", "status", "geometry"]])
    write_geojson(scenario_dir / "roads_failed.geojson", roads_failed[["u", "v", "key", "status", "geometry"]])
    write_geojson(scenario_dir / "evac_paths.geojson", evac_paths[["u", "v", "key", "status", "geometry"]])

    assets = manifest.setdefault("assets", {})
    assets["roads_base"] = "roads_base.geojson"
    assets["roads_failed"] = "roads_failed.geojson"
    assets["evac_paths"] = "evac_paths.geojson"

    layers = manifest.setdefault("layers", {})
    layers["roads_base"] = {"type": "line", "asset": "roads_base.geojson", "color": "#111111"}
    layers["roads_failed"] = {"type": "line", "asset": "roads_failed.geojson", "color": "#d62728"}
    layers["evac_paths"] = {"type": "line", "asset": "evac_paths.geojson", "color": "#2ca02c"}

    water = layers.setdefault("water_mask", {})
    if "bounds_wgs84" not in water and "aoi" in manifest and "bounds_wgs84" in manifest["aoi"]:
        water["bounds_wgs84"] = list(manifest["aoi"]["bounds_wgs84"])

    t = manifest.setdefault("time", {})
    if "frame_dates" not in t and all(k in t for k in ["start", "end", "n_frames"]):
        t["frame_dates"] = _build_frame_dates(t["start"], t["end"], int(t["n_frames"]))

    return write_json(manifest_path, manifest)


def add_vector_overlays_to_all_scenarios(
    scenarios_root: Path,
    edges_gdf: gpd.GeoDataFrame,
) -> List[Path]:
    """
    Enrich all scenario bundles under docs/scenarios with vector overlays.
    """
    updated: List[Path] = []
    for d in sorted(scenarios_root.iterdir()):
        if d.is_dir() and (d / "manifest.json").exists():
            updated.append(add_vector_overlays_to_scenario_bundle(d, edges_gdf))
    return updated
