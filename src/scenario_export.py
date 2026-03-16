"""
Scenario export helpers for static GitHub Pages playback.

This module writes Kivu-style artifacts:
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

import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image


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
                "frame_path_template": "frames/water_mask/t{frame:04d}.png",
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
