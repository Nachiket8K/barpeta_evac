from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd


def _parse_int_list(value: str) -> list[int]:
    vals: list[int] = []
    for s in str(value).split(","):
        t = s.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        raise ValueError("Expected at least one integer value")
    return vals


def _parse_float_list(value: str) -> list[float]:
    vals: list[float] = []
    for s in str(value).split(","):
        t = s.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("Expected at least one float value")
    return vals


def _pbg_slug(v: float) -> str:
    s = f"{float(v):.6g}"
    return s.replace("-", "m").replace(".", "p")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rebuild weekly evacuation scenarios for selected seeds/background failure rates")
    ap.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds (e.g., 1,5,7,42,100). Default uses config DEFAULT_SCENARIO_SEEDS.",
    )
    ap.add_argument(
        "--p-backgrounds",
        type=str,
        default="0.02",
        help="Comma-separated p_background values (e.g., 0,0.02,0.05,0.1,0.5)",
    )
    ap.add_argument(
        "--scenarios-subdir",
        type=str,
        default="scenarios",
        help="Output subdirectory under docs/ (default: scenarios)",
    )
    ap.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear output scenarios folder before run.",
    )
    ap.add_argument(
        "--force-pbg-in-id",
        action="store_true",
        help="Always include p_background suffix in scenario_id.",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    from src import io, raster_features, scenario_export, simulation
    from src.config import (
        DEFAULT_ACCESS_RADIUS_M,
        DEFAULT_SCENARIO_SEEDS,
        WEEKLY_END_DATE,
        WEEKLY_START_DATE,
        WET_CLASS_VALUES,
    )

    docs_root = Path("docs")
    scenarios_root = docs_root / str(args.scenarios_subdir)

    start_date = WEEKLY_START_DATE
    end_date = WEEKLY_END_DATE
    mask_dem_weight = 0.25
    mask_threshold = 0.50
    mask_rgba = (65, 155, 223)
    mask_alpha = 150

    scenario_seeds = _parse_int_list(args.seeds) if str(args.seeds).strip() else list(DEFAULT_SCENARIO_SEEDS)
    p_background_values = _parse_float_list(args.p_backgrounds)
    damage_month = "oct"
    n_trucks = 20
    access_radius_m = float(DEFAULT_ACCESS_RADIUS_M)
    water_over_threshold = 0.95
    route_weight = "length"
    exit_k = 24
    exit_epsilon_deg = 0.003

    graphml_path = Path("outputs/tables/roads_drive_lcc.graphml")
    roads_features_parquet = Path("outputs/tables/roads_edges_features.parquet")
    roads_features_gpkg = Path("outputs/tables/roads_edges_features.gpkg")
    buildings_snapshot_path = Path("outputs/tables/buildings_features_snapshot.parquet")

    weekly_dates = pd.date_range(start_date, end_date, freq="7D")
    if weekly_dates[-1] != pd.Timestamp(end_date):
        weekly_dates = weekly_dates.append(pd.DatetimeIndex([pd.Timestamp(end_date)]))
    frame_dates = [str(d.date()) for d in weekly_dates]
    n_frames = len(weekly_dates)

    aoi = io.load_aoi_full().to_crs("EPSG:4326")
    buildings = (
        gpd.read_parquet(buildings_snapshot_path)
        if buildings_snapshot_path.exists()
        else io.load_buildings()
    )
    if buildings.crs is None:
        buildings = buildings.set_crs(aoi.crs)

    with io.open_lulc_month("may") as src_may, io.open_lulc_month("oct") as src_oct, io.open_dem() as src_dem:
        weekly = raster_features.build_weekly_wet_masks_from_may_oct_dem(
            src_may,
            src_oct,
            src_dem,
            n_frames=n_frames,
            wet_classes=WET_CLASS_VALUES,
            dem_weight=mask_dem_weight,
            threshold=mask_threshold,
            return_scores=False,
        )
        water_bounds = [
            float(src_may.bounds.left),
            float(src_may.bounds.bottom),
            float(src_may.bounds.right),
            float(src_may.bounds.top),
        ]
    masks = weekly["masks"]

    g = ox.load_graphml(graphml_path)
    if roads_features_parquet.exists():
        edges_features = gpd.read_parquet(roads_features_parquet)
    elif roads_features_gpkg.exists():
        edges_features = gpd.read_file(roads_features_gpkg)
    else:
        raise FileNotFoundError("Missing roads_edges_features parquet/gpkg")
    g = simulation.attach_edge_features_from_gdf(g, edges_features, strict=False)

    roads_base = edges_features[[c for c in ["u", "v", "key", "geometry"] if c in edges_features.columns]].copy()
    if roads_base.crs is None:
        roads_base = roads_base.set_crs("EPSG:4326")
    else:
        roads_base = roads_base.to_crs("EPSG:4326")
    roads_base["status"] = "intact"

    bounds = tuple(aoi.total_bounds.tolist())
    centroid = aoi.unary_union.centroid
    source_node = ox.distance.nearest_nodes(g, X=float(centroid.x), Y=float(centroid.y))
    exit_nodes = simulation.choose_exit_nodes_boundary_bbox(
        g,
        bbox=bounds,
        k=exit_k,
        epsilon_deg=exit_epsilon_deg,
    )
    if not exit_nodes:
        raise RuntimeError("No exit nodes found; increase EXIT_EPSILON_DEG")

    with io.open_population_raster() as pop_src:
        buildings_people = simulation.allocate_population_from_raster_to_buildings(
            buildings,
            pop_raster_src=pop_src,
            out_col="people",
            building_id_col="building_id",
        )
    pop_method = "raster"

    # Zero-population buildings do not affect people-weighted KPIs and are expensive
    # to process for distance calculations/export. Keep populated buildings for simulation.
    buildings_people_sim = buildings_people[buildings_people["people"] > 0].copy()
    if len(buildings_people_sim):
        bb = buildings_people_sim.geometry.bounds
        buildings_people_sim = buildings_people_sim.set_geometry(
            gpd.points_from_xy(
                (bb.minx.to_numpy(dtype=float) + bb.maxx.to_numpy(dtype=float)) * 0.5,
                (bb.miny.to_numpy(dtype=float) + bb.maxy.to_numpy(dtype=float)) * 0.5,
                crs=buildings_people_sim.crs,
            )
        )

    total_people = int(buildings_people["people"].sum())
    print(f"population_method={pop_method}")
    print(f"total_people={total_people}")
    print(f"buildings_total={len(buildings_people)} populated_buildings={len(buildings_people_sim)}")

    if args.clear_output:
        scenario_export.reset_dir(scenarios_root)
    else:
        scenario_export.ensure_dir(scenarios_root)
    index_path = scenarios_root / "index.json"
    summary_rows: list[dict] = []

    include_pbg_in_id = bool(args.force_pbg_in_id) or (len(p_background_values) > 1)

    for p_background in p_background_values:
        for seed in scenario_seeds:
            if include_pbg_in_id:
                scenario_id = f"barpeta_{damage_month}_seed_{seed}_pbg_{_pbg_slug(p_background)}"
            else:
                scenario_id = f"barpeta_{damage_month}_seed_{seed}"
            scenario_dir = scenarios_root / scenario_id
            scenario_export.reset_dir(scenario_dir)

            scenario_export.write_weekly_frame_stack(
                masks,
                scenario_dir / "frames" / "water_mask",
                rgb=mask_rgba,
                alpha=mask_alpha,
            )
            
            print(f"frames_written={scenario_id}", flush=True)

            params = simulation.SimulationParams(
                month=damage_month,
                n_trucks=n_trucks,
                n_people=total_people,
                access_radius_m=access_radius_m,
                seed=int(seed),
                route_weight=route_weight,
                failure=simulation.FailureModelParams(
                    water_over_threshold=water_over_threshold,
                    p_background=p_background,
                ),
            )

            print(f"running_realization={scenario_id}", flush=True)
            detail = simulation.run_one_realization_detailed(
                g,
                buildings_people_sim,
                source_node,
                exit_nodes,
                params,
                np.random.default_rng(seed),
                people_col="people",
            )
            print(f"realization_done={scenario_id}", flush=True)

            failed_keys = (
                detail["failed_edges"][["u", "v", "key"]].copy()
                if len(detail["failed_edges"])
                else pd.DataFrame(columns=["u", "v", "key"])
            )
            evac_keys = (
                detail["evac_edges"][["u", "v", "key"]].copy()
                if len(detail["evac_edges"])
                else pd.DataFrame(columns=["u", "v", "key"])
            )

            roads_failed = scenario_export.edge_keys_to_geodataframe(roads_base, failed_keys)
            roads_failed["status"] = "failed"
            evac_paths = scenario_export.edge_keys_to_geodataframe(roads_base, evac_keys)
            evac_paths["status"] = "evac"

            scenario_export.write_geojson(scenario_dir / "roads_base.geojson", roads_base[["u", "v", "key", "status", "geometry"]])
            scenario_export.write_geojson(scenario_dir / "roads_failed.geojson", roads_failed[["u", "v", "key", "status", "geometry"]])
            scenario_export.write_geojson(scenario_dir / "evac_paths.geojson", evac_paths[["u", "v", "key", "status", "geometry"]])

            metrics = dict(detail["metrics"])
            metrics.update(
                {
                    "scenario_id": scenario_id,
                    "seed": int(seed),
                    "p_background": float(p_background),
                    "allocation_method": pop_method,
                    "n_people_allocated": total_people,
                    "n_frames": int(n_frames),
                    "date_start": str(pd.Timestamp(start_date).date()),
                    "date_end": str(pd.Timestamp(end_date).date()),
                }
            )
            scenario_export.write_json(scenario_dir / "metrics.json", metrics)
            scenario_export.write_csv(scenario_dir / "failed_edges.csv", detail["failed_edges"])
            scenario_export.write_csv(scenario_dir / "evac_edges.csv", detail["evac_edges"])
            scenario_export.write_json(
                scenario_dir / "routes.json",
                {
                    "source_node": str(source_node),
                    "exit_nodes": [str(e) for e in exit_nodes],
                    "routes": [[str(n) for n in r] if r is not None else None for r in detail["routes"]],
                },
            )

            print(f"export_buildings_geojson={scenario_id}", flush=True)
            buildings_meta = scenario_export.export_buildings_access_geojson_chunks(
                scenario_dir,
                detail["buildings"],
                people_col="people",
                dist_col="dist_to_connected_road_m",
                building_id_col="building_id",
                include_zero_people=False,
                max_chunk_size_mb=95.0,
                min_chunks_if_exceeds=2,
            )
            zones_meta = scenario_export.export_accessibility_zones_geojson(
                scenario_dir,
                detail["buildings"],
                out_name="accessibility_zones.geojson",
                people_col="people",
                dist_col="dist_to_connected_road_m",
                accessible_col="is_accessible_network",
                default_threshold_m=access_radius_m,
                cell_size_m=600.0,
            )
            print(f"export_buildings_geojson_done={scenario_id}", flush=True)
            scenario_export.write_admin_boundary_geojson(scenario_dir, aoi_gdf=aoi)

            manifest = scenario_export.build_manifest(
                scenario_id=scenario_id,
                n_frames=n_frames,
                start_date=str(pd.Timestamp(start_date).date()),
                end_date=str(pd.Timestamp(end_date).date()),
                parameters={
                    "seed": int(seed),
                    "damage_month": damage_month,
                    "access_radius_m": access_radius_m,
                    "water_over_threshold": water_over_threshold,
                    "p_background": p_background,
                    "mask_dem_weight": mask_dem_weight,
                    "mask_threshold": mask_threshold,
                    "population_allocation_method": pop_method,
                },
                assets={
                    "metrics": "metrics.json",
                    "failed_edges": "failed_edges.csv",
                    "evac_edges": "evac_edges.csv",
                    "routes": "routes.json",
                    "buildings_access_chunks": buildings_meta["files"],
                    "buildings_access_format": "geojson",
                    "buildings_access_summary": {
                        "feature_count": int(buildings_meta["feature_count"]),
                        "people_total": int(buildings_meta["people_total"]),
                        "chunk_count": int(buildings_meta["chunk_count"]),
                        "include_zero_people": False,
                    },
                    "accessibility_zones": zones_meta["file"],
                    "accessibility_zones_summary": {
                        "zone_count": int(zones_meta.get("zone_count", 0)),
                        "people_total": int(zones_meta.get("people_total", 0)),
                        "cell_size_m": float(zones_meta.get("cell_size_m", 600.0)),
                        "default_threshold_m": float(zones_meta.get("default_threshold_m", access_radius_m)),
                    },
                    "admin_boundary": "admin_boundary.geojson",
                    "roads_base": "roads_base.geojson",
                    "roads_failed": "roads_failed.geojson",
                    "evac_paths": "evac_paths.geojson",
                },
                aoi_bounds_wgs84=list(bounds),
                overlay_bounds_wgs84=water_bounds,
                frame_dates=frame_dates,
                vector_layers={
                    "roads_base": {"type": "line", "asset": "roads_base.geojson", "color": "#111111"},
                    "roads_failed": {"type": "line", "asset": "roads_failed.geojson", "color": "#d62728"},
                    "evac_paths": {"type": "line", "asset": "evac_paths.geojson", "color": "#2ca02c"},
                    "admin_boundary": {"type": "line", "asset": "admin_boundary.geojson", "color": "#1f77b4"},
                    "buildings_access": {
                        "type": "circle",
                        "asset": "buildings_access_chunks",
                        "color_ok": "#2ca02c",
                        "color_bad": "#d62728",
                    },
                    "accessibility_zones": {
                        "type": "fill",
                        "asset": "accessibility_zones.geojson",
                        "color_ok": "#2ca02c",
                        "color_bad": "#d62728",
                    },
                },
            )
            scenario_export.write_json(scenario_dir / "manifest.json", manifest)

            scenario_export.update_index_json(
                index_path,
                {
                    "scenario_id": scenario_id,
                    "path": f"{str(args.scenarios_subdir).rstrip('/')}/{scenario_id}/manifest.json",
                    "parameters": {
                        "seed": int(seed),
                        "damage_month": damage_month,
                        "access_radius_m": access_radius_m,
                        "p_background": float(p_background),
                    },
                },
                options_updates={
                    "seed": [int(seed)],
                    "damage_month": [damage_month],
                    "access_radius_m": [access_radius_m],
                    "p_background": [float(p_background)],
                },
            )
            summary_rows.append(metrics)
            print(
                f"built={scenario_id} p_background={p_background:.6g} chunks={buildings_meta['chunk_count']} "
                f"people_total={buildings_meta['people_total']} zones={zones_meta.get('zone_count', 0)}",
                flush=True,
            )

    sort_cols = [c for c in ["p_background", "seed"] if c in pd.DataFrame(summary_rows).columns]
    summary = pd.DataFrame(summary_rows)
    if len(summary) and sort_cols:
        summary = summary.sort_values(sort_cols).reset_index(drop=True)
    summary.to_csv(scenarios_root / "scenario_summary.csv", index=False)
    print("done", flush=True)


if __name__ == '__main__':
    main()