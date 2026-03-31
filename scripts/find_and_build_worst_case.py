from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import geopandas as gpd
import numpy as np
import osmnx as ox


def _parse_int_list(value: str) -> list[int]:
    vals: list[int] = []
    for s in str(value).split(","):
        t = s.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        raise ValueError("Expected at least one integer")
    return vals


def _expand_bbox(bbox: tuple[float, float, float, float], buffer_deg: float) -> tuple[float, float, float, float]:
    left, bottom, right, top = bbox
    b = max(float(buffer_deg), 0.0)
    return (left - b, bottom - b, right + b, top + b)


def _dedupe_preserve_order(nodes: Sequence) -> list:
    out: list = []
    seen = set()
    for n in nodes:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _subsample_evenly(nodes: Sequence, k: int) -> list:
    arr = list(nodes)
    if k <= 0 or len(arr) <= k:
        return arr
    idx = np.linspace(0, len(arr) - 1, num=int(k)).astype(int)
    return [arr[i] for i in idx]


def _discover_exit_nodes_via_buffered_graph(
    *,
    g_buffered,
    g_sim,
    aoi_bbox: tuple[float, float, float, float],
    epsilon_deg: float,
    k: int,
) -> list:
    left, bottom, right, top = aoi_bbox
    ug = g_buffered.to_undirected(as_view=False)

    candidate_xy: list[tuple[float, float]] = []
    for n, d in g_buffered.nodes(data=True):
        x = d.get("x", None)
        y = d.get("y", None)
        try:
            x = float(x)
            y = float(y)
        except Exception:
            continue

        near_boundary = (
            abs(x - left) <= epsilon_deg
            or abs(x - right) <= epsilon_deg
            or abs(y - bottom) <= epsilon_deg
            or abs(y - top) <= epsilon_deg
        )
        if not near_boundary:
            continue

        has_outward_neighbor = False
        for nbr in ug.neighbors(n):
            nd = ug.nodes[nbr]
            x2 = nd.get("x", None)
            y2 = nd.get("y", None)
            try:
                x2 = float(x2)
                y2 = float(y2)
            except Exception:
                continue
            outside = (x2 < left) or (x2 > right) or (y2 < bottom) or (y2 > top)
            if outside:
                has_outward_neighbor = True
                break

        if has_outward_neighbor:
            candidate_xy.append((x, y))

    if not candidate_xy:
        return []

    mapped = []
    for x, y in candidate_xy:
        try:
            mapped.append(ox.distance.nearest_nodes(g_sim, X=float(x), Y=float(y)))
        except Exception:
            continue

    mapped = _dedupe_preserve_order(mapped)
    mapped = [n for n in mapped if n in g_sim.nodes]
    return _subsample_evenly(mapped, int(k))


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Scan many random seeds at fixed WOT using evac-path computation only, "
            "stop on first worst-case seed (evac_edges<=threshold), then build one full scenario."
        )
    )
    ap.add_argument("--wot", type=float, default=0.75, help="water_over_threshold to scan")
    ap.add_argument("--p-background", type=float, default=0.02, help="p_background value")
    ap.add_argument("--month", type=str, default="oct", help="Damage month (default: oct)")
    ap.add_argument("--n-trucks", type=int, default=20, help="Number of trucks used in route generation")
    ap.add_argument("--target-max-evac-edges", type=int, default=1, help="Stop when evac edge count is <= this value")
    ap.add_argument("--seed-list", type=str, default="", help="Optional comma-separated explicit seed list (overrides range)")
    ap.add_argument("--seed-start", type=int, default=1, help="Start of seed range (inclusive)")
    ap.add_argument("--seed-end", type=int, default=10000, help="End of seed range (inclusive)")
    ap.add_argument("--source-lon", type=float, default=91.007571, help="Fixed source longitude")
    ap.add_argument("--source-lat", type=float, default=26.325782, help="Fixed source latitude")
    ap.add_argument("--exit-k", type=int, default=24, help="Maximum number of exits")
    ap.add_argument("--exit-epsilon-deg", type=float, default=0.003, help="Exit boundary epsilon")
    ap.add_argument(
        "--exit-discovery-mode",
        type=str,
        default="buffered",
        choices=["buffered", "legacy"],
        help="Exit discovery mode (matches rebuild script defaults)",
    )
    ap.add_argument("--exit-buffer-deg", type=float, default=0.05, help="AOI buffer (deg) for buffered exit discovery")
    ap.add_argument(
        "--exit-buffer-graphml",
        type=str,
        default="outputs/tables/roads_exit_buffered.graphml",
        help="Cached buffered graph path",
    )
    ap.add_argument("--build-worst-case", action="store_true", help="Build full scenario immediately after finding a seed")
    ap.add_argument(
        "--worst-case-subdir",
        type=str,
        default="scenarios_worst_case",
        help="Output scenarios subdir used for built worst-case scenario",
    )
    ap.add_argument(
        "--clear-worst-case-output",
        action="store_true",
        help="Clear the worst-case output subdir before building full scenario",
    )
    ap.add_argument(
        "--accel-mode",
        type=str,
        default="auto",
        choices=["auto", "none", "numba-cpu", "numba-cuda"],
        help="Forwarded to rebuild_weekly_scenarios.py",
    )
    ap.add_argument("--max-export-file-mb", type=float, default=95.0)
    ap.add_argument("--warn-file-over-mb", type=float, default=95.0)
    ap.add_argument("--fail-file-over-mb", type=float, default=100.0)
    ap.add_argument("--python-exe", type=str, default=sys.executable, help="Python interpreter for full build step")
    ap.add_argument(
        "--scan-report-json",
        type=str,
        default="outputs/tables/worst_case_scan_report.json",
        help="Where to write scan/build metadata report",
    )
    return ap.parse_args()


def _seed_sequence(args: argparse.Namespace) -> list[int]:
    if str(args.seed_list).strip():
        return _parse_int_list(args.seed_list)
    if int(args.seed_end) < int(args.seed_start):
        raise ValueError("--seed-end must be >= --seed-start")
    return list(range(int(args.seed_start), int(args.seed_end) + 1))


def main() -> None:
    args = _parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))

    from src import io, roads, simulation
    from src.config import EVAC_DRIVABLE_HIGHWAYS

    seed_values = _seed_sequence(args)
    print(
        f"scan_start wot={float(args.wot):.6g} p_background={float(args.p_background):.6g} "
        f"seeds={len(seed_values)} target_max_evac_edges={int(args.target_max_evac_edges)}",
        flush=True,
    )

    graphml_path = Path("outputs/tables/roads_drive_lcc.graphml")
    roads_features_parquet = Path("outputs/tables/roads_edges_features.parquet")
    roads_features_gpkg = Path("outputs/tables/roads_edges_features.gpkg")

    aoi = io.load_aoi_full().to_crs("EPSG:4326")
    bounds = tuple(aoi.total_bounds.tolist())

    g = ox.load_graphml(graphml_path)
    if roads_features_parquet.exists():
        edges_features = gpd.read_parquet(roads_features_parquet)
    elif roads_features_gpkg.exists():
        edges_features = gpd.read_file(roads_features_gpkg)
    else:
        raise FileNotFoundError("Missing roads_edges_features parquet/gpkg")

    g = simulation.attach_edge_features_from_gdf(g, edges_features, strict=False)

    source_node = ox.distance.nearest_nodes(g, X=float(args.source_lon), Y=float(args.source_lat))
    print(f"source_node={source_node}", flush=True)

    exit_nodes: list = []
    if args.exit_discovery_mode == "buffered":
        buffered_bbox = _expand_bbox(bounds, float(args.exit_buffer_deg))
        b_left, b_bottom, b_right, b_top = buffered_bbox
        buffered_graphml = Path(args.exit_buffer_graphml)
        g_buffered = None

        if buffered_graphml.exists():
            try:
                g_buffered = ox.load_graphml(buffered_graphml)
                print(f"exit_buffer_graph_loaded={buffered_graphml}", flush=True)
            except Exception as e:
                print(f"exit_buffer_graph_load_failed={type(e).__name__}", flush=True)

        if g_buffered is None:
            try:
                g_buffered = roads.download_graph_from_bbox(
                    bbox=(b_top, b_bottom, b_right, b_left),
                    network_type="all",
                    simplify=True,
                    retain_all=True,
                    timeout=180,
                )
                g_buffered = roads.filter_graph_by_highway(g_buffered, EVAC_DRIVABLE_HIGHWAYS)
                roads.save_graphml(g_buffered, buffered_graphml)
                print(f"exit_buffer_graph_saved={buffered_graphml}", flush=True)
            except Exception as e:
                print(f"exit_buffer_graph_download_failed={type(e).__name__}", flush=True)
                g_buffered = None

        if g_buffered is not None:
            exit_nodes = _discover_exit_nodes_via_buffered_graph(
                g_buffered=g_buffered,
                g_sim=g,
                aoi_bbox=bounds,
                epsilon_deg=float(args.exit_epsilon_deg),
                k=int(args.exit_k),
            )
            print(f"exit_discovery_mode=buffered exit_count={len(exit_nodes)}", flush=True)

    if not exit_nodes:
        exit_nodes = simulation.choose_exit_nodes_boundary_bbox(
            g,
            bbox=bounds,
            k=int(args.exit_k),
            epsilon_deg=float(args.exit_epsilon_deg),
        )
        print(f"exit_discovery_mode=legacy exit_count={len(exit_nodes)}", flush=True)

    exit_nodes = [n for n in _dedupe_preserve_order(exit_nodes) if n in g.nodes]
    if not exit_nodes:
        raise RuntimeError("No exit nodes found")

    found: dict | None = None
    for i, seed in enumerate(seed_values, start=1):
        rng = np.random.default_rng(int(seed))

        failed = simulation.sample_failed_edges(
            g,
            month=str(args.month),
            params=simulation.FailureModelParams(
                water_over_threshold=float(args.wot),
                p_background=float(args.p_background),
            ),
            rng=rng,
        )
        H = simulation.graph_without_failed_edges(g, failed, copy_graph=True)
        routes = simulation.routes_for_trucks(
            H,
            source_node=source_node,
            exit_nodes=exit_nodes,
            n_trucks=int(args.n_trucks),
            weight="length",
            rng=rng,
            strategy="round_robin",
        )
        evac_edges = simulation.node_paths_to_edge_keys(H, routes)

        evac_edges_count = int(len(evac_edges))
        n_routes_found = int(sum(1 for r in routes if r is not None))

        print(
            f"scan seed={seed} idx={i}/{len(seed_values)} evac_edges={evac_edges_count} "
            f"routes_found={n_routes_found}/{int(args.n_trucks)} failed_edges={len(failed)}",
            flush=True,
        )

        if evac_edges_count <= int(args.target_max_evac_edges):
            found = {
                "seed": int(seed),
                "water_over_threshold": float(args.wot),
                "p_background": float(args.p_background),
                "evac_edges_count": int(evac_edges_count),
                "n_routes_found": int(n_routes_found),
                "n_trucks": int(args.n_trucks),
                "failed_edges_count": int(len(failed)),
                "exit_count": int(len(exit_nodes)),
                "month": str(args.month),
            }
            print(
                f"worst_case_found seed={seed} wot={float(args.wot):.6g} evac_edges={evac_edges_count}",
                flush=True,
            )
            break

    report = {
        "scan": {
            "wot": float(args.wot),
            "p_background": float(args.p_background),
            "month": str(args.month),
            "seed_count": int(len(seed_values)),
            "seed_start": int(seed_values[0]) if seed_values else None,
            "seed_end": int(seed_values[-1]) if seed_values else None,
            "target_max_evac_edges": int(args.target_max_evac_edges),
        },
        "found": found,
        "build": None,
    }

    if found and bool(args.build_worst_case):
        rebuild_script = project_root / "scripts" / "rebuild_weekly_scenarios.py"
        build_cmd = [
            str(args.python_exe),
            str(rebuild_script),
            "--scenarios-subdir",
            str(args.worst_case_subdir),
            "--seeds",
            str(found["seed"]),
            "--water-over-thresholds",
            f"{float(found['water_over_threshold']):.6g}",
            "--p-backgrounds",
            f"{float(found['p_background']):.6g}",
            "--accel-mode",
            str(args.accel_mode),
            "--max-export-file-mb",
            f"{float(args.max_export_file_mb):.6g}",
            "--warn-file-over-mb",
            f"{float(args.warn_file_over_mb):.6g}",
            "--fail-file-over-mb",
            f"{float(args.fail_file_over_mb):.6g}",
        ]
        if bool(args.clear_worst_case_output):
            build_cmd.append("--clear-output")

        print("build_start", flush=True)
        print("[run] " + " ".join(build_cmd), flush=True)
        rc = int(subprocess.run(build_cmd).returncode)
        report["build"] = {
            "attempted": True,
            "return_code": rc,
            "scenarios_subdir": str(args.worst_case_subdir),
            "clear_output": bool(args.clear_worst_case_output),
            "command": build_cmd,
        }
        if rc != 0:
            raise SystemExit(rc)
        print("build_done", flush=True)

    report_path = Path(args.scan_report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"scan_report={report_path.as_posix()}", flush=True)

    if found is None:
        print("No seed met the worst-case threshold in the scanned range.", flush=True)
        raise SystemExit(2)

    print("done", flush=True)


if __name__ == "__main__":
    main()
