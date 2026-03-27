from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Chunk existing large aggregate GeoJSONs and patch scenario manifests."
    )
    ap.add_argument(
        "--scenarios-root",
        type=str,
        default="docs/scenarios_wot_seed_sweep",
        help="Scenarios root containing aggregates/ and scenario manifest folders.",
    )
    ap.add_argument(
        "--max-file-mb",
        type=float,
        default=95.0,
        help="Target max size per GeoJSON file/chunk.",
    )
    return ap.parse_args()


def _write_geojson(path: Path, payload: dict) -> int:
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return int(path.stat().st_size)


def _split_geojson_if_needed(path: Path, max_file_mb: float) -> Tuple[List[str], float]:
    max_bytes = int(float(max_file_mb) * 1024.0 * 1024.0)
    if not path.exists():
        return [], 0.0

    original_size = int(path.stat().st_size)
    if original_size <= max_bytes:
        return [path.name], original_size / (1024.0 * 1024.0)

    obj = json.loads(path.read_text(encoding="utf-8"))
    features = list(obj.get("features", []))
    if not features:
        return [path.name], original_size / (1024.0 * 1024.0)

    base = {k: v for k, v in obj.items() if k != "features"}
    part_count = max(2, int(math.ceil(original_size / float(max_bytes))))

    while True:
        # Clean previous attempt files.
        for old in path.parent.glob(f"{path.stem}_part*{path.suffix}"):
            old.unlink(missing_ok=True)

        rows_per_part = int(math.ceil(len(features) / float(part_count)))
        part_names: List[str] = []
        part_sizes: List[int] = []

        for i in range(part_count):
            lo = i * rows_per_part
            hi = min((i + 1) * rows_per_part, len(features))
            if lo >= hi:
                break
            part_name = f"{path.stem}_part{i + 1}{path.suffix}"
            part_path = path.parent / part_name
            payload = dict(base)
            payload["features"] = features[lo:hi]
            sz = _write_geojson(part_path, payload)
            part_names.append(part_name)
            part_sizes.append(sz)

        if not part_sizes:
            raise RuntimeError(f"Failed to write chunk parts for {path}")

        if max(part_sizes) <= max_bytes or part_count >= len(features):
            # Remove oversize original once part files are ready.
            path.unlink(missing_ok=True)
            return part_names, max(part_sizes) / (1024.0 * 1024.0)

        part_count = min(len(features), int(part_count * 2))


def _patch_manifests(
    scenarios_root: Path,
    parts_map: Dict[str, List[str]],
    max_file_mb: float,
    max_written_map: Dict[str, float],
) -> int:
    patched = 0
    for manifest_path in scenarios_root.glob("barpeta_*/manifest.json"):
        obj = json.loads(manifest_path.read_text(encoding="utf-8"))
        assets = obj.setdefault("assets", {})
        rel = assets.get("stranded_seed_aggregate")
        if not isinstance(rel, str):
            continue
        base_name = Path(rel).name
        part_names = parts_map.get(base_name)
        if not part_names:
            continue

        part_rels = [f"../aggregates/{n}" for n in part_names]
        assets["stranded_seed_aggregate"] = part_rels[0]
        assets["stranded_seed_aggregate_parts"] = part_rels

        summary = assets.setdefault("stranded_seed_aggregate_summary", {})
        summary["chunk_count"] = int(len(part_rels))
        summary["max_chunk_size_mb"] = float(max_file_mb)
        summary["max_chunk_written_mb"] = float(max_written_map.get(base_name, 0.0))

        manifest_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        patched += 1

    return patched


def _scan_over_limit(root: Path, max_file_mb: float) -> List[Tuple[Path, float]]:
    limit = int(float(max_file_mb) * 1024.0 * 1024.0)
    out: List[Tuple[Path, float]] = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file():
            sz = int(p.stat().st_size)
            if sz > limit:
                out.append((p, sz / (1024.0 * 1024.0)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def main() -> None:
    args = _parse_args()
    scenarios_root = Path(args.scenarios_root)
    aggregates_dir = scenarios_root / "aggregates"
    if not aggregates_dir.exists():
        raise FileNotFoundError(f"Missing aggregates directory: {aggregates_dir}")

    parts_map: Dict[str, List[str]] = {}
    max_written_map: Dict[str, float] = {}

    for agg_path in sorted(aggregates_dir.glob("*.geojson")):
        part_names, max_written_mb = _split_geojson_if_needed(agg_path, float(args.max_file_mb))
        if part_names:
            parts_map[agg_path.name] = part_names
            max_written_map[agg_path.name] = float(max_written_mb)
            print(
                f"aggregate={agg_path.name} parts={len(part_names)} max_chunk_mb={max_written_mb:.3f}",
                flush=True,
            )

    patched = _patch_manifests(
        scenarios_root,
        parts_map,
        float(args.max_file_mb),
        max_written_map,
    )
    print(f"manifests_patched={patched}", flush=True)

    offenders = _scan_over_limit(scenarios_root, 100.0)
    print(f"over_100mb_count={len(offenders)}", flush=True)
    for p, mb in offenders[:200]:
        print(f"over_100mb file={p.as_posix()} size_mb={mb:.3f}", flush=True)


if __name__ == "__main__":
    main()
