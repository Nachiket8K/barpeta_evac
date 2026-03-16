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
