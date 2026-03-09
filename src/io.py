"""
File sources and minimal I/O used by the notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .config import gpd, rasterio, MONTH_KEYS
import json
import geojson

AOI_PATH = Path("data/raw/BarpetaAOI/BarpetaAOI.shp")
AOI_PATH_ACTUAL = Path("data/processed/barpeta_aoi.geojson")
BUILDINGS_PATH = Path("data/raw/buildings/Buildings_barpeta.shp")

# LULC rasters per month
LULC_TIF_PATHS: Dict[str, Path] = {
    "jan": Path("data/raw/lulc/LULC_JAN2025.tif"),
    "may": Path("data/raw/lulc/LULC_MAY2025.tif"),
    "oct": Path("data/raw/lulc/LULC_OCT2025.tif"),
}

# DEM raster (OpenTopography export)
DEM_TIF_PATH = Path("data/raw/dem/barpeta_dem.tif")


def load_aoi(path: Path = AOI_PATH) -> gpd.GeoDataFrame:
    """Load AOI polygon(s)."""
    return gpd.read_file(path)
    
def load_aoi_full(path: Path = AOI_PATH_ACTUAL) -> gpd.GeoDataFrame:
    """Load AOI polygon(s)."""
    df = gpd.read_file(path)
    return df


def load_buildings(path: Path = BUILDINGS_PATH) -> gpd.GeoDataFrame:
    """Load building footprints."""
    return gpd.read_file(path)


def open_lulc_month(month: str, paths: Dict[str, Path] = LULC_TIF_PATHS):
    """Open a single LULC raster for a month (returns rasterio DatasetReader)."""
    m = month.strip().lower()
    if m not in paths:
        raise ValueError(f"Unknown month '{month}'. Expected one of {list(paths.keys())}.")
    return rasterio.open(paths[m])


def open_all_lulc(paths: Dict[str, Path] = LULC_TIF_PATHS):
    """Open all month rasters (returns dict of rasterio DatasetReader). Caller closes."""
    out = {}
    for m in MONTH_KEYS:
        if m not in paths:
            raise ValueError(f"Missing LULC path for month '{m}'.")
        out[m] = rasterio.open(paths[m])
    return out


def open_dem(path: Path = DEM_TIF_PATH):
    """Open DEM raster (returns rasterio DatasetReader)."""
    return rasterio.open(path)


def close_opened(srcs: Dict[str, object]) -> None:
    """Close a dict of opened rasterio datasets."""
    for s in srcs.values():
        try:
            s.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Snapshot saving used in notebook
# ---------------------------------------------------------------------

def save_buildings_snapshot_parquet(gdf_buildings, out_path: Path) -> Path:
    """
    Save buildings GeoDataFrame to parquet snapshot (as in notebook).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_buildings.to_parquet(out_path, index=False)
    return out_path
