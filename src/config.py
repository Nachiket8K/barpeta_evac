"""
Global imports and project-wide constants.
"""

from __future__ import annotations

# ---- Standard libraries ----
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---- Core geospatial stack ----
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.ndimage import distance_transform_edt


# ---- Geometry / GIS ----
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union

# ---- Raster ----
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import from_bounds, Window
from rasterio.features import geometry_mask

# ---- OSM / graphs ----
import osmnx as ox
import networkx as nx

# ---- Plotting ----
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
import contextily as ctx
import rioxarray as rxr

# Categorical legend used in LULC .tif data

LULC_CLASSES: List[Tuple[int, str, str]] = [
    (0, "#419bdf", "water"),
    (1, "#397d49", "trees"),
    (2, "#88b053", "grass"),
    (3, "#7a87c6", "flooded_vegetation"),
    (4, "#e49635", "crops"),
    (5, "#dfc35a", "shrub_and_scrub"),
    (6, "#c4281b", "built"),
    (7, "#a59b8f", "bare"),
    (8, "#b39fe1", "snow_and_ice"),
]

# “Wet / hazardous” classes (as used throughout the notebook)
WET_CLASS_VALUES: Set[int] = {0, 3}  # water + flooded_vegetation

COMMON_NODATA_VALUES: Set[int] = {15, 255}
DEFAULT_LULC_NODATA: int = 255
DEFAULT_LULC_NODATA_ALT: int = 15
DEFAULT_DEM_NODATA: float = 0.0

MONTH_KEYS: Tuple[str, ...] = ("jan", "may", "oct")

# Rebuild notebook defaults (May -> Oct weekly animation + scenario tuning)
WEEKLY_START_DATE: str = "2025-05-01"
WEEKLY_END_DATE: str = "2025-10-31"
DEFAULT_ACCESS_RADIUS_M: float = 1000.0
DEFAULT_SCENARIO_SEEDS: Tuple[int, ...] = (7, 42, 101, 202, 404)

EVAC_DRIVABLE_HIGHWAYS: Set[str] = {
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    # Extend later if needed:
    # "unclassified", "residential", "living_street", "service", "track", "road",
}
