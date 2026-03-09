"""
Feature engineering from rasters (no plotting).

Contains the notebook’s core computations:
- p_hazard_<month> for road edges (fraction of sampled points that are wet)
- water_prox_<month> for buildings (distance to nearest wet pixel)
- elev_mean for buildings (polygon mean from DEM, iterative padding)
- FSI_<month> from elevation + proximity
"""

from __future__ import annotations

from typing import Iterable, Optional, Set

from .config import (
    np, gpd, rowcol, distance_transform_edt,
    WET_CLASS_VALUES, DEFAULT_DEM_NODATA,
)


# ---------------------------------------------------------------------
# Roads: p_hazard for one month (no merges; writes directly to gdf_edges)
# ---------------------------------------------------------------------

def _densify_linestring(line, step_m: float):
    """Interpolate points along a LineString in its current CRS (must be meters)."""
    if line is None or line.is_empty:
        return []
    length = line.length
    if length == 0:
        return [line.interpolate(0)]
    distances = np.arange(0, length + step_m, step_m)
    return [line.interpolate(d) for d in distances]


def add_raster_hazard_feature_one_month(
    gdf_edges: gpd.GeoDataFrame,
    src,
    *,
    out_col: str,
    step_m: float = 25.0,
    hazard_classes: Optional[Set[int]] = None,
    nodata_values: Optional[Iterable[int]] = None,
) -> gpd.GeoDataFrame:
    """
    Adds a single-month hazard feature to gdf_edges:
      out_col = fraction of sampled raster values that are in hazard_classes.

    Gray/NaN results occur when all sampled values are nodata or the edge is empty.
    """
    if hazard_classes is None:
        hazard_classes = set(WET_CLASS_VALUES)

    # nodata sentinels
    if nodata_values is None:
        nodata_values = set()
        if src.nodata is not None:
            nodata_values.add(int(src.nodata))
    else:
        nodata_values = set(int(x) for x in nodata_values)

    if gdf_edges.crs is None:
        raise ValueError("gdf_edges.crs is None.")
    if src.crs is None:
        raise ValueError("Raster src.crs is None.")

    utm_crs = gdf_edges.estimate_utm_crs()
    edges_m = gdf_edges.to_crs(utm_crs)

    p_list = []

    for geom_m in edges_m.geometry:
        if geom_m is None or geom_m.is_empty:
            p_list.append(np.nan)
            continue

        length = geom_m.length
        dists = np.arange(0, length + step_m, step_m) if length > 0 else np.array([0.0])
        pts_m = [geom_m.interpolate(d) for d in dists]

        pts = gpd.GeoSeries(pts_m, crs=utm_crs).to_crs(src.crs)
        coords = [(p.x, p.y) for p in pts]

        vals = np.array([v[0] for v in src.sample(coords)], dtype=float)

        # drop nodata + NaN
        if len(nodata_values) > 0:
            for nd in nodata_values:
                vals[vals == nd] = np.nan
        vals = vals[~np.isnan(vals)]

        if vals.size == 0:
            p_list.append(np.nan)
            continue

        vi = vals.astype(int)
        p_list.append(float(np.mean(np.isin(vi, list(hazard_classes)))))

    out = gdf_edges.copy()
    out[out_col] = p_list
    return out


# ---------------------------------------------------------------------
# Buildings: water proximity to nearest wet pixel (for the given month)
# ---------------------------------------------------------------------

def _approx_meters_per_degree(lat_deg: float):
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat_deg))
    return meters_per_deg_lon, meters_per_deg_lat


def add_water_proximity_from_raster(
    gdf_buildings: gpd.GeoDataFrame,
    src,
    *,
    month: str,
    wet_classes: Optional[Set[int]] = None,
    use_centroids: bool = True,
    nodata_override=None,
) -> gpd.GeoDataFrame:
    """
    Adds water_prox_<month> = distance (meters) to nearest wet pixel.
    Uses scipy.ndimage.distance_transform_edt on (~wet) mask.
    """
    if wet_classes is None:
        wet_classes = set(WET_CLASS_VALUES)

    if gdf_buildings.crs is None:
        raise ValueError("gdf_buildings.crs is None.")
    if src.crs is None:
        raise ValueError("Raster src.crs is None.")

    out_col = f"water_prox_{month.strip().lower()}"

    # building points to evaluate
    if use_centroids:
        b_utm = gdf_buildings.to_crs(gdf_buildings.estimate_utm_crs())
        pts = b_utm.centroid.to_crs(src.crs)
    else:
        pts = gdf_buildings.to_crs(src.crs).geometry

    arr = src.read(1).astype(np.int32)

    nodata = src.nodata if src.nodata is not None else None
    if nodata_override is not None:
        nodata = nodata_override

    wet = np.isin(arr, list(wet_classes))
    if nodata is not None:
        wet = wet & (arr != int(nodata))

    # EDT sampling (meters)
    transform = src.transform
    if src.crs.is_geographic:
        left, bottom, right, top = src.bounds
        lat0 = (bottom + top) / 2.0
        m_lon, m_lat = _approx_meters_per_degree(lat0)
        xres_deg = abs(transform.a)
        yres_deg = abs(transform.e)
        sampling = (yres_deg * m_lat, xres_deg * m_lon)  # (row, col)
    else:
        xres = abs(transform.a)
        yres = abs(transform.e)
        sampling = (yres, xres)

    dist = distance_transform_edt(~wet, sampling=sampling).astype(np.float32)

    height, width = dist.shape
    distances = []

    for p in pts:
        if p is None or p.is_empty:
            distances.append(np.nan)
            continue
        r, c = rowcol(transform, p.x, p.y)
        if r < 0 or r >= height or c < 0 or c >= width:
            distances.append(np.nan)
        else:
            distances.append(float(dist[r, c]))

    out = gdf_buildings.copy()
    out[out_col] = distances
    return out


# ---------------------------------------------------------------------
# Buildings: elevation mean from DEM (polygon mean + iterative padding)
# ---------------------------------------------------------------------

def _polygon_mean_elevation_subset(
    gdf_subset: gpd.GeoDataFrame,
    dem_src,
    *,
    pad_pixels: float,
    nodata_override: Optional[float],
    all_touched: bool,
):
    from .config import from_bounds, Window, geometry_mask  # centralized imports

    full = Window(0, 0, dem_src.width, dem_src.height)
    xres = abs(dem_src.transform.a)
    yres = abs(dem_src.transform.e)
    pad_x = pad_pixels * xres
    pad_y = pad_pixels * yres

    nodata = dem_src.nodata if dem_src.nodata is not None else None
    if nodata_override is not None:
        nodata = nodata_override

    means = np.full(len(gdf_subset), np.nan, dtype=np.float32)

    for i, geom in enumerate(gdf_subset.geometry):
        if geom is None or geom.is_empty:
            continue

        minx, miny, maxx, maxy = geom.bounds
        minx -= pad_x; maxx += pad_x
        miny -= pad_y; maxy += pad_y

        w = from_bounds(minx, miny, maxx, maxy, transform=dem_src.transform)
        try:
            w = w.intersection(full)
        except Exception:
            continue

        w = w.round_offsets(op="floor").round_lengths(op="ceil")
        if w.width < 1 or w.height < 1:
            continue

        arr = dem_src.read(1, window=w).astype(np.float32)
        transform = dem_src.window_transform(w)

        inside = geometry_mask(
            [geom],
            out_shape=arr.shape,
            transform=transform,
            invert=True,
            all_touched=all_touched,
        )

        vals = arr[inside]
        if vals.size == 0:
            continue

        if nodata is not None:
            vals = vals[vals != float(nodata)]

        if vals.size:
            means[i] = float(vals.mean())

    return means


def add_elev_mean_iterative_padding(
    gdf_buildings: gpd.GeoDataFrame,
    dem_src,
    *,
    out_col: str = "elev_mean",
    nodata_override: float = DEFAULT_DEM_NODATA,
    all_touched: bool = True,
    start_pad_px: float = 0.5,
    step_pad_px: float = 0.5,
    max_pad_px: float = 5.0,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Computes elev_mean with iterative padding:
      0.5px -> 1.0px -> 1.5px -> ... -> max_pad_px
    Only retries polygons still NaN after each pass.
    """
    if gdf_buildings.crs is None:
        raise ValueError("gdf_buildings.crs is None.")
    if dem_src.crs is None:
        raise ValueError("DEM CRS is None.")

    g = gdf_buildings.to_crs(dem_src.crs).copy()
    g[out_col] = np.nan

    pads = np.arange(start_pad_px, max_pad_px + 1e-9, step_pad_px)

    for pad in pads:
        nan_mask = g[out_col].isna().to_numpy()
        remaining = int(nan_mask.sum())
        if remaining == 0:
            break

        if verbose:
            print(f"[elev] pad={pad:.2f}px, remaining NaNs: {remaining}")

        sub = g.loc[nan_mask, ["geometry"]].copy()
        means = _polygon_mean_elevation_subset(
            sub,
            dem_src,
            pad_pixels=float(pad),
            nodata_override=nodata_override,
            all_touched=all_touched,
        )

        success = ~np.isnan(means)
        n_success = int(success.sum())
        if n_success:
            g.loc[sub.index[success], out_col] = means[success]

        if verbose:
            print(f"[elev] pad={pad:.2f}px, newly filled: {n_success}")

    return g.to_crs(gdf_buildings.crs)


# ---------------------------------------------------------------------
# Buildings: Flood Susceptibility Index (FSI) from elevation + proximity
# ---------------------------------------------------------------------

def add_flood_susceptibility_index(
    gdf_buildings,
    *,
    elev_col: str = "elev_mean",
    months=("jan", "may", "oct"),
    prox_prefix: str = "water_prox_",
    prox_cap_m: float = 2000.0,
    elev_p_low: float = 0.10,
    elev_p_high: float = 0.90,
    w_prox: float = 0.5,
    w_elev: float = 0.5,
    out_prefix: str = "FSI_",
):
    """
    Adds FSI_<month> columns based on elevation and water proximity.

    FSI = w_prox * prox_risk + w_elev * elev_risk
      prox_risk = 1 - clip(water_prox / prox_cap_m, 0, 1)
      elev_risk = 1 - clip((elev - p10) / (p90 - p10), 0, 1)

    If proximity is NaN, FSI stays NaN (matches your notebook logic).
    """
    out = gdf_buildings.copy()

    elev = out[elev_col].astype(float)
    p_lo = float(elev.quantile(elev_p_low))
    p_hi = float(elev.quantile(elev_p_high))
    denom = max(p_hi - p_lo, 1e-6)

    elev_scaled = np.clip((elev - p_lo) / denom, 0, 1)  # 0 low, 1 high
    elev_risk = 1.0 - elev_scaled

    for m in months:
        prox_col = f"{prox_prefix}{m}"
        if prox_col not in out.columns:
            raise KeyError(f"Missing proximity column: {prox_col}")

        prox = out[prox_col].astype(float)
        prox_scaled = np.clip(prox / prox_cap_m, 0, 1)  # 0 near, 1 far
        prox_risk = 1.0 - prox_scaled

        fsi = w_prox * prox_risk + w_elev * elev_risk
        fsi = fsi.where(~prox.isna(), np.nan)

        out[f"{out_prefix}{m}"] = fsi

    return out
