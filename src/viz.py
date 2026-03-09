"""
Visualization utilities with optional auto-saving into outputs/.

Every function can save figures to disk:
- save=True (default): stores PNG in outputs/graphs with structured naming
- returns (fig, ax, saved_path)

No code execution here; these are module contents to import from notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .config import (np, gpd, plt, mcolors, Patch, ctx, LULC_CLASSES)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _slug(s: str) -> str:
    s = s.strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", ".", "/"):
            out.append("_")
    # collapse repeats
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


def _save_fig(
    fig,
    *,
    out_dir: Path,
    filename: str,
    dpi: int = 200,
    tight: bool = True,
) -> Path:
    _ensure_dir(out_dir)
    path = out_dir / filename
    if tight:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=dpi)
    return path


def _resolve_ctx_provider(provider_str: str):
    """
    Resolve a contextily provider string like "CartoDB.Positron" safely (no eval).
    """
    node = ctx.providers
    for part in provider_str.split("."):
        node = getattr(node, part)
    return node


# ---------------------------------------------------------------------
# Raster display (classified)
# ---------------------------------------------------------------------

def show_classified_raster(
    tif_path: str,
    *,
    classes: List[Tuple[int, str, str]] = LULC_CLASSES,
    nodata: int = 255,
    figsize=(10, 8),
    title: str = "Land Use / Land Cover (LULC)",
    save: bool = True,
    out_dir: Path = Path("outputs/graphs"),
    name: Optional[str] = None,
    dpi: int = 200,
):
    import rioxarray as rxr

    da = rxr.open_rasterio(tif_path).squeeze("band", drop=True)
    arr = da.values
    arr_masked = np.ma.masked_equal(arr, nodata)

    colors = [c[1] for c in classes]
    cmap = mcolors.ListedColormap(colors, name="lulc")
    bounds = np.arange(-0.5, 9.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(arr_masked, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.axis("off")

    legend_patches = [Patch(facecolor=hexcol, edgecolor="none", label=f"{val}: {name_}")
                      for val, hexcol, name_ in classes]
    ax.legend(handles=legend_patches, loc="lower left", frameon=True, title="Classes")

    saved_path = None
    if save:
        base = name or f"raster_classified_{_slug(Path(tif_path).stem)}"
        fn = f"{base}.png"
        saved_path = _save_fig(fig, out_dir=out_dir, filename=fn, dpi=dpi)

    return fig, ax, saved_path


# ---------------------------------------------------------------------
# Roads on raster background (categorical tif)
# ---------------------------------------------------------------------

CLASS_COLORS = {
    0: ("#419bdf", "water"),
    1: ("#397d49", "trees"),
    2: ("#88b053", "grass"),
    3: ("#7a87c6", "flooded_vegetation"),
    4: ("#e49635", "crops"),
    5: ("#dfc35a", "shrub_and_scrub"),
    6: ("#c4281b", "built"),
    7: ("#a59b8f", "bare"),
    8: ("#b39fe1", "snow_and_ice"),
}


def plot_roads_over_tif(
    gdf_edges: gpd.GeoDataFrame,
    src,
    *,
    hazard_col: str,
    figsize=(14, 10),
    raster_alpha: float = 1.0,
    roads_linewidth: float = 1.2,
    roads_alpha: float = 0.95,
    roads_cmap: str = "RdYlGn_r",   # green (low) -> red (high)
    vmin: float = 0.0,
    vmax: float = 1.0,
    show_raster_legend: bool = True,
    show_hazard_colorbar: bool = True,
    title: Optional[str] = None,
    save: bool = True,
    out_dir: Path = Path("outputs/graphs"),
    name: Optional[str] = None,
    dpi: int = 200,
):
    if gdf_edges.crs is None:
        raise ValueError("gdf_edges.crs is None.")
    if src.crs is None:
        raise ValueError("Raster src.crs is None.")
    if hazard_col not in gdf_edges.columns:
        raise KeyError(f"Missing hazard column '{hazard_col}'")

    edges = gdf_edges.to_crs(src.crs)

    arr = src.read(1).astype(np.int32)
    nodata = src.nodata
    if nodata is not None:
        arr = np.ma.masked_where(arr == int(nodata), arr)

    colors = [CLASS_COLORS[k][0] for k in sorted(CLASS_COLORS)]
    cmap_r = mcolors.ListedColormap(colors, name="lulc_fixed")
    bounds = np.arange(-0.5, 9.5, 1.0)
    norm_r = mcolors.BoundaryNorm(bounds, cmap_r.N)

    fig, ax = plt.subplots(figsize=figsize)

    left, bottom, right, top = src.bounds
    ax.imshow(
        arr,
        extent=(left, right, bottom, top),
        origin="upper",
        alpha=raster_alpha,
        cmap=cmap_r,
        norm=norm_r,
        interpolation="nearest",
    )

    known = edges[edges[hazard_col].notna()]
    unknown = edges[edges[hazard_col].isna()]

    if len(unknown) > 0:
        unknown.plot(ax=ax, color="lightgrey", linewidth=roads_linewidth, alpha=roads_alpha)

    if len(known) > 0:
        known.plot(
            ax=ax,
            column=hazard_col,
            cmap=roads_cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=roads_linewidth,
            alpha=roads_alpha,
            legend=show_hazard_colorbar,
        )

    if show_raster_legend:
        patches = [Patch(facecolor=CLASS_COLORS[k][0], edgecolor="none", label=f"{k}: {CLASS_COLORS[k][1]}")
                   for k in sorted(CLASS_COLORS)]
        ax.legend(handles=patches, loc="lower left", frameon=True, title="Raster classes")

    ax.set_axis_off()
    ax.set_title(title or f"Road hazard overlay ({hazard_col})")

    saved_path = None
    if save:
        base = name or f"roads_over_tif__{_slug(hazard_col)}"
        fn = f"{base}.png"
        saved_path = _save_fig(fig, out_dir=out_dir, filename=fn, dpi=dpi)

    return fig, ax, saved_path


# ---------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------

def plot_series_histogram(
    series,
    *,
    title: str,
    xlabel: str,
    bins: int = 40,
    xlim=None,
    log_y: bool = False,
    save: bool = True,
    out_dir: Path = Path("outputs/graphs"),
    name: Optional[str] = None,
    dpi: int = 200,
):
    s = series.dropna()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(s.values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if xlim is not None:
        ax.set_xlim(xlim)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    saved_path = None
    if save:
        base = name or f"hist_{_slug(title)[:60]}"
        fn = f"{base}.png"
        saved_path = _save_fig(fig, out_dir=out_dir, filename=fn, dpi=dpi)

    return fig, ax, saved_path


# ---------------------------------------------------------------------
# Threshold maps: roads
# ---------------------------------------------------------------------

def plot_roads_threshold(
    gdf_edges: gpd.GeoDataFrame,
    *,
    hazard_col: str,
    threshold: float = 0.2,
    src=None,                      # optional raster background
    show_raster: bool = True,
    aoi: Optional[gpd.GeoDataFrame] = None,
    figsize=(14, 10),
    roads_linewidth: float = 1.4,
    roads_alpha: float = 0.95,
    title: Optional[str] = None,
    save: bool = True,
    out_dir: Path = Path("outputs/graphs"),
    name: Optional[str] = None,
    dpi: int = 200,
):
    if hazard_col not in gdf_edges.columns:
        raise KeyError(f"Column '{hazard_col}' not found in gdf_edges.")
    if gdf_edges.crs is None:
        raise ValueError("gdf_edges.crs is None.")
    if show_raster and src is None:
        raise ValueError("show_raster=True requires src (opened GeoTIFF).")

    plot_crs = src.crs if (src is not None) else gdf_edges.crs
    edges = gdf_edges.to_crs(plot_crs)

    risky = edges[edges[hazard_col].notna() & (edges[hazard_col] >= threshold)]
    safe = edges[edges[hazard_col].notna() & (edges[hazard_col] < threshold)]
    unknown = edges[edges[hazard_col].isna()]

    fig, ax = plt.subplots(figsize=figsize)

    if show_raster and src is not None:
        arr = src.read(1).astype(np.int32)
        nodata = src.nodata
        if nodata is not None:
            arr = np.ma.masked_where(arr == int(nodata), arr)
        left, bottom, right, top = src.bounds
        ax.imshow(arr, extent=(left, right, bottom, top), origin="upper", interpolation="nearest")

    if len(unknown) > 0:
        unknown.plot(ax=ax, color="lightgrey", linewidth=roads_linewidth, alpha=roads_alpha)
    if len(safe) > 0:
        safe.plot(ax=ax, color="green", linewidth=roads_linewidth, alpha=roads_alpha)
    if len(risky) > 0:
        risky.plot(ax=ax, color="red", linewidth=roads_linewidth, alpha=roads_alpha)

    if aoi is not None:
        aoi_plot = aoi.to_crs(plot_crs)
        aoi_plot.boundary.plot(ax=ax, linewidth=2.0, color="black")
        minx, miny, maxx, maxy = aoi_plot.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    ax.set_axis_off()
    ax.set_title(title or f"Roads thresholded by {hazard_col} (threshold={threshold})")

    patches = [
        Patch(facecolor="red", edgecolor="none", label=f"{hazard_col} ≥ {threshold}"),
        Patch(facecolor="green", edgecolor="none", label=f"{hazard_col} < {threshold}"),
        Patch(facecolor="lightgrey", edgecolor="none", label="NaN / unknown"),
    ]
    ax.legend(handles=patches, loc="upper right", frameon=True, title="Road classes")

    saved_path = None
    if save:
        base = name or f"roads_threshold__{_slug(hazard_col)}__thr_{threshold}"
        fn = f"{base}.png"
        saved_path = _save_fig(fig, out_dir=out_dir, filename=fn, dpi=dpi)

    return fig, ax, saved_path


# ---------------------------------------------------------------------
# Threshold maps: buildings (water proximity and FSI)
# ---------------------------------------------------------------------

def plot_buildings_waterprox_threshold(
    gdf_buildings: gpd.GeoDataFrame,
    *,
    month: str,
    threshold_m: float = 250.0,
    aoi: Optional[gpd.GeoDataFrame] = None,
    base: str = "contextily",  # "contextily" | "raster" | "none"
    src=None,
    use_centroids: bool = True,
    figsize=(14, 10),
    point_size: float = 6.0,
    alpha: float = 0.9,
    contextily_provider: str = "CartoDB.Positron",
    title: Optional[str] = None,
    save: bool = True,
    out_dir: Path = Path("outputs/graphs"),
    name: Optional[str] = None,
    dpi: int = 200,
):
    m = month.strip().lower()
    col = f"water_prox_{m}"
    if col not in gdf_buildings.columns:
        raise KeyError(f"Column '{col}' not found in gdf_buildings.")
    if gdf_buildings.crs is None:
        raise ValueError("gdf_buildings.crs is None.")

    base = base.strip().lower()
    if base not in {"contextily", "raster", "none"}:
        raise ValueError("base must be one of: 'contextily', 'raster', 'none'.")
    if base == "raster" and src is None:
        raise ValueError("base='raster' requires src.")

    if base == "contextily":
        plot_crs = "EPSG:3857"
    elif base == "raster":
        plot_crs = src.crs
    else:
        plot_crs = gdf_buildings.crs

    if use_centroids:
        b_utm = gdf_buildings.to_crs(gdf_buildings.estimate_utm_crs())
        pts = gpd.GeoSeries(b_utm.centroid, crs=b_utm.crs).to_crs(plot_crs)
        gdf_pts = gpd.GeoDataFrame(gdf_buildings.drop(columns="geometry"), geometry=pts, crs=plot_crs)
    else:
        gdf_pts = gdf_buildings.to_crs(plot_crs).copy()

    s = gdf_pts[col].astype(float)
    near = gdf_pts[s.notna() & (s < threshold_m)]
    far = gdf_pts[s.notna() & (s >= threshold_m)]
    unk = gdf_pts[s.isna()]

    fig, ax = plt.subplots(figsize=figsize)

    if base == "raster":
        arr = src.read(1).astype(np.int32)
        nodata = src.nodata
        if nodata is not None:
            arr = np.ma.masked_where(arr == int(nodata), arr)
        left, bottom, right, top = src.bounds
        ax.imshow(arr, extent=(left, right, bottom, top), origin="upper", interpolation="nearest")

    if len(unk) > 0:
        unk.plot(ax=ax, color="lightgrey", markersize=point_size, alpha=alpha)
    if len(far) > 0:
        far.plot(ax=ax, color="green", markersize=point_size, alpha=alpha)
    if len(near) > 0:
        near.plot(ax=ax, color="red", markersize=point_size, alpha=alpha)

    if aoi is not None:
        aoi_plot = aoi.to_crs(plot_crs)
        aoi_plot.boundary.plot(ax=ax, linewidth=2.0, color="black")
        minx, miny, maxx, maxy = aoi_plot.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    if base == "contextily":
        provider = _resolve_ctx_provider(contextily_provider)
        ctx.add_basemap(ax, source=provider)

    ax.set_axis_off()
    ax.set_title(title or f"Buildings by {col} threshold={threshold_m:.0f}m (base={base})")

    patches = [
        Patch(facecolor="red", edgecolor="none", label=f"< {threshold_m:.0f}m"),
        Patch(facecolor="green", edgecolor="none", label=f"≥ {threshold_m:.0f}m"),
        Patch(facecolor="lightgrey", edgecolor="none", label="NaN / unknown"),
    ]
    ax.legend(handles=patches, loc="upper right", frameon=True, title="Building classes")

    saved_path = None
    if save:
        base_name = name or f"bldg_waterprox__{m}__thr_{int(threshold_m)}m__base_{base}"
        fn = f"{_slug(base_name)}.png"
        saved_path = _save_fig(fig, out_dir=out_dir, filename=fn, dpi=dpi)

    return fig, ax, saved_path


def plot_buildings_fsi_threshold(
    gdf_buildings: gpd.GeoDataFrame,
    *,
    month: str = "may",
    threshold: float = 0.5,
    fsi_prefix: str = "FSI_",
    aoi: Optional[gpd.GeoDataFrame] = None,
    base: str = "none",
    src=None,
    use_centroids: bool = True,
    figsize=(14, 10),
    point_size: float = 6.0,
    alpha: float = 0.9,
    contextily_provider: str = "CartoDB.Positron",
    title: Optional[str] = None,
    save: bool = True,
    out_dir: Path = Path("outputs/graphs"),
    name: Optional[str] = None,
    dpi: int = 200,
):
    m = month.strip().lower()
    col = f"{fsi_prefix}{m}"
    if col not in gdf_buildings.columns:
        raise KeyError(f"Column '{col}' not found in gdf_buildings.")
    if gdf_buildings.crs is None:
        raise ValueError("gdf_buildings.crs is None.")

    base = base.strip().lower()
    if base not in {"contextily", "raster", "none"}:
        raise ValueError("base must be one of: 'contextily', 'raster', 'none'.")
    if base == "raster" and src is None:
        raise ValueError("base='raster' requires src.")

    if base == "contextily":
        plot_crs = "EPSG:3857"
    elif base == "raster":
        plot_crs = src.crs
    else:
        plot_crs = gdf_buildings.crs

    if use_centroids:
        b_utm = gdf_buildings.to_crs(gdf_buildings.estimate_utm_crs())
        pts = gpd.GeoSeries(b_utm.centroid, crs=b_utm.crs).to_crs(plot_crs)
        gdf_pts = gpd.GeoDataFrame(gdf_buildings.drop(columns="geometry"), geometry=pts, crs=plot_crs)
    else:
        gdf_pts = gdf_buildings.to_crs(plot_crs).copy()

    s = gdf_pts[col].astype(float)
    high = gdf_pts[s.notna() & (s > threshold)]
    low = gdf_pts[s.notna() & (s <= threshold)]
    nan = gdf_pts[s.isna()]

    fig, ax = plt.subplots(figsize=figsize)

    if base == "raster":
        arr = src.read(1).astype(np.int32)
        nodata = src.nodata
        if nodata is not None:
            arr = np.ma.masked_where(arr == int(nodata), arr)
        left, bottom, right, top = src.bounds
        ax.imshow(arr, extent=(left, right, bottom, top), origin="upper", interpolation="nearest")

    if len(low) > 0:
        low.plot(ax=ax, color="green", markersize=point_size, alpha=alpha, marker="o")
    if len(high) > 0:
        high.plot(ax=ax, color="red", markersize=point_size, alpha=alpha, marker="o")
    if len(nan) > 0:
        ax.scatter(
            nan.geometry.x, nan.geometry.y,
            s=point_size, marker="s",
            facecolors="none", edgecolors="black",
            linewidths=0.8, alpha=1.0
        )

    if aoi is not None:
        aoi_plot = aoi.to_crs(plot_crs)
        aoi_plot.boundary.plot(ax=ax, linewidth=2.0, color="black")
        minx, miny, maxx, maxy = aoi_plot.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    if base == "contextily":
        provider = _resolve_ctx_provider(contextily_provider)
        ctx.add_basemap(ax, source=provider)

    ax.set_axis_off()
    ax.set_title(title or f"Buildings by {col} threshold={threshold} (base={base})")

    patches = [
        Patch(facecolor="red", edgecolor="none", label=f"{col} > {threshold}"),
        Patch(facecolor="green", edgecolor="none", label=f"{col} ≤ {threshold}"),
        Patch(facecolor="white", edgecolor="black", label=f"{col} is NaN"),
    ]
    ax.legend(handles=patches, loc="upper right", frameon=True, title="FSI classes")

    saved_path = None
    if save:
        base_name = name or f"bldg_fsi__{m}__thr_{threshold}__base_{base}"
        fn = f"{_slug(base_name)}.png"
        saved_path = _save_fig(fig, out_dir=out_dir, filename=fn, dpi=dpi)

    return fig, ax, saved_path
