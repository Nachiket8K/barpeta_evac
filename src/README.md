# `src/` package overview

This folder contains the core Python modules for the **Barpeta evacuation / flood exposure** project. The intent is to keep notebooks thin: notebooks should primarily orchestrate the workflow by calling functions defined here, while `src/` holds reusable, testable logic.

## Design conventions

- **Single responsibility**: each module owns one “layer” (config, I/O, roads, raster features, visualization).
- **Notebook parity**: functions reflect what already exists in the current notebook. More advanced routing/simulation logic will be added later.
- **CRS discipline**:
  - Plotting functions pick a CRS based on the chosen base layer (Contextily → EPSG:3857, GeoTIFF → raster CRS).
  - Feature engineering functions reproject internally as needed (e.g., densification for roads in a metric CRS; sampling rasters in raster CRS).
- **Outputs**:
  - Visualization functions optionally auto-save figures to `outputs/graphs/` using structured names with timestamps.

---

# `config.py`

Centralized imports and project-wide constants.

### What belongs here
- Common imports (NumPy, GeoPandas, Rasterio, OSMnx, Matplotlib, etc.)
- Categorical definitions (LULC classes and their colors/names)
- Constants shared across the codebase (month keys, nodata values, “wet” class values, drivable highway filters)

### Key constants
- `LULC_CLASSES`: list of `(class_id, hex_color, class_name)` used to render the GeoTIFF classification map.
- `WET_CLASS_VALUES`: set of class IDs considered “wet / hazardous” (currently `{0, 3}` = `water` + `flooded_vegetation`).
- `COMMON_NODATA_VALUES`: nodata sentinels seen in month rasters (`{15, 255}`).
- `MONTH_KEYS`: canonical month tokens used in column naming (currently `("jan", "may", "oct")`).
- `EVAC_DRIVABLE_HIGHWAYS`: set of OSM `highway` tags considered relevant for drivable evacuation routing.

---

# `io.py`

File locations and minimal I/O used by the notebook.

### What belongs here
- Canonical file path definitions for:
  - AOI shapefile
  - Building footprint shapefile
  - Monthly LULC GeoTIFFs
  - DEM GeoTIFF
- Minimal loaders/openers used in the notebook (no graph logic here)
- Snapshot saving for intermediate GeoDataFrames

### Functions
- `load_aoi(path=AOI_PATH) -> GeoDataFrame`
  - Reads the AOI shapefile.
- `load_buildings(path=BUILDINGS_PATH) -> GeoDataFrame`
  - Reads the building footprints shapefile.
- `open_lulc_month(month, paths=LULC_TIF_PATHS) -> rasterio.DatasetReader`
  - Opens a single month’s LULC GeoTIFF.
- `open_all_lulc(paths=LULC_TIF_PATHS) -> dict[str, DatasetReader]`
  - Opens all month rasters (caller should close them).
- `open_dem(path=DEM_TIF_PATH) -> DatasetReader`
  - Opens the DEM raster.
- `close_opened(srcs: dict) -> None`
  - Utility to close a dict of open raster datasets.
- `save_buildings_snapshot_parquet(gdf, out_path) -> Path`
  - Writes a GeoDataFrame snapshot as parquet (used as “checkpointing” during EDA / feature engineering).

---

# `roads.py`

Road network construction and storage utilities (OSMnx + NetworkX).

### What belongs here
- Downloading road graphs from OSM within a bounding box
- Filtering edges by `highway` tags
- Selecting largest connected component
- Writing graph data to disk (GPKG/GraphML)

### Functions
- `download_graph_from_bbox(bbox, network_type="drive", simplify=True, retain_all=True, timeout=180) -> nx.MultiDiGraph`
  - Downloads a graph from OSM using an explicit bounding box.
- `graph_to_gdfs(G) -> (nodes_gdf, edges_gdf)`
  - Converts an OSMnx graph into GeoDataFrames.
- `filter_graph_by_highway(G, keep_highways) -> nx.MultiDiGraph`
  - Edge-induced subgraph containing only selected `highway` classes.
- `largest_connected_component(G) -> nx.MultiDiGraph`
  - Extracts the LCC (connectivity computed on an undirected projection).
- `save_graph_gpkg(G, out_path, nodes_layer="nodes", edges_layer="edges") -> Path`
  - Saves nodes/edges to a GeoPackage (the format used in the notebook).
- `save_graphml(G, out_path) -> Path`
  - Saves graph to GraphML for later reuse / inspection.

---

# `raster_features.py`

Feature engineering: compute building/road attributes from rasters (no plotting).

### What belongs here
- Any computation that creates **new columns** on roads/buildings from rasters:
  - Road hazard probability (`p_hazard_<month>`)
  - Building water proximity (`water_prox_<month>`)
  - Mean building elevation (`elev_mean`)
  - Flood Susceptibility Index (`FSI_<month>`)
- Logic should be deterministic and testable; visualization belongs in `viz.py`.

### Functions

## Road hazard probability
- `add_raster_hazard_feature_one_month(gdf_edges, src, out_col, step_m=25.0, hazard_classes=WET_CLASS_VALUES, nodata_values=None) -> GeoDataFrame`
  - Densifies each road geometry (in a metric CRS) and samples raster values at interpolated points (in raster CRS).
  - Outputs `out_col` as the fraction of sampled points that fall into hazard classes (e.g., wet pixels).
  - **NaN output** typically indicates:
    - geometry empty/invalid, or
    - all sampled raster values were nodata/outside raster.

## Building water proximity
- `add_water_proximity_from_raster(gdf_buildings, src, month, wet_classes=WET_CLASS_VALUES, use_centroids=True, nodata_override=None) -> GeoDataFrame`
  - Computes a distance transform on the raster and samples distance-to-wet at each building (typically at centroid).
  - Adds `water_prox_<month>` column (meters).
  - For geographic rasters (EPSG:4326), uses an approximate meters-per-degree conversion at the raster’s mid-latitude.

## Building elevation mean (iterative padding)
- `add_elev_mean_iterative_padding(gdf_buildings, dem_src, out_col="elev_mean", nodata_override=0.0, all_touched=True, start_pad_px=0.5, step_pad_px=0.5, max_pad_px=5.0, verbose=True) -> GeoDataFrame`
  - Computes per-polygon mean elevation from the DEM.
  - If the polygon mask yields no valid pixels, the function retries with increasingly larger pixel padding around the polygon bounds (0.5px, 1.0px, 1.5px, …).
  - Intended to reduce NaNs caused by alignment/edge effects and narrow polygons.

## Flood Susceptibility Index (FSI)
- `add_flood_susceptibility_index(gdf_buildings, elev_col="elev_mean", months=("jan","may","oct"), prox_prefix="water_prox_", prox_cap_m=2000.0, elev_p_low=0.10, elev_p_high=0.90, w_prox=0.5, w_elev=0.5, out_prefix="FSI_") -> GeoDataFrame`
  - Creates `FSI_<month>` columns combining:
    - proximity risk (closer to wet pixels → higher risk)
    - elevation risk (lower elevation → higher risk)
  - Uses quantiles (p10/p90) to scale elevation robustly.
  - If proximity is NaN, the resulting FSI is NaN (consistent with the notebook logic).

---

# `viz.py`

All map rendering and charting utilities, with optional auto-saving.

### What belongs here
- Any Matplotlib/GeoPandas plotting
- Histogram generation
- Threshold visualizations for roads/buildings
- Optional background layers:
  - Contextily (web tiles, EPSG:3857)
  - Raster GeoTIFF background (raster CRS)

### Output behavior
Most functions accept:
- `save: bool = True`
- `out_dir: Path = Path("outputs/graphs")`
- `name: Optional[str] = None`
- `dpi: int = 200`

They return `(fig, ax, saved_path)` where `saved_path` is `None` if `save=False`.

### Functions

## Raster visualization
- `show_classified_raster(tif_path, classes=LULC_CLASSES, nodata=255, ...)`
  - Renders the classified raster using the class color legend and writes a labeled PNG.

## Roads over raster
- `plot_roads_over_tif(gdf_edges, src, hazard_col, ...)`
  - Renders the LULC raster as a base layer and overlays road segments colored by `hazard_col`.

## Generic histogram helper
- `plot_series_histogram(series, title, xlabel, ...)`
  - Produces a standard histogram and saves it.

## Road threshold map
- `plot_roads_threshold(gdf_edges, hazard_col, threshold=0.2, src=None, show_raster=True, aoi=None, ...)`
  - Classifies road segments as risky/safe based on `hazard_col` threshold.
  - Optional raster base layer and AOI overlay.

## Building threshold map: water proximity
- `plot_buildings_waterprox_threshold(gdf_buildings, month, threshold_m=250, base="contextily"|"raster"|"none", ...)`
  - Plots buildings colored by whether `water_prox_<month>` is below or above the threshold.
  - Optional AOI boundary overlay.

## Building threshold map: FSI
- `plot_buildings_fsi_threshold(gdf_buildings, month="may", threshold=0.5, base="contextily"|"raster"|"none", ...)`
  - Plots buildings categorized by `FSI_<month>`:
    - red: above threshold
    - green: below threshold
    - NaN: empty square marker with black border (as requested in the notebook)

---

# `routing.py`

Placeholder for upcoming work.

### Purpose (future)
- Shortest path queries and evacuation routing
- Accessibility scoring and facility reachability
- Scenario metrics (robustness, redundancy, bottlenecks)
- Integration with hazard constraints (edge penalties / closures) from `p_hazard_*`

Currently minimal by design while we finalize EDA + feature engineering parity with the notebook.
