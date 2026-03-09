# Notebooks — Barpeta Evacuation / Flood Exposure Project

This folder contains the project notebooks in a staged workflow. The notebooks are intentionally thin: reusable logic is contained in `src/`, while notebooks orchestrate data loading, explanations of model choices, feature generation, and visualization.

## Directory conventions

- **Inputs**
  - `data/` contains raw/processed datasets (often gitignored if large).
- **Code**
  - `src/` contains reusable modules (I/O, raster feature engineering, road graph utilities, visualization helpers).
- **Outputs**
  - `outputs/graphs/` stores generated plots and maps.
  - `outputs/tables/` stores intermediate feature tables (parquet, gpkg, graph files).

All notebooks are designed to save figures to `outputs/graphs/` by default (through `src/viz.py` helpers).

---

## Notebook 01 — `01_explore.ipynb` (EDA)

Purpose: Establish baseline exploratory analysis and validate data integrity.

What it does:
- Loads **AOI** and **building footprints**.
- Opens **monthly LULC rasters** and prints:
  - CRS, bounds, NoData, unique values, nodata counts.
- Builds an **AOI polygon from raster bounds** (useful when OSM roads extend beyond the district boundary used in the satellite pipeline).
- Downloads a road graph (from raster bounds), filters to evac-relevant road types, extracts LCC, and saves graph checkpoints.
- Engineers and visualizes:
  - Road `p_hazard_<month>` (line sampling on wet classes) + histograms + threshold maps
  - Building `water_prox_<month>` + histograms + threshold maps
  - Building `elev_mean` (from DEM with iterative padding) + histogram
  - Building `FSI_<month>` + histograms + threshold maps
- Saves a **buildings feature snapshot**:
  - `outputs/tables/buildings_features_snapshot.parquet`

Primary outputs:
- Figures: `outputs/graphs/` (multiple, auto-named)
- Tables: `outputs/tables/buildings_features_snapshot.parquet`
- Graph checkpoints: `outputs/tables/roads_drive_lcc.gpkg`, `outputs/tables/roads_drive_lcc.graphml`

---

## Notebook 02 — `02_build_network.ipynb` (Network + Road Feature Engineering)

Purpose: Build a clean road network dataset and compute road hazard features per month.

What it does:
- Opens monthly LULC rasters and uses raster bounds to define the road download bbox.
- Downloads road graph and prepares:
  - evac-drivable filter
  - largest connected component (LCC)
  - node/edge GeoDataFrames
  - graph checkpoints (GPKG + GraphML)
- Computes road hazard features:
  - `p_hazard_<month>`: line-sampled wet class fraction (fast / scalable)
  - `p_buffer_hazard_<month>`: wet pixel fraction inside a buffer around each edge (slower; supports `MAX_EDGES` for testing)
- Saves **roads edge feature snapshots**:
  - `outputs/tables/roads_edges_features.parquet`
  - `outputs/tables/roads_edges_features.gpkg`

Primary outputs:
- Figures: `outputs/graphs/` (maps + histograms)
- Tables: `outputs/tables/roads_edges_features.parquet` (+ optional gpkg)
- Graph checkpoints: `outputs/tables/roads_drive_lcc.gpkg`, `outputs/tables/roads_drive_lcc.graphml`

---

## Notebook 03 — `03_integrated_diagnostics.ipynb` 

Purpose: Validate how datasets connect and ensure features behave as expected.

Key concept:
- **Buildings** connect to **LULC + DEM**
- **Road edges** connect to **LULC only**

What it does:
- Loads the feature checkpoints created by notebooks 01 and 02:
  - buildings: `outputs/tables/buildings_features_snapshot.parquet`
  - roads: `outputs/tables/roads_edges_features.parquet`
- Produces diagnostic plots (saved to `outputs/graphs/03_integrated/`):
  - Buildings:
    - monthly histograms (`water_prox_<month>`, `FSI_<month>`, `elev_mean`)
    - cross-month boxplots
    - bivariate scatter diagnostics:
      - elevation vs water proximity
      - elevation vs FSI
      - water proximity vs FSI
    - spatial threshold maps (raster baselayer + AOI):
      - water proximity threshold (250 m)
      - FSI threshold (0.5)
  - Roads:
    - monthly histograms and cross-month boxplots for `p_hazard_<month>` and `p_buffer_hazard_<month>`
    - scatter comparison: `p_hazard_<month>` vs `p_buffer_hazard_<month>`
    - optional threshold maps (`thr=0.2`) for quick spatial validation

Primary outputs:
- Figures: `outputs/graphs/03_integrated/`
- Optional refreshed snapshots available in the notebook

---
## Notebook 04 — `04_simulation_outline.ipynb` (Simulation Blueprint / Pseudo-code)

Purpose: Provide a **rough template** for the evacuation simulation design before implementing a full agent-based or discrete-event model.

Key concept:
- **Road network damage** is sampled stochastically from raster-derived hazard features.
- **Trucks** compute viable routes from a **city source** to **exit nodes** under damage.
- **People** are assigned to **buildings**, and an **accessibility score** is computed based on distance to the nearest computed `evac_path`.

What it does:
- Defines the intended simulation components and data contracts:
  - Inputs:
    - road graph `G` (NetworkX) with edge geometry/length and hazard columns (`p_hazard_<month>`, `p_buffer_hazard_<month>`)
    - buildings GeoDataFrame with polygon geometries (and optional population proxy)
  - Outputs (planned):
    - per-run metrics (e.g., access fraction within a radius, mean/median distance)
    - optional per-building distance tables
    - saved maps and distributions for scenario comparisons
- Provides pseudo-code for:
  - selecting a **source node** near Barpeta city center
  - choosing **exit nodes** (boundary exits or directional exits)
  - computing per-edge **failure probabilities**:
    - “over water” → deterministic failure
    - buffer wetness → probabilistic failure using `p_buffer_hazard_<month>`
    - fallback baseline failure for unknowns
  - applying damage to create a **damaged routing graph**
  - generating truck routes (`evac_path`) under damage
  - sampling people into buildings (uniform or weighted)
  - computing accessibility: distance from building centroids to nearest `evac_path`
  - running a Monte Carlo loop across many realizations/months

Primary outputs:
- This notebook is primarily **documentation + pseudo-code** (not a runnable simulation yet).
- Intended future outputs:
  - Tables: `outputs/tables/sim_results_<month>.parquet`
  - Figures: distributions of accessibility metrics and maps of damaged edges vs evac paths

---

## Recommended run order

1. `01_explore.ipynb`
2. `02_build_network.ipynb`
3. `03_data_analysis.ipynb`
3. `04_simulation_outline.ipynb`


---

## Troubleshooting notes

- **Large data volume**: building footprints can be very large; sampling was used for early stages. Current version is computationally intensive.
- **NaNs in raster-derived features**:
  - Often caused by geometries outside raster bounds or sampling only NoData pixels.
  - For elevation, edge cases can occur due to raster window rounding; iterative padding is used to reduce NaNs.

---
