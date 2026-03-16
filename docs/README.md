# Barpeta Flooding — Static Scenario Assets

This folder stores precomputed artifacts for GitHub Pages playback.

- `scenarios/index.json`: scenario catalog for the web UI
- `scenarios/<scenario_id>/manifest.json`: per-scenario metadata
- `scenarios/<scenario_id>/frames/water_mask/t####.png`: weekly transparent flood masks
- scenario CSV/JSON details (failed edges, evac edges, routes, building accessibility)

Run `notebooks/05_barpeta_rebuild_weekly_scenarios.ipynb` to regenerate these artifacts.
