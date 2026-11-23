# Changelog

## Unreleased

- `Spatial` and `Local` follow the local-projection method of Palmer et al. (2020): `num_spatial_members` members are drawn once from the full global ensemble (an equal share per climate trajectory) and shared by every component; each member draws its GRD, CMIP and GIA patterns independently of its global member, with one GRD model across all fingerprints; and `total_rsl` is summed member by member before percentiles are taken [#119](https://github.com/MetOffice/ProFSea-tool/pull/119)
- Spatial components return a `SeparableProjection` (member time series × patterns), which gives exact percentile shortcuts, one memory-bounded chunk plan (`chunk_size_mb`) and a vectorised NaN-aware `members_percentiles`, so spatial runs are several times faster and use less memory; Zarr output is written in chunks of about 32 MB (`zarr_chunk_mb`), about 100 files for a 1° run instead of 86,000 [#119](https://github.com/MetOffice/ProFSea-tool/pull/119)
- Full-member outputs use a `member` axis with `climate_member` and `process_member` coordinates, and `compute_total` replaces `sum_comp` [#119](https://github.com/MetOffice/ProFSea-tool/pull/119)
- Fix `sum_components` double counting the total or summing percentiles, `Local` blanking a site when one member draws a pattern that is masked there, saving all spatial outputs together holding whole member arrays in memory (72 GB at 1°, now under 4 GB), and the run summary's output-size estimate counting only one of the outputs [#119](https://github.com/MetOffice/ProFSea-tool/pull/119)
- IMBIE3 ice-sheet observations are averaged to annual means weighted by the number of days in each month [#119](https://github.com/MetOffice/ProFSea-tool/pull/119)
- Updated `profsea-assets` fetch to get latest assets version with cmip5-patterns and fingerprints re-scaled [#120](https://github.com/MetOffice/ProFSea-tool/pull/120)
- Minor update to CLA documentation [#117](https://github.com/MetOffice/ProFSea-tool/pull/117)
- Update `.gitignore` and remove excess files [#118](https://github.com/MetOffice/ProFSea-tool/pull/118)
- `AGENTS.md` guide to the design philosophy, conventions and known rough edges for coding agents and contributors, loaded by Claude Code via `CLAUDE.md`, and this changelog, checked on every PR [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- Fix circular import when `profsea.utils` is imported first, and stop the test suite downloading the Zenodo assets [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- `SterodynamicCMIP5` placeholder raises `NotImplementedError` instead of an abstract-class `TypeError` [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- Example script, experiments script and user guide pass `start_yr`, `baseline_start` and `baseline_end` to `Global`; worked example uses the 1986–2005 baseline for `Spatial` and `Local` too [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- Fix the docs build: API reference import of `local_model`, broken links in the documentation guide and the `Global` docstring [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- Fix the docs build: save cell outputs to `worked_example.ipynb` preventing sphinx from executing notebook.
