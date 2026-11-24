# Changelog

## Unreleased

- Packaging for PyPI: the regional assets are downloaded to the user cache directory (or `$PROFSEA_DATA_DIR`, or an existing `profsea/profsea-assets` copy in a source checkout) instead of into the installed package, with retries on transient Zenodo errors; the wheel ships only the code and `auxiliary_data` (the experiments script moves to `experimental_scripts/`); package metadata gains a licence, Python requirement, classifiers and project links; a release workflow publishes GitHub releases to PyPI; and the WAIS and Peninsula ISMIP6 parameter files carry the right titles [#NNN](https://github.com/MetOffice/ProFSea-tool/pull/NNN)
- Updated `profsea-assets` fetch to get latest assets version with cmip5-patterns and fingerprints re-scaled [#120](https://github.com/MetOffice/ProFSea-tool/pull/120)
- Minor update to CLA documentation [#117](https://github.com/MetOffice/ProFSea-tool/pull/117)
- Update `.gitignore` and remove excess files [#118](https://github.com/MetOffice/ProFSea-tool/pull/118)
- `AGENTS.md` guide to the design philosophy, conventions and known rough edges for coding agents and contributors, loaded by Claude Code via `CLAUDE.md`, and this changelog, checked on every PR [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- Fix circular import when `profsea.utils` is imported first, and stop the test suite downloading the Zenodo assets [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- `SterodynamicCMIP5` placeholder raises `NotImplementedError` instead of an abstract-class `TypeError` [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- Example script, experiments script and user guide pass `start_yr`, `baseline_start` and `baseline_end` to `Global`; worked example uses the 1986–2005 baseline for `Spatial` and `Local` too [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- Fix the docs build: API reference import of `local_model`, broken links in the documentation guide and the `Global` docstring [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- Fix the docs build: save cell outputs to `worked_example.ipynb` preventing sphinx from executing notebook.
