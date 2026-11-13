# Changelog

## Unreleased

- `AGENTS.md` guide to the design philosophy, conventions and known rough edges for coding agents and contributors, loaded by Claude Code via `CLAUDE.md`, and this changelog, checked on every PR [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- Fix circular import when `profsea.utils` is imported first, and stop the test suite downloading the Zenodo assets [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- `SterodynamicCMIP5` placeholder raises `NotImplementedError` instead of an abstract-class `TypeError` [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- Example script, experiments script and user guide pass `start_yr`, `baseline_start` and `baseline_end` to `Global`; worked example uses the 1986–2005 baseline for `Spatial` and `Local` too [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
- Fix the docs build: API reference import of `local_model`, broken links in the documentation guide and the `Global` docstring [#114](https://github.com/MetOffice/ProFSea-tool/pull/114)
