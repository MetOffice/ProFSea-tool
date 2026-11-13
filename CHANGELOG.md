# Changelog

All notable changes to ProFSea are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Every pull
request adds a line under `Unreleased`, ending with a link to the pull request.

History before this file was started is available from the
[commit log](https://github.com/MetOffice/ProFSea-tool/commits/profsea-v3) and the
[v1.0.0](https://github.com/MetOffice/ProFSea-tool/releases/tag/v1.0.0) and
[v2.0.0](https://github.com/MetOffice/ProFSea-tool/releases/tag/v2.0.0) releases.

## [Unreleased]

### Added

- `AGENTS.md`, a guide to the code base, design philosophy and conventions for coding
  agents and new contributors.
- `CHANGELOG.md` (this file).

### Changed

- `SterodynamicCMIP5` now raises `NotImplementedError` with a pointer to
  `SterodynamicCMIP6` when constructed. It is still a placeholder, but previously failed
  with an unrelated abstract-class `TypeError`.

### Fixed

- `import profsea.utils` no longer fails with a circular `ImportError` when it is the
  first ProFSea import.
- `ukcp18_example_script.py` passes the ocean heat content array to `ThermalExpansion`
  through `data_input`, matching the current constructor.
- The test suite no longer downloads the Zenodo assets or leaves
  `profsea/temp_fingerprints.zip` behind (`test_custom_paths_missing` now patches
  `fetch_assets`).
