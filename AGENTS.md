# AGENTS.md — ProFSea Codebase Summary

## Overview

ProFSea (`profsea`, v3) is a rapid, modular sea-level simulator from the Met Office. It
emulates each contribution to sea-level change (ice sheets, glaciers, thermal expansion,
land water, GIA, sterodynamics) statistically, driven only by global mean surface
temperature (GMST) and ocean heat content (OHC) anomalies, and produces Monte Carlo
ensembles of global mean sea-level rise (GMSLR) plus gridded or site-specific relative
sea-level change.

It is a single pure-Python package (Python 3.12; numpy, xarray, dask, scipy, zarr, rich).
The design philosophy follows SpeedyWeather.jl: **every component is as modular and as
general as possible**, and a model is nothing more than a user-chosen collection of
components.

## Design Philosophy

These rules decide how new code should look. When a change conflicts with one of them,
stop and raise it with the user rather than working around it.

1. **Models are thin orchestrators.** `Global`, `Spatial` and `Local` build a state, hand
   out random streams, loop over components and label the output. They know nothing about
   any particular physical process. Never add `if name == "glacier"` style logic, or
   process-specific keyword arguments, to a model class.
2. **Components are self-contained.** The constructor holds configuration and calibration
   choices; `project(state, rng)` turns the shared state into an array of the canonical
   shape. A component never reaches into another component or into the model object.
3. **Shared information travels through the state.** Anything more than one component
   needs (drivers, ensemble sizes, time axis, target grid, dtype) belongs in a `*State`
   dataclass. Add a field there instead of adding arguments to `project`.
4. **A new method is a new class, not a flag.** Alternative science lives side by side
   (`GreenlandAR6` vs `GreenlandSMBAR5` + `GreenlandDynAR5`, `LandwaterAR5` vs
   `LandwaterAR6`) and the user picks by instantiating. Name classes
   `<Process><Method or source>`.
5. **The user owns composition and naming.** Components are passed as
   `{name: instance}`. Any subset, any order, any keys and repeated classes must work
   (e.g. three `AntarcticaISMIP6` instances for `wais`, `eais`, `peninsula`). Do not
   assume a key exists.
6. **Forcing-agnostic.** Any GMST/OHC trajectory, any ensemble size, any scenario label.
   Do not add lookups keyed on scenario names; the AR5 legacy components that do this
   (`AntarcticaDynAR5`, `GreenlandDynAR5`) are the exception, not the pattern.
7. **The spatial stage is decoupled from the global stage.** Spatial components accept
   any `(members…, time)` projection and any pattern files. Defaults point at the Zenodo
   assets but every path is overridable. Gridded and site output share one code path via
   `SpatialComponent.extract_spatial`, so a spatial component written once works in both
   `Spatial` and `Local`.
8. **Generalise before specialising.** No new hard-coded years, array lengths, file paths
   or ensemble sizes; derive them from the state or take them as constructor arguments.
   The existing constants (start year 2006, AR5 end 2100, index `94`/`95`) are legacy to
   be lifted out over time, not copied.
9. **Lazy and light by default.** Spatial work stays in dask until `save_components`;
   arrays use `state.dtype` (float32 by default).

## Repository Layout

```
ProFSea-tool/
├── profsea/
│   ├── __init__.py            # "profsea" logger with a RichHandler
│   ├── components/
│   │   ├── core/              # Framework: base classes, states, models
│   │   ├── global_/           # GMSLR components (numpy)
│   │   └── spatial/           # Regional components (dask)
│   ├── utils/                 # utils.py (sampling, regridding, I/O), ui.py (rich panels)
│   ├── auxiliary_data/        # Calibration files; component-obs/ holds observations
│   ├── profsea-assets/        # Zenodo download (gitignored, fetched on demand)
│   ├── development/           # Exploratory notebooks, not part of the package API
│   └── scm_profsea_experiments.py  # Research script; needs `fair`, `rich_argparse`
├── tests/                     # pytest; mirrors the package (core/, components/, utils/)
├── docs/source/               # Sphinx (rst); notebooks in docs/source/tutorial_notebooks/
├── ukcp18_example_script.py   # End-to-end global → spatial example
├── pyproject.toml             # Packaging, ruff config, version
├── CHANGELOG.md               # User-facing change history (update in every PR)
├── AGENTS.md / CLAUDE.md      # Agent guide; CLAUDE.md only imports AGENTS.md
├── profsea-env.yml            # Development conda environment
└── docs-environment.yml       # Documentation conda environment
```

## Key Types

| Type | File | Purpose |
|------|------|---------|
| `Component` | `core/base.py` | ABC; one abstract method `project(state, rng)` |
| `SpatialComponent` | `core/base.py` | Adds `global_projection` (abstract property), `fetch_assets`, `extract_spatial`, `broadcast_spatiotemporal` |
| `ClimateState` | `core/state.py` | Drivers and ensemble layout for the global stage |
| `SpatialState` / `LocalState` | `core/state.py` | Target grid or target sites, output sizes, baseline |
| `Observations` | `core/observations.py` | Cached, re-baselined observational records used to initialise global components |
| `Global` | `core/global_model.py` | Runs `Component`s on GMST forcing → GMSLR per component |
| `Spatial` | `core/spatial_model.py` | Runs `SpatialComponent`s on a regular lat/lon grid |
| `Local` | `core/local_model.py` | Runs `SpatialComponent`s at named `(lat, lon)` sites |
| `time_projection` | `core/time_projection.py` | AR5-style quadratic-in-time projection shared by AR5 components |

### Component Hierarchy

```
Component
├── ThermalExpansion                      # global_/expansion.py
├── Glacier                               # global_/glacier.py
├── GreenlandAR6, GreenlandSMBAR5, GreenlandDynAR5
├── AntarcticaISMIP6, AntarcticaSMBAR5, AntarcticaDynAR5
├── LandwaterAR6, LandwaterAR5
└── SpatialComponent
    ├── Fingerprint                       # GRD fingerprints × a global projection
    ├── SterodynamicCMIP6                 # CMIP6 zos/zostoga regression patterns × expansion
    ├── SterodynamicCMIP5                 # placeholder; raises NotImplementedError
    └── GIA                               # linear-in-time rates, no climate driver
```

`Global` and `Spatial` are exported from `profsea.components.core`; `Local` must be
imported from `profsea.components.core.local_model`.

## Information Flow

```
GMST anomaly (nt, n_years)   +   OHC anomaly (given to ThermalExpansion)
    ↓  Global(start_yr, end_yr, baseline_start, baseline_end, ...).run(scenario, T_change, member_seed)
ClimateState  (T_ens, T_int_ens, T_int_med, shared `fraction`, years, baseline, sizes, dtype)
    ↓  component.project(state, rng, observations)   # threaded, one RNG stream per component
    ↓  if initialise_from_observations=False: rebaseline_projection(...) per component
{name: DataArray (climate_member, process_member, time)}      # metres
    ↓  each DataArray handed to a SpatialComponent as `global_projection`
SpatialState (grid)  or  LocalState (sites)
    ↓  component.project(state, rng)        # lazy dask
{name: DataArray (percentile, time, lat, lon)}  or  (percentile, time, site)
    ↓  sum_components(...)  →  "total_gmslr" / "total_rsl" added to the dict in place
    ↓  save_components(...) →  Zarr v3 (default) or NetCDF, computed here
```

## Array and Ensemble Conventions

- **Two ensemble axes.** `climate_member` (size `nt`) indexes input forcing trajectories;
  `process_member` (size `num_members`) indexes Monte Carlo draws of component
  parameters. `Global.run` overwrites `nt` from the shape of `T_change`.
- **Global component output** is always `(nt, num_members, n_years)` in the state dtype.
- **Spatial component output** is `(members, time, *spatial)` when percentiles are
  requested and `(nt, num_members, time, *spatial)` otherwise, where `*spatial` is
  `(lat, lon)` for `SpatialState` and `(site,)` for `LocalState`. Build it with
  `broadcast_spatiotemporal`, which handles all four cases.
- **Time axis.** Annual. `Global` takes `start_yr` and `end_yr`, with
  `n_years = end_yr - start_yr`, so `start_yr=2006, end_yr=2301` gives 295 steps
  labelled 2006–2300. Forcing arrays must have exactly `n_years` columns
  (`check_shapes`). Use `state.start_yr`, never a literal year, to locate a year in the
  time axis (e.g. `2100 - state.start_yr`). `Spatial` and `Local` still start at 2006.
- **Baselines and observations.** `Global` also takes `baseline_start` and
  `baseline_end` (validated by `validate_projection_years`). With
  `initialise_from_observations=True` (the default) each component with an
  observational record (all except `ThermalExpansion`) starts from the observed value in
  `start_yr - 1`, read through the `Observations` object passed to
  `project`; with `False` the model passes `observations=None` and re-baselines every
  component over the baseline period afterwards. Components must handle both cases.
- **Units.** Sea level in metres throughout; GMST in K; OHC in joules (the expansion
  efficiency is applied as m per YJ). Convert literature values (mm, Gt) on entry and
  say so in a comment.
- **Percentiles are real members.** `sample_members_2D` returns the ensemble members
  closest to each percentile curve, not pointwise percentiles.
- **Feeding the spatial stage.** Run `Global` with `output_percentiles=None` (the
  default) so the full `(climate_member, process_member, time)` ensemble is available;
  the reduction to percentiles happens in `Spatial`/`Local`.
- **Grid.** `grid_config` gives start/end/step for lon and lat; the default is a 1° global
  grid of cell centres. Longitudes are normalised to [-180, 180).

## Randomness

`run(member_seed=...)` builds a `np.random.SeedSequence` and spawns one child stream per
component, in dictionary order. Results are therefore reproducible and independent of
thread scheduling, and a component's draws change only if its position in the dictionary
changes.

- Use only the `rng` passed to `project`. Never call `np.random.*` module functions or
  create a new generator inside a component.
- `state.fraction` is a shared uniform array of shape `(nt, num_members)`. Use it when a
  component must be correlated with others (as the AR5 Antarctic SMB and dynamics terms
  are); draw from `rng` when it should be independent.

## Typical Usage

```python
from profsea.components.core import Global, Spatial
from profsea.components.global_ import (
    AntarcticaISMIP6,
    Glacier,
    GreenlandAR6,
    LandwaterAR6,
    ThermalExpansion,
)
from profsea.components.spatial import GIA, Fingerprint, SterodynamicCMIP6

global_model = Global(
    components={
        "expansion": ThermalExpansion(ohc_change),  # (nt, n_years), joules
        "greenland": GreenlandAR6(),
        "wais": AntarcticaISMIP6(region="wais"),
        "glacier": Glacier(),
        "landwater": LandwaterAR6(),
    },
    start_yr=2006,
    end_yr=2301,
    baseline_start=1995,
    baseline_end=2014,
    num_members=1000,
)
projections = global_model.run(
    scenario="my_scenario", T_change=t_change, member_seed=42
)
global_model.sum_components(projections)  # adds "total_gmslr"

spatial_model = Spatial(
    components={
        "sterodynamic": SterodynamicCMIP6(projections["expansion"]),
        "greenland": Fingerprint(
            projections["greenland"], fingerprint_component="greenland"
        ),
        "glacier": Fingerprint(projections["glacier"], fingerprint_component="glacier"),
        "gia": GIA(),
    },
)
spatial_model.run(member_seed=42)
spatial_model.sum_components(spatial_model.results)  # adds "total_rsl"
spatial_model.save_components(spatial_model.results, scenario_name="my_scenario")
```

For sites instead of a grid, use
`Local(components=..., locations={"Aberdeen": (57.144, -2.080)})` with the same spatial
components.

## Extending ProFSea

**New global component**

1. Subclass `Component` in `profsea/components/global_/<process>.py`.
2. Put configuration in `__init__`; load calibration files from `auxiliary_data/` once
   (see the `functools.lru_cache` loaders in `greenland.py` and `landwater.py`).
3. Implement `project(self, state: ClimateState, rng, observations)` and return
   `(state.nt, state.num_members, state.n_years)` in `state.dtype`, in metres. When
   `observations` is not `None`, initialise from the observed value in
   `state.start_yr - 1` (see `Glacier.project`); when it is `None`, project without it.
   If the component needs a new observational record, add a reader to
   `profsea/utils/utils.py`, an accessor to `Observations`, and the data under
   `auxiliary_data/component-obs/`.
4. Do not mutate `state` or its arrays: components run concurrently in a thread pool and
   share them. Avoid mutating `self` in `project` for the same reason.
5. Export it from `profsea/components/global_/__init__.py` (`__all__`), add it to
   `docs/source/api_reference.rst`, and add `tests/components/test_<process>.py`.

**New spatial component**

1. Subclass `SpatialComponent` in `profsea/components/spatial/`.
2. Call `self.fetch_assets()` in `__init__` if it uses the Zenodo assets, and accept an
   optional path argument so users can supply their own patterns.
3. Provide the `global_projection` property (a dask array).
4. In `project`, regrid patterns with `self.extract_spatial(pattern, state)` and combine
   with `self.broadcast_spatiotemporal(temporal, spatial)`. Keep everything lazy; do not
   call `.compute()` or `.values` on gridded data.
5. Support both `sample_spatial=True` (draw a pattern per member) and `False` (pattern
   mean, "storyline" mode) where several patterns exist.
6. Export, document and test as above.

**New driver or shared quantity**: add a field to the relevant state dataclass and fill it
in the model's `run`. **New output format or target geometry**: extend `save_components`
or add a new state/model pair rather than branching inside components.

## Data Assets

- `profsea/auxiliary_data/` is tracked in git and holds small calibration inputs (ISMIP6
  Antarctic parameters, Greenland calibration CSV, AR6 land-water projections) and, in
  `component-obs/`, the observational records read by `Observations` (IMBIE3 ice
  sheets, Zheng 2026). Keep additions small.
- `profsea/profsea-assets/` (GRD fingerprints, GIA rates, CMIP6 sterodynamic patterns) is
  downloaded from Zenodo record 20427061 by `fetch_zenodo_fingerprints` the first time a
  `Spatial`, `Local` or spatial component is constructed. It is gitignored; never commit
  it. New large assets go into a new Zenodo release, and the download URL (currently
  repeated in `base.py`, `spatial_model.py` and `local_model.py`) must be updated in all
  three places.

## Testing

```bash
conda env create -f profsea-env.yml && conda activate profsea
pip install -e ".[dev]"

pytest tests/                                         # full suite, a few seconds
pytest tests/components/test_glacier.py               # one file
pytest tests/components/test_glacier.py::test_glacier_projection_shape
```

- Every new feature, bug fix or logic change needs unit tests in the same commit. A
  change is not complete until its tests have been added and run.
- Tests mirror the package: `tests/core/`, `tests/components/`, `tests/utils/`. There is
  no `conftest.py`; each file builds its own small state helper (see `get_dummy_state` in
  `tests/components/test_glacier.py`).
- Keep tests small and fast: a few members, a few years, synthetic `xr.DataArray`
  patterns. Always assert the output shape, and seed with `np.random.default_rng(42)`.
- **Tests must not touch the network or the real assets.** Patch
  `SpatialComponent.fetch_assets` for component tests and
  `profsea.components.core.<module>.fetch_zenodo_fingerprints` for model tests, and write
  only under `tmp_path`.

## Code Style

```bash
ruff check . --fix
ruff format .
```

CI runs `ruff check .` and `ruff format --check .`; both must pass.

- Ruff settings live in `pyproject.toml`: line length 88, target `py312`, rule sets
  `E, F, W, I, UP`, double quotes. Notebooks are excluded.
- Start modules with `from __future__ import annotations` and type-hint public
  signatures. This is required, not stylistic: the docs build mocks numpy, xarray,
  dask, scipy and others (`autodoc_mock_imports` in `docs/source/conf.py`), so an
  annotation such as `list | np.ndarray` evaluated at import time breaks the API
  reference.
- NumPy-style docstrings with `Parameters` / `Returns` sections, written as
  `name: type`. State array shapes in docstrings and in trailing comments, e.g.
  `# (climate_mem, process_mem, time)`.
- Prefer descriptive names (`temperature`, `global_projection`) over abbreviations.
  Established shorthand from the literature is fine (`T_int`, `smb`, `gia`, `ohc`).
- Vectorise with numpy/dask broadcasting; no Python loops over members or years.
- Cite the source (paper, IPCC chapter and table, upstream repository) next to every
  empirical constant.
- Log through `logging.getLogger(__name__)`; rich markup is enabled. Run summaries and
  panels belong in `profsea/utils/ui.py`. No bare `print`.
- British spelling in documentation and user-facing text ("realisation", "licence").

## Documentation

- Sphinx sources are in `docs/source/` (`.rst` plus notebooks through `nbsphinx`).
  Notebooks are committed without outputs, so the docs build executes them; they must
  run top to bottom with the `profsea` kernel, and the build needs `pandoc` on `PATH`.
  Executing them writes files into `docs/source/tutorial_notebooks/`, some of them
  tracked, so build from a copy if you need to keep those untouched.
- Build with `conda env create -f docs-environment.yml`, then
  `make --directory=docs html`; output lands in `docs/build/html`. The build should
  finish without warnings.
- `api_reference.rst` lists classes explicitly with `automodule :members:`. Add every
  new public class there.
- Tutorial notebooks live in `docs/source/tutorial_notebooks/`. Files they write there
  are gitignored; add any new output file to `.gitignore` too.
- Docs deploy to GitHub Pages on every push to `profsea-v3`.

## Git and Pull Requests

- Open an issue and agree the approach before any significant change.
- Version 3 development happens on `profsea-v3`: the test, lint and docs workflows only
  trigger for that branch, so branch from it and target it in PRs.
- Branch names are `<initials>/<short-hyphenated-description>`, e.g.
  `js/fix-greenland-dynamics`.
- Before pushing: `pytest tests/`, `ruff check . --fix`, `ruff format .`,
  `make --directory=docs html`.
- Every PR adds a line to `CHANGELOG.md` under `## [Unreleased]`, in the matching
  `Added` / `Changed` / `Fixed` / `Removed` group, written for users and ending with the
  PR link: `- Brief description [#NNN](https://github.com/MetOffice/ProFSea-tool/pull/NNN)`.
  On release, rename `Unreleased` to the version and date and start a new empty one.
- First-time contributors must add a row to `CONTRIBUTORS.md` to sign `CLA.md`; a CI
  check enforces it. This is a human's decision — never add a contributor row on
  someone's behalf.
- The version is set statically in `pyproject.toml` and asserted in
  `tests/utils/test_utils.py`; change both together.
- Do not commit run outputs or forcing data from the repository root (`*.nc`, `*.zarr/`,
  `*.csv`, `*.png`), nor anything under `profsea/profsea-assets/`.

## Known Rough Edges

Current behaviour that will surprise you. Do not copy these patterns; fix them only as
part of an agreed change.

- **Stale package metadata.** If `test_save_components_netcdf` reports
  `ProFSea v0.0.0`, the `profsea.egg-info/` directory in the repository root is out of
  date; rerun `pip install -e ".[dev]"`.
- **Component order in `Spatial`/`Local`.** `nt` and `num_members` are read from the
  first component's `global_projection`. `GIA` carries a `(1, 1)` dummy, so do not put it
  first in the dictionary.
- **Interpolation options are not wired up.** `Spatial(grid_interpolation=...)` and
  `Local(interpolation_method=...)` are accepted but ignored: gridded output is always
  linear and site output is always bilinear.
- **Start year only flexible in `Global`.** `Spatial` and `Local` still hard-code
  `start_year = 2006`, so the spatial stage assumes global projections that start in
  2006. `Global` also still sets `endofhistory = 2006` and `endofAR5 = 2100` for the
  AR5 components.
- **`Component.project` signature.** The abstract method in `core/base.py` still
  declares `project(state, rng)`, but `Global` calls `project(state, rng, observations)`
  and every global component accepts the third argument. Spatial components keep the
  two-argument form.
- **Scenario-dependent legacy components.** `AntarcticaDynAR5` raises `KeyError` for
  scenarios other than `rcp26/45/60/85` unless `cum_emissions_total` is given;
  `GreenlandDynAR5` switches range on `rcp85`/`ssp585`.
- **`SterodynamicCMIP5`** is an unimplemented placeholder; constructing it raises
  `NotImplementedError`.
- **`ThermalExpansion.project`** rewrites `self.data_input` on first call.
- **Global side effects on import.** `spatial_model.py` and `local_model.py` call
  `warnings.filterwarnings("ignore")`; `sterodynamic.py` calls `logging.basicConfig` and
  logs through the root logger.
- **Development-guide drift.** `docs/source/development_guide.rst` says to branch from
  `main`, and `documentation.rst` links to sections that do not exist.
