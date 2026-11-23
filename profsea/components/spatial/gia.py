from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from profsea.components.core.base import SeparableProjection, SpatialComponent
from profsea.components.core.state import SpatialState

PROFSEA_DIR = Path(__file__).resolve().parents[2]
GIA_DIR = PROFSEA_DIR / "profsea-assets" / "gia"


class GIA(SpatialComponent):
    """Relative sea-level change from glacial isostatic adjustment (GIA).

    GIA is the ongoing response of the solid Earth to past ice-sheet loading.
    Each GIA model gives a constant rate of local sea-level change (mm/yr), so
    the change relative to the baseline grows linearly with time. It does not
    depend on the warming scenario, so this component has no global projection.
    """

    def __init__(
        self,
        gia_paths: str | Path = None,
        sample_spatial: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        gia_paths: str, Path, list of str/Path, or None
            GIA rate files (NetCDF, mm/yr), or directories containing them.
            A file may hold several models along a "model" dimension. Defaults
            to the GIA estimates in the downloaded assets.
        sample_spatial: bool, optional
            If True, each member uses one GIA model, drawn at random with equal
            probability. If False, every member uses the mean of the models.
            Default is True.
        """

        self.fetch_assets()
        self.sample_spatial = sample_spatial

        if gia_paths is None:
            raw_paths = [GIA_DIR]
        elif isinstance(gia_paths, (str, Path)):
            raw_paths = [Path(gia_paths)]
        else:
            raw_paths = [Path(p) for p in gia_paths]

        # Resolve directories into files, and keep direct file paths
        self.gia_files = []
        for p in raw_paths:
            if not p.exists():
                raise FileNotFoundError(f"Provided GIA path does not exist: {p}")

            if p.is_dir():
                # Extract all .nc files if a directory is passed
                self.gia_files.extend(list(p.glob("*.nc")))
            elif p.is_file() and p.suffix == ".nc":
                # Keep direct NetCDF files
                self.gia_files.append(p)

        if not self.gia_files:
            raise FileNotFoundError(
                f"No GIA NetCDF files found in the provided paths: {raw_paths}"
            )

    def _load_and_interpolate_rates(self, state: SpatialState) -> da.Array:
        """
        Lazily loads all GIA files, regrids them, and stacks them into a
        single 3D array of shape (total_models, lat, lon).

        Parameters
        ----------
        state: SpatialState
            The spatial state containing the target grid information.

        Returns
        -------
        da.Array
            A Dask array of shape (total_models, lat, lon) containing the regridded GIA rates.
        """
        grids = []
        for path in self.gia_files:
            gia_da = xr.open_dataarray(path, chunks={"lat": -1, "lon": -1})
            interp_da = self.extract_spatial(gia_da, state)
            data = interp_da.data

            # Determine expected spatial dims based on state
            spatial_dims = 2 if hasattr(state, "grid_lats") else 1

            # If the raw file lacks a 'model' dimension, prepend it
            if data.ndim == spatial_dims:
                data = data[None, ...]

            grids.append(data)

        # Concatenate along the model axis (axis 0)
        return da.concatenate(grids, axis=0)

    def project_separable(
        self, state: SpatialState, rng: np.random.Generator
    ) -> SeparableProjection:
        """
        Accumulated GIA: elapsed time since the baseline midpoint times the rates.

        Parameters
        ----------
        state: SpatialState | LocalState
            Target grid or sites, baseline period and number of members.
        rng: np.random.Generator
            Random number generator for drawing a GIA model per member if
            sample_spatial is True.

        Returns
        -------
        SeparableProjection
            Shared time series in years (scaled from mm to m) and GIA rates
            (n_models, *spatial).
        """
        gia_rates = self._load_and_interpolate_rates(state)  # mm/yr

        # Sea-level change relative to the baseline = rate x years elapsed
        # since the middle of the baseline period (2005 for 1995-2014).
        baseline_start, baseline_end = state.baseline_yrs
        baseline_middle = (baseline_start + baseline_end + 1) / 2
        years = state.endofhistory + np.arange(state.n_years)
        mm_to_m = 1e-3
        years_since_baseline = (years - baseline_middle) * mm_to_m

        patterns, pattern_index = self.choose_patterns(
            gia_rates, state, rng, self.sample_spatial
        )
        # Every member shares this time series; members differ only by GIA model
        return SeparableProjection(
            temporal=years_since_baseline,
            patterns=patterns,
            pattern_index=pattern_index,
        )
