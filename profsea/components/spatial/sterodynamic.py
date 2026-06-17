from __future__ import annotations

import logging
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from profsea.components.core.base import SpatialComponent
from profsea.components.core.state import ClimateState
from profsea.utils import sample_members_2D

logging.basicConfig(level=logging.WARNING)

PROFSEA_DIR = Path(__file__).resolve().parents[2]
PATTERNS_DIR = PROFSEA_DIR / "profsea-assets" / "cmip6-patterns"


class SterodynamicCMIP6(SpatialComponent):
    """
    Parameters and Attributes
    -------------------------
    global_projection: np.ndarray
        Array of global sea level rise projections to use as input for sterodynamic projection.
    """

    def __init__(
        self,
        global_projection: xr.DataArray,
        patterns_dir: str = None,
        sample_spatial: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        global_projection: xr.DataArray
             A 2D array (members x years) of global projections to apply the fingerprints to.
        patterns_dir: str, optional
             Path to directory containing CMIP6 sterodynamic patterns.
        sample_spatial: bool, optional
             If True, randomly sample a different fingerprint pattern for each member. If False, use the mean of all provided fingerprints for all members (storyline mode). Default is False.
        """
        # Convert to dask array for cheap as possible compute
        self._global_projection = da.from_array(global_projection.data, chunks="auto")
        self.sample_spatial = sample_spatial

        if patterns_dir is None:
            self.patterns_dir = PATTERNS_DIR
        else:
            self.patterns_dir = Path(patterns_dir)

    @property
    def global_projection(self):
        return self._global_projection

    def _load_CMIP6_slopes(self) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Load in the CMIP6 slope coefficients.

        Parameters
        ----------
        None

        Returns
        -------
        xr.DataArray
            A dask array of shape (n_models, n_lats, n_lons) containing the sterodynamic
            fingerprint patterns (i.e., regression coefficients) for each CMIP6 model.
        xr.DataArray
            A dask array of shape (n_models, n_lats, n_lons) containing the land mask for
            each CMIP6 model, if present.
        """
        slope_files = sorted(
            Path(self.patterns_dir).glob("*/zos_regression_ssp585_*.nc"),
            key=lambda p: p.name,
        )

        if not slope_files:
            raise FileNotFoundError(
                f"No NetCDF slope files found in {self.patterns_dir}"
            )

        # Lazily load all files keeping metadata intact
        datasets = [
            xr.open_dataset(f, chunks={"lat": 45, "lon": 45})[
                "zos_zostoga_regression_slope"
            ]
            for f in slope_files
        ]

        # Concatenate along a new dimension (representing the ensemble/models)
        slopes_stack = xr.concat(datasets, dim="model")

        # Read land mask if present
        mask_files = sorted(
            Path(self.patterns_dir).glob("*/zos_mask_ssp585_*.nc"), key=lambda p: p.name
        )

        if mask_files:
            if len(slope_files) == len(mask_files):
                self.land_mask_present = True

                datasets_mask = [
                    xr.open_dataset(f, chunks={"lat": 45, "lon": 45})["zos_mask"]
                    for f in mask_files
                ]
                mask_stack = xr.concat(datasets_mask, dim="model")
                # mask_stack = mask_stack.sum(dim='model', skipna=True)
            else:
                logging.warning(
                    "There is a mismatch between number of slope files and mask files. "
                    "Ignoring mask files."
                )
                self.land_mask_present = False
                mask_stack = None
        else:
            self.land_mask_present = False
            mask_stack = None

        return slopes_stack, mask_stack

    def _calc_expansion_contribution(
        self, rng: np.random.Generator, state: ClimateState
    ) -> da.Array:
        """
        Calculate the thermal expansion contribution to the regional component of
        sea level rise.

        Parameters
        ----------
        rng: np.random.Generator
            Random number generator for sampling spatial patterns if needed.
        state: ClimateState
            The state object containing the target grid information and number of members.

        Returns
        -------
        da.Array
            A dask array of shape (members, years, lat, lon) containing the thermal expansion contribution to the sterodynamic component for each member and year.
        """
        # Select slope coefficients based on the MIP
        coeffs_da, mask_da = self._load_CMIP6_slopes()

        if self.land_mask_present:  # apply land mask
            coeffs_da = coeffs_da.where(mask_da == 0.0)

        # Get the data either at sites or on a grid
        interp_da = self.extract_spatial(coeffs_da, state)
        coeffs = interp_da.data
        spatial_shape = coeffs.shape[1:]  # either (lat, lon) or (site,)

        if self.sample_spatial:
            rand_samples = rng.choice(
                coeffs.shape[0], size=state.n_members, replace=True
            )
            return coeffs[rand_samples, :, :]
        else:
            # Calc pattern ensemble mean
            mean_coeff = da.nanmean(coeffs, axis=0)
            return da.broadcast_to(
                mean_coeff,
                (state.n_members, *spatial_shape),
            )

    def project(self, state: ClimateState, rng) -> np.ndarray:
        """
        Project the sterodynamic component by applying the CMIP6 patterns to the global expansion projection.

        Parameters
        ----------
        state: ClimateState
            The state object containing the target grid information and number of members.
        rng: np.random.Generator
            Random number generator for sampling spatial patterns if needed.

        Returns
        -------
        np.ndarray
            A numpy array of shape (members, years, lat, lon) containing the sterodynamic component for each member and year.
        """
        # Calculate percentiles locally without mutating self
        if state.output_percentiles is not None:
            current_projection = sample_members_2D(
                self.global_projection, state.output_percentiles
            )
        else:
            current_projection = self.global_projection

        expansion_contribution = self._calc_expansion_contribution(rng, state)

        return self.broadcast_spatiotemporal(current_projection, expansion_contribution)


class SterodynamicCMIP5(SpatialComponent):
    """
    Placeholder for a sterodynamic SLR component based on CMIP5 projections.
    """

    def __init__(self):
        pass

    def project(self, state, rng) -> da.Array:
        # Dynamically determine the spatial shape to prevent crashes when using LocalState
        if hasattr(state, "target_lats"):
            spatial_shape = (len(state.target_lats),)
        else:
            spatial_shape = (state.grid_lats.shape[0], state.grid_lons.shape[0])

        return da.zeros((state.n_members, state.n_years, *spatial_shape))
