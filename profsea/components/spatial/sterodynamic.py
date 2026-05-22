from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from profsea.components.core.base import SpatialComponent
from profsea.components.core.state import ClimateState
from profsea.utils import interpolate_to_grid, sample_members_2D


class SterodynamicCMIP6(SpatialComponent):
    """
    Parameters & Attributes
    ----------
    global_projection: np.ndarray
        Array of global sea level rise projections to use as input for sterodynamic projection.
    """

    def __init__(
        self,
        global_projection: np.ndarray,
        patterns_dir: str = None,
        sample_spatial: bool = False,
    ):
        # Convert to dask array for cheap as possible compute
        self._global_projection = da.from_array(global_projection, chunks="auto")
        self.sample_spatial = sample_spatial

        if patterns_dir is None:
            raise FileNotFoundError(
                "Please specify the path to the CMIP6 sterodynamic "
                "patterns using the 'patterns_dir' argument."
            )
        else:
            self.patterns_dir = Path(patterns_dir)

    @property
    def global_projection(self):
        return self._global_projection

    def _load_CMIP6_slopes(self) -> da.Array:
        """
        Load in the CMIP6 slope coefficients.
        :return: 3D Dask array of regression coefficients (model, lat, lon)
        """
        slope_files = list(Path(self.patterns_dir).glob("*/zos_regression_ssp585_*.nc"))

        if not slope_files:
            raise FileNotFoundError(
                f"No NetCDF slope files found in {self.patterns_dir}"
            )

        # Lazily load all files keeping metadata intact
        datasets = [
            xr.open_dataarray(f, chunks={"lat": 45, "lon": 45}) for f in slope_files
        ]

        # Concatenate along a new dimension (representing the ensemble/models)
        slopes_stack = xr.concat(datasets, dim="model")

        return slopes_stack

    def _calc_expansion_contribution(
        self, rng: np.random.Generator, state: ClimateState
    ) -> da.Array:
        """
        Calculate the thermal expansion contribution to the regional component of
        sea level rise.
        :param scenario: emission scenario
        :param nsmps: determine the number of samples
        :return: expansion estimates converted to mm/yr
        """
        # Select slope coefficients based on the MIP
        coeffs_da = self._load_CMIP6_slopes()

        # Align the grid coordinates + interpolate if necessary
        interp_da = interpolate_to_grid(coeffs_da, state.grid_lats, state.grid_lons)
        coeffs = interp_da.data

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
                (state.n_members, state.grid_lats.shape[0], state.grid_lons.shape[0]),
            )

    def project(self, state: ClimateState, rng) -> np.ndarray:
        # Calculate percentiles locally without mutating self
        if state.output_percentiles is not None:
            current_projection = sample_members_2D(
                self.global_projection, state.output_percentiles
            )
        else:
            current_projection = self.global_projection

        expansion_contribution = self._calc_expansion_contribution(rng, state)

        sterodynamic_projection = (
            current_projection[:, :, None, None] * expansion_contribution[:, None, :, :]
        )
        return sterodynamic_projection


class SterodynamicCMIP5(SpatialComponent):
    """
    Placeholder for a sterodynamic SLR component based on CMIP5 projections.
    """

    def __init__(self):
        pass

    def project(self, state: ClimateState, rng) -> np.ndarray:
        # For now, just return zeros as a placeholder
        return np.zeros(
            (
                state.n_members,
                state.n_years,
                state.grid_lats.shape[0],
                state.grid_lons.shape[0],
            )
        )
