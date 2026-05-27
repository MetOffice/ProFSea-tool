from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from profsea.components.core.base import SpatialComponent
from profsea.components.core.state import SpatialState
from profsea.utils import interpolate_to_grid, sample_members_2D


class Fingerprint(SpatialComponent):
    """
    Spatial fingerprint component that applies spatial patterns to global projections.
    """

    def __init__(
        self,
        global_projection: np.ndarray,
        fingerprint_paths: str | Path | list[str | Path],
        scaling_factor: float = 1.0,
        sample_spatial: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        global_projection: np.ndarray
             A 2D array (members x years) of global projections to apply the fingerprints to.
        fingerprint_paths: str, Path, or list of str/Path
             Path(s) to NetCDF files containing the spatial fingerprint patterns. Each file should contain a DataArray with dimensions (lat, lon).
        scaling_factor: float, optional
             Optional multiplier to apply to the fingerprint patterns (e.g., to convert from m to mm). Default is 1.0 (no scaling).
        sample_spatial: bool, optional
             If True, randomly sample a different fingerprint pattern for each member. If False, use the mean of all provided fingerprints for all members (storyline mode). Default is False.
        """
        self._global_projection = da.from_array(global_projection, chunks="auto")
        self.scaling_factor = scaling_factor
        self.sample_spatial = sample_spatial

        # Normalize the input to always be a list of Path objects
        if isinstance(fingerprint_paths, (str, Path)):
            self.fp_paths = [Path(fingerprint_paths)]
        else:
            self.fp_paths = [Path(p) for p in fingerprint_paths]

        for p in self.fp_paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing fingerprint file: {p}")

    @property
    def global_projection(self):
        return self._global_projection

    def _load_and_interpolate(self, state: SpatialState) -> da.Array:
        """
        Lazily load and regrid all provided fingerprints.

        Parameters
        ----------
        state: SpatialState
            The state object containing the target grid information.

        Returns
        -------
        da.Array
        """
        grids = []
        for path in self.fp_paths:
            fp_da = xr.open_dataarray(path, chunks={"latitude": 45, "longitude": 45})
            fp_interp = interpolate_to_grid(fp_da, state.grid_lats, state.grid_lons)
            grids.append(fp_interp.data * self.scaling_factor)

        # Stack them into a 3D array: (n_fingerprints, lat, lon)
        return da.stack(grids, axis=0)

    def project(self, state: SpatialState, rng: np.random.Generator) -> da.Array:
        """
        Calculate the spatial projection by applying the fingerprints to the global projection.

        Parameters
        ----------
        state: SpatialState
            The state object containing the target grid information and number of members.
        rng: np.random.Generator
            Random number generator for sampling fingerprints if sample_spatial is True.

        Returns
        -------
        da.Array
            A 4D array of shape (members, years, lat, lon) containing the spatial projections for each member and year.
        """
        fps = self._load_and_interpolate(state)  # Shape: (n_fps, lat, lon)

        # Handle the global projection
        if state.output_percentiles is not None:
            global_proj = sample_members_2D(
                self.global_projection, state.output_percentiles
            )
        else:
            global_proj = self.global_projection

        # Determine the spatial fingerprint for each member
        n_fps = fps.shape[0]

        if n_fps == 1:
            # Only one fingerprint available
            selected_fps = da.broadcast_to(
                fps[0],
                (state.n_members, state.grid_lats.shape[0], state.grid_lons.shape[0]),
            )
        elif self.sample_spatial:
            # Probabilistic mode: pick a random fingerprint per member
            fp_indices = rng.integers(0, n_fps, size=state.n_members)
            selected_fps = fps[fp_indices, :, :]
        else:
            # Storyline mode: take the mean of the available fingerprints
            mean_fp = da.nanmean(fps, axis=0)
            selected_fps = da.broadcast_to(
                mean_fp,
                (state.n_members, state.grid_lats.shape[0], state.grid_lons.shape[0]),
            )

        # Broadcast and multiply: (members, years) * (members, lat, lon)
        spatial_projection = global_proj[:, :, None, None] * selected_fps[:, None, :, :]
        return spatial_projection
