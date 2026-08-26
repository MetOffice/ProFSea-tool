from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from profsea.components.core.base import SpatialComponent
from profsea.components.core.state import SpatialState
from profsea.utils import reformat_global_projection

PROFSEA_DIR = Path(__file__).resolve().parents[2]
FP_DIR = PROFSEA_DIR / "profsea-assets" / "grd-fingerprints"

FP_PATH_MAP = {
    "greendyn": [
        FP_DIR / "greendyn_klemann.nc",
        FP_DIR / "greendyn_slangen.nc",
        FP_DIR / "greendyn_spada.nc",
    ],
    "greensmb": [
        FP_DIR / "greensmb_klemann.nc",
        FP_DIR / "greensmb_slangen.nc",
        FP_DIR / "greensmb_spada.nc",
    ],
    "greenland": [FP_DIR / "greenland_ar6.nc"],
    "landwater": [FP_DIR / "landwater_slangen.nc"],
    "wais": [FP_DIR / "wais.nc"],
    "eais": [FP_DIR / "eais.nc"],
    "antdyn": [
        FP_DIR / "antdyn_klemann.nc",
        FP_DIR / "antdyn_slangen.nc",
        FP_DIR / "antdyn_spada.nc",
    ],
    "antsmb": [
        FP_DIR / "antsmb_klemann.nc",
        FP_DIR / "antsmb_slangen.nc",
        FP_DIR / "antsmb_spada.nc",
    ],
    "glacier": [
        FP_DIR / "glacier_klemann.nc",
        FP_DIR / "glacier_slangen.nc",
        FP_DIR / "glacier_spada.nc",
    ],
}


class Fingerprint(SpatialComponent):
    """
    Spatial fingerprint component that applies spatial patterns to global projections.
    """

    def __init__(
        self,
        global_projection: xr.DataArray,
        fingerprint_component: str,
        fingerprint_paths: str | Path | list[str | Path] = None,
        scaling_factor: float = 1.0,
        sample_spatial: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        global_projection: xr.DataArray
             A 2D array (members x years) of global projections to apply the fingerprints to.
        fingerprint_paths: str, Path, or list of str/Path
             Path(s) to NetCDF files containing the spatial fingerprint patterns. Each file should contain a DataArray with dimensions (lat, lon).
        scaling_factor: float, optional
             Optional multiplier to apply to the fingerprint patterns (e.g., to convert from m to mm). Default is 1.0 (no scaling).
        sample_spatial: bool, optional
             If True, randomly sample a different fingerprint pattern for each member. If False, use the mean of all provided fingerprints for all members (storyline mode). Default is False.
        """
        self._global_projection = da.from_array(global_projection.data, chunks="auto")
        self.scaling_factor = scaling_factor
        self.sample_spatial = sample_spatial
        self.fingerprint_component = fingerprint_component

        # Default paths if not provided (can be overridden by user input)
        if fingerprint_paths is None:
            # Determine default paths based on the component type
            try:
                self.fp_paths = [Path(p) for p in FP_PATH_MAP[fingerprint_component]]
            except KeyError:
                raise ValueError(
                    f"No default fingerprint paths found for fingerprint component '{fingerprint_component}'. Please provide explicit paths."
                )
        # Normalize the input to always be a list of Path objects
        elif isinstance(fingerprint_paths, (str, Path)):
            self.fp_paths = [Path(fingerprint_paths)]
        else:
            self.fp_paths = [Path(p) for p in fingerprint_paths]

        if fingerprint_paths is not None:
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
            fp_da = xr.open_dataarray(path, chunks={"lat": 45, "lon": 45})
            fp_interp = self.extract_spatial(fp_da, state)
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
        spatial_shape = fps.shape[1:]

        global_projection = reformat_global_projection(self.global_projection, state)

        # Determine the spatial fingerprint for each member
        n_fps = fps.shape[0]

        if n_fps == 1:
            # Only one fingerprint available
            selected_fps = da.broadcast_to(
                fps[0],
                (state.num_output_members, *spatial_shape),
            )
        elif self.sample_spatial:
            # Probabilistic mode: pick a random fingerprint per member
            fp_indices = rng.integers(0, n_fps, size=state.num_output_members)
            selected_fps = fps[fp_indices, ...]
        else:
            # Storyline mode: take the mean of the available fingerprints
            mean_fp = da.nanmean(fps, axis=0)
            selected_fps = da.broadcast_to(
                mean_fp,
                (state.num_output_members, *spatial_shape),
            )

        # Broadcast and multiply: (members, years) * (members, lat, lon)
        return self.broadcast_spatiotemporal(global_projection, selected_fps)
