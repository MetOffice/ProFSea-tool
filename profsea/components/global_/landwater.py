import functools

from pathlib import Path
import numpy as np
import xarray as xr

from profsea.components.core.base import Component
from profsea.components.core.global_model import ClimateState
from profsea.components.core.time_projection import time_projection


@functools.lru_cache(maxsize=1)
def load_landwater_projection():
    """Loads the NetCDF once and keeps the VALUES in memory."""
    path = (
        Path(__file__).parents[3] / "aux_data" / "ssp_global_landwater_projections.nc"
    )
    with xr.open_dataset(path) as ds:
        ds.load()
    return ds


class LandwaterAR6(Component):
    def __init__(self):
        self.lw_ds = load_landwater_projection()

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """Project land water storage contribution to GMSLR.
        This follows the IPCC AR6 methodology as closely as possible.
        Projections are relative to 1996-2014 baseline.

        Returns
        -------
        np.ndarray
            Land water storage contribution to GMSLR.
        """

        lw_ds = self.lw_ds

        # Interpolate to annual projections
        interp_ds = lw_ds.interp(
            years=np.arange(2005, 2301, 1), method="linear"
        ).squeeze()
        interp_ds["sea_level_change"].values * 1e-3  # mm to m

        # Base array: Shape (n_samples_in_nc, 296)
        lw_base = interp_ds["sea_level_change"].values * 1e-3
        n_samples_nc = lw_base.shape[0]

<<<<<<< HEAD
        # Sample!
        sample_indices = rng.integers(0, n_samples_nc, size=(state.nt, state.nm))

        # Resulting shape: (nt, nm, nyr)
        lw_ens = lw_base[sample_indices, 1 : state.nyr + 1]
        return lw_ens.reshape(state.nt * state.nm, state.nyr)
=======
        # Make a Monte Carlo ensemble of projections
        full_repeats = (state.nt * state.num_members) // lw.shape[0]
        remainder = (state.nt * state.num_members) % lw.shape[0]
        lw = np.vstack([np.tile(lw, (full_repeats, 1)), lw[:remainder]])
        lw = lw.reshape(state.nt * state.num_members, lw.shape[1])
        lw = lw[:, 1 : state.nyr + 1]  # Start at 2006, end at end_yr
>>>>>>> profsea-climate-v2


<<<<<<< HEAD
=======

>>>>>>> profsea-climate-v2
class LandwaterAR5(Component):
    def __init__(self):
        self.startratemean = 0.38
        self.startratepm = 0.49 - 0.38

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """Old projection function. Project land water storage
        contribution to GMSLR.

        Returns
        -------
        np.ndarray
            Land water storage contribution to GMSLR.
        """

        # The rate at start is the one for 1993-2010 from the budget table.
        # The final amount is the mean for 2081-2100.
        nyr = (
            state.endofAR5 - 2081 + 1
        )  # number of years of the time-mean of the final amount
        final = [-0.01, 0.09]  # AR5

        return time_projection(
            state, self.startratemean, self.startratepm, final, rng, nfinal=nyr
        )
