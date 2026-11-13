from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import xarray as xr

from profsea.components.core.base import Component
from profsea.components.core.global_model import ClimateState
from profsea.components.core.observations import Observations
from profsea.components.core.time_projection import time_projection


@functools.lru_cache(maxsize=1)
def load_landwater_projection():
    """Loads the NetCDF once and keeps the VALUES in memory."""
    path = (
        Path(__file__).parents[2] / "aux_data" / "ssp_global_landwater_projections.nc"
    )
    with xr.open_dataset(path) as ds:
        ds.load()
    return ds


class LandwaterAR6(Component):
    def __init__(self):
        self.lw_ds = load_landwater_projection()

    def project(
        self,
        state: ClimateState,
        rng: np.random.Generator,
        observations: Observations | None,
    ) -> np.ndarray:
        """Project land water storage contribution to GMSLR. This follows the IPCC
        AR6 methodology as closely as possible. The observed contribution, from
        Zheng et al., (2026), is added at the `start_yr` relative to the specified
        baseline.

        Parameters
        ----------
        state : ClimateState
            Climate state containing projection years and baseline period.
        rng : np.random.Generator
            Random number generator used for uncertainty sampling.
        observations : Observations or None
            Observational data used to initialise the projection. This component
            requires observational initialisation and raises an error if None.

        Returns
        -------
        np.ndarray
            Land water storage contribution to GMSLR.
        """
        if observations is None:
            raise ValueError("LandwaterAR6() requires observational initialisation.")

        lw_ds = self.lw_ds.astype(state.dtype)

        # Interpolate to annual projections
        interp_ds = lw_ds.interp(
            years=np.arange(2005, 2301, 1), method="linear"
        ).astype(state.dtype)

        # Base array: Shape (n_samples_in_nc, 296)
        lw_base = interp_ds["sea_level_change"].values * 1e-3
        n_samples_nc = lw_base.shape[0]

        # Sample!
        sample_indices = rng.integers(
            0, n_samples_nc, size=(state.nt, state.num_members)
        )

        years = interp_ds["years"].values

        # read sea level contribution and rate (relative to specified baseline) at start_yr
        # from Zheng+ 2026 (Sci. Adv.)
        obs = observations.landwater()

        sle = obs["sle"].sel(time=state.start_yr - 1).item()

        # Resulting shape: (nt, nm, n_years)
        lw_ens = lw_base[sample_indices]
        idx_start = np.where(years == state.start_yr)[0][0]
        lw_ens = lw_ens - lw_ens[:, :, idx_start, None]

        lw_ens += sle

        year_mask = (years >= state.start_yr) & (years < state.end_yr)
        lw_ens = lw_ens[:, :, year_mask]

        return lw_ens.reshape(state.nt, state.num_members, state.n_years)


class LandwaterAR5(Component):
    def __init__(self):
        pass

    def project(
        self, state: ClimateState, rng: np.random.Generator, observations: Observations
    ) -> np.ndarray:
        """Project land water storage contribution to GMSLR.

        The AR5 end-of-century very-likely range is reported for 2081-2100
        relative to the 1986-2005 baseline (Table 13.5 AR5). The projected
        contribution is expressed relative to the baseline specified in the
        ClimateState.

        Parameters
        ----------
        state : ClimateState
            Climate state containing projection years, and baseline period,
        rng : np.random.Generator
            Random number generator used for uncertainty sampling.
        observations : Observations or None
            Observational data used to initialise the projection. This component
            requires observational initialisation and raises an error if None.

        Returns
        -------
        np.ndarray
            Land water storage contribution to GMSLR.
        """
        if observations is None:
            raise ValueError("LandwaterAR5() requires observational initialisation.")

        # The rate at start is the one for 1993-2010 from the budget table.
        # The final amount is the mean for 2081-2100.
        nyr = (
            state.endofAR5 - 2081 + 1
        )  # number of years of the time-mean of the final amount

        # AR5 Table 13.5 very-likely range for 2081-2100, relative to 1986-2005
        final_l = -0.01
        final_u = 0.09
        # Convert to a change relative to the projection start_yr
        obs_ar5 = observations.landwater(
            baseline_start=1986,
            baseline_end=2005,
        )

        start_offset = obs_ar5["sle"].sel(time=state.start_yr - 1).item()
        final = [final_l - start_offset, final_u - start_offset]

        # read sea level contribution and rate (relative to specified baseline) at start_yr from
        # Zheng+ 2026 (Sci. Adv.)
        obs = observations.landwater()

        sle = obs["sle"].sel(time=state.start_yr - 1).item()

        # This follows the AR5 methodology, which adopts an effective contribution rate of 0.38 [0.26
        # to 0.49] mm/yr over the baseline 1993-2010 (Ar5 Ch13 Table 13.1).

        rate_obs = obs.sel(time=slice(state.start_yr - 13, state.start_yr + 4))

        x = rate_obs.time.values
        y = rate_obs.sle.values
        sigma = rate_obs.sle_unc.values

        coeffs, cov = np.polyfit(
            x,
            y,
            deg=1,
            w=1 / sigma,
            cov=True,
        )

        # returns rates and uncertainty in mm/yr
        startratemean = coeffs[0] * 1e3
        startratepm = np.sqrt(cov[0, 0]) * 1e3

        return (
            time_projection(state, startratemean, startratepm, final, rng, nfinal=nyr)
            + sle
        )
