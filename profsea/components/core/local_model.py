from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from profsea.utils import (
    PROFSEA_ASSETS_URL,
    PROFSEA_ASSETS_VERSION,
    fetch_zenodo_fingerprints,
    save_components,
)

from .base import SpatialComponent
from .regional import RegionalEnsemble
from .state import LocalState

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PROFSEA_DIR = Path(__file__).resolve().parent.parent.parent


class Local(RegionalEnsemble):
    """Sea-level projections at named sites, such as tide gauges.

    Runs the same regional Monte Carlo as :class:`Spatial` (see
    :class:`~profsea.components.core.regional.RegionalEnsemble`), but extracts
    patterns at each site instead of regridding them. Outputs have dimensions
    (percentile, time, site), or (member, time, site) when every member is kept.

    Site values come from the four surrounding cells of each pattern's grid,
    ignoring land cells. If all four are land in one model's pattern, that
    model has no value at the site, and members that drew it are left out of
    the site's percentiles. A site with no value in some component for every
    member is treated as land and set to NaN in all outputs.
    """

    def __init__(
        self,
        components: dict[str, SpatialComponent],
        locations: dict[str, tuple[float, float]],
        interpolation_method: str = "nearest",
        start_yr: int = 2006,
        end_yr: int = 2301,
        baseline_yrs: tuple = (1995, 2014),
        output_percentiles: list | np.ndarray = [5, 17, 50, 83, 95],
        num_spatial_members: int = 1000,
        dtype: np.dtype = np.float32,
        compute_total: bool = True,
        chunk_size_mb: float = 64,
    ) -> None:
        """
        Parameters
        ----------
        components: dict
            Spatial components keyed by name, e.g.
            ``{"glacier": Fingerprint(...), "gia": GIA()}``. All components
            driven by a global projection must come from the same ``Global``
            run.
        locations: dict
            Site names mapped to (latitude, longitude), e.g.
            ``{"Aberdeen": (57.144, -2.080)}``.
        interpolation_method: str, optional
            Interpolation method for extracting patterns at sites. Currently not
            applied: sites always use the land-aware four-cell interpolation
            described above. Default is 'nearest'.
        start_yr: int, optional
            The projections run from start_yr to end_yr - 1. Default is 2006.
        end_yr: int, optional
            The projections run from start_yr to end_yr - 1. Default is 2301.
        baseline_yrs: tuple, optional
            First and last year of the baseline period. Default is (1995, 2014).
        output_percentiles: list or np.ndarray, optional
            Percentiles to output, taken over the Monte Carlo members at every
            site and year. If None or empty, every member is output. Default is
            [5, 17, 50, 83, 95].
        num_spatial_members: int, optional
            Number of Monte Carlo members, drawn from the global ensemble with an
            equal share per climate trajectory. If equal to the global ensemble
            size, every global member is used exactly once. Palmer et al. (2020)
            used 100,000 per tide gauge, which is cheap at a few sites.
            Default is 1000.
        dtype: np.dtype, optional
            Data type of the outputs. Default is np.float32.
        compute_total: bool, optional
            If True, add "total_rsl": the components summed member by member,
            before any percentiles are taken. Default is True.
        chunk_size_mb: float, optional
            Approximate memory, in MB, for one block of the total together with
            the blocks of every component summed into it. Default is 64.
        """
        fetch_zenodo_fingerprints(
            zenodo_url=PROFSEA_ASSETS_URL,
            data_dir=PROFSEA_DIR,
            expected_folder_name="profsea-assets",
            version=PROFSEA_ASSETS_VERSION,
        )

        self.start_yr = start_yr
        self.end_yr = end_yr
        self.baseline_yrs = baseline_yrs
        self.output_percentiles = output_percentiles
        self.n_years = self.end_yr - self.start_yr
        self.interpolation_method = interpolation_method
        self.num_spatial_members = num_spatial_members
        self.dtype = dtype
        self.compute_total = compute_total
        self.chunk_size_mb = chunk_size_mb

        self._set_components(components, "Local")

        # Parse the locations dictionary
        self.site_names = list(locations.keys())
        self.target_lats = [coords[0] for coords in locations.values()]
        self.target_lons = [coords[1] for coords in locations.values()]

        # Length of the outputs' leading axis: percentiles or members
        if self._has_percentiles():
            self.num_output_members = len(self.output_percentiles)
        else:
            self.num_output_members = self.num_spatial_members

        logger.info(
            f"Baseline period = {self.baseline_yrs[0]} to {self.baseline_yrs[1]}"
        )
        logger.info(f"Configured for {len(self.site_names)} specific target locations.")

    # Shared output writer (profsea.utils.save_components), used as a method
    save_components = save_components

    def _spatial_coords(self) -> tuple[list[str], dict, str]:
        """Site dimension and coordinates (see RegionalEnsemble._spatial_coords)."""
        coords = {
            "site": self.site_names,
            "lat": ("site", self.target_lats),
            "lon": ("site", self.target_lons),
        }
        return ["site"], coords, "Local"

    def _apply_universal_mask(self) -> None:
        """Set every output to NaN at sites that some component treats as land.

        A site counts as land if any component has no value there for every
        member (typically because of the sterodynamic land mask). A site where
        only some members are missing, because they drew a model that is
        masked there, is kept. Updates ``self.results`` in place and warns
        about the sites it masks.
        """
        if not hasattr(self, "results") or not self.results:
            return

        # A component has data at a site if any member (or percentile) is valid
        # there. Checking a single member is not enough: with sampled patterns,
        # one member can draw a model that is masked at the site while others
        # are not.
        ensemble_dim = "percentile" if self._has_percentiles() else "member"
        has_data = [
            comp.isel(time=0).notnull().any(dim=ensemble_dim)
            for comp in self.results.values()
        ]

        # Site is valid if it has data in ALL components
        valid_site_mask = (
            xr.concat(has_data, dim="component").all(dim="component").compute()
        )
        # Keep only the site index so applying the mask cannot clash with the
        # scalar coordinates left over from the slicing above
        valid_site_mask = valid_site_mask.reset_coords(drop=True)

        masked_sites = [
            site
            for site, is_valid in zip(self.site_names, valid_site_mask.values)
            if not is_valid
        ]
        if masked_sites:
            logger.warning(
                f"The following site(s) may be over land: {', '.join(masked_sites)}. "
                "Expect NaN values for all components. "
                "Either try increasing your grid resolution or check your site coordinates."
            )

        for name in self.results.keys():
            self.results[name] = self.results[name].where(valid_site_mask)

    def run(self, member_seed: int = 42) -> dict[str, xr.DataArray]:
        """
        Run the local model to generate site-specific projections.

        Parameters
        ----------
        member_seed: int, optional
            Seed for the Monte Carlo member and pattern draws. Default is 42.

        Returns
        -------
        dict
            Projections for each component, plus "total_rsl" if compute_total,
            of shape (percentiles or members, n_years, n_sites). The first year
            is computed straight away to find land sites; everything else is
            computed when the results are used or saved.
        """
        logger.info(
            f"Simulating {len(self.components)} sea-level components for {len(self.site_names)} sites..."
        )
        self.results = self._run_ensemble(
            member_seed,
            LocalState,
            (len(self.site_names),),
            "Localising components...",
            target_lats=self.target_lats,
            target_lons=self.target_lons,
            interpolation_method=self.interpolation_method,
        )
        self._apply_universal_mask()
        return self.results
