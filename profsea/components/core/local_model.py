import logging
import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from rich.console import Console
from rich.progress import track

from profsea.utils import fetch_zenodo_fingerprints, save_components

from .base import SpatialComponent
from .state import LocalState

logger = logging.getLogger(__name__)
console = Console()
warnings.filterwarnings("ignore")

PROFSEA_DIR = Path(__file__).resolve().parent.parent.parent
ZENODO_DOWNLOAD_LINK = (
    "https://zenodo.org/records/20427061/files/profsea-assets.zip?download=1"
)


class Local:
    """Site-specific (Local) sea level rise component emulator."""

    def __init__(
        self,
        components: dict[str, SpatialComponent],
        locations: dict[str, tuple[float, float]],
        interpolation_method: str = "nearest",
        end_year: int = 2301,
        baseline_yrs: tuple = (1995, 2014),
        output_percentiles: list | np.ndarray = [5, 17, 50, 83, 95],
    ) -> None:
        """
        Parameters
        ----------
        components: dict
            Dictionary of spatial components to include in the model.
        locations: dict
            Dictionary mapping site names to (latitude, longitude) tuples.
            Example: {"Aberdeen": (57.144, -2.080)}
        interpolation_method: str, optional
            Interpolation method to use when interpolating patterns to the target sites. Default is 'nearest'.
        end_year: int, optional
            The final year of the projections. Default is 2301.
        baseline_yrs: tuple, optional
            Tuple defining the start and end years of the baseline period. Default is (1995, 2014).
        output_percentiles: list or np.ndarray, optional
            List or array of percentiles to sample from the ensemble for output.
        """
        fetch_zenodo_fingerprints(
            zenodo_url=ZENODO_DOWNLOAD_LINK,
            data_dir=PROFSEA_DIR,
            expected_folder_name="profsea-assets",
        )

        self.components = components
        self.end_year = end_year
        self.baseline_yrs = baseline_yrs
        self.output_percentiles = output_percentiles
        self.start_year = 2006
        self.n_years = self.end_year - self.start_year
        self.interpolation_method = interpolation_method

        # Parse the locations dictionary
        self.site_names = list(locations.keys())
        self.target_lats = [coords[0] for coords in locations.values()]
        self.target_lons = [coords[1] for coords in locations.values()]

        if self.output_percentiles is not None and len(self.output_percentiles) > 0:
            self.num_members = len(self.output_percentiles)
        else:
            self.num_members = next(
                iter(self.components.values())
            ).global_projection.shape[0]

        logger.info(
            f"Baseline period = {self.baseline_yrs[0]} to {self.baseline_yrs[1]}"
        )
        logger.info(f"Configured for {len(self.site_names)} specific target locations.")

    # Instance method!
    save_components = save_components

    def _arr_to_xr(self, arr_dict: dict[str, da.Array]) -> dict[str, xr.DataArray]:
        """
        Convert Dask arrays to xarray DataArrays with site coordinates.

        Parameters
        ----------
        arr_dict: dict
            Dictionary of Dask arrays to convert. Keys should be the component names and values should be Dask arrays of shape (members, time, sites).

        Returns
        -------
        dict
            Dictionary of xarray DataArrays with site coordinates. Keys are the same as in arr_dict.
        """
        xr_dict = {}
        member_dim = "percentile" if self.output_percentiles is not None else "member"

        for name, arr in arr_dict.items():
            xr_dict[name] = xr.DataArray(
                arr,
                dims=[member_dim, "time", "site"],
                coords={
                    member_dim: self.output_percentiles
                    if self.output_percentiles is not None
                    else np.arange(arr.shape[0]),
                    "time": np.arange(self.start_year, self.start_year + arr.shape[1]),
                    "site": self.site_names,
                    "lat": ("site", self.target_lats),
                    "lon": ("site", self.target_lons),
                },
                attrs={
                    "units": "m",
                    "long_name": f"Local {name} sea-level projections",
                    "source": "ProFSea-Climate v0.1",
                },
            )
        return xr_dict

    def _apply_universal_mask(self) -> None:
        """
        Identify locations that evaluate to NaN in ANY component (typically driven
        by the sterodynamic land mask) and propagate that NaN to ALL components.
        This ensures physical consistency: a site over land has no valid components.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not hasattr(self, "results") or not self.results:
            return

        member_dim = "percentile" if self.output_percentiles is not None else "member"
        spatial_slices = [
            comp.isel({member_dim: 0, "time": 0}) for comp in self.results.values()
        ]

        # Site is valid if it is non-NaN in ALL components
        stacked_slices = xr.concat(spatial_slices, dim="component")
        valid_site_mask = stacked_slices.notnull().all(dim="component").compute()

        # Inform user of land mask
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

        # Apply the mask uniformly
        for name in self.results.keys():
            self.results[name] = self.results[name].where(valid_site_mask)

    def run(self, member_seed: int = 42) -> dict[str, xr.DataArray]:
        """
        Run the local model to generate site-specific projections.

        Parameters
        ----------
        member_seed: int, optional
            Seed for random number generation to ensure reproducibility of member sampling. Default is 42.

        Returns
        -------
        dict
            Dictionary of local projections for each component, where keys are component names and values are xarray DataArrays of shape (n_members, n_years, n_sites).
        """
        seed_seq = np.random.SeedSequence(member_seed)

        logger.info(
            f"Simulating {len(self.components)} sea-level components for {len(self.site_names)} sites..."
        )

        state = LocalState(
            n_years=self.n_years,
            n_members=self.num_members,
            target_lats=self.target_lats,
            target_lons=self.target_lons,
            interpolation_method=self.interpolation_method,
            output_percentiles=self.output_percentiles,
            baseline_yrs=self.baseline_yrs,
        )

        child_seeds = seed_seq.spawn(len(self.components))
        comp_rngs = {
            name: np.random.default_rng(s)
            for name, s in zip(self.components.keys(), child_seeds)
        }

        local_projections = {}

        console.print()
        for name, comp in track(
            self.components.items(), description="Localising components..."
        ):
            # Project method should return array of shape (members, time, sites)
            lazy_projection = comp.project(state, comp_rngs[name])

            # Rechunk to optimize for reading full time-series per site
            lazy_projection = lazy_projection.rechunk({0: -1, 1: -1, 2: -1})
            local_projections[name] = lazy_projection
        console.print()

        self.results = self._arr_to_xr(local_projections)
        self._apply_universal_mask()
        return self.results

    def sum_components(self, components: dict[str, xr.DataArray]) -> xr.DataArray:
        """
        Sum the local components to get total sea-level change.

        Parameters
        ----------
        components: dict
            Dictionary of xarray DataArrays representing individual components.

        Returns
        -------
        xr.DataArray
            A single xarray DataArray representing the total sea-level projections, with appropriate attributes.
        """
        total_rsl = xr.concat(components.values(), dim="component").sum(
            dim="component", skipna=False
        )
        total_rsl.attrs = {
            "units": "m",
            "long_name": "Local total sea-level projections",
            "source": "ProFSea-Climate v0.1",
        }
        components["total_rsl"] = total_rsl
        return total_rsl
