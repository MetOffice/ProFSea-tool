from __future__ import annotations

import logging
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from profsea.components.core.base import SeparableProjection, SpatialComponent
from profsea.components.core.state import SpatialState

logging.basicConfig(level=logging.WARNING)

PROFSEA_DIR = Path(__file__).resolve().parents[2]
PATTERNS_DIR = PROFSEA_DIR / "profsea-assets"


class SterodynamicCMIP6(SpatialComponent):
    """Local sterodynamic sea-level change from global thermal expansion.

    Sterodynamic change is local sea-level change from ocean density and
    circulation. For each CMIP model, a regression slope relates the local
    change (zos + zostoga) to global thermal expansion (zostoga); multiplying
    a member's global thermal expansion by a model's slope gives its local
    sterodynamic change. Despite the name, ``cmip_era`` selects CMIP6 (default)
    or CMIP5 patterns.
    """

    def __init__(
        self,
        global_projection: xr.DataArray,
        patterns_dir: str = None,
        cmip_era: str = "CMIP6",
        sample_spatial: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        global_projection: xr.DataArray
            Full global thermal expansion ensemble from ``Global``, shape
            (climate_member, process_member, time), or (members, time).
        patterns_dir: str, optional
            Directory of regression patterns, one sub-directory per model.
            Defaults to the "{cmip_era}-patterns" folder of the assets.
        cmip_era: str, optional
            "CMIP6" (default, 22 models) or "CMIP5" (21 models, as used by
            Palmer et al., 2020).
        sample_spatial: bool, optional
            If True, each member uses one CMIP model's slope, drawn at random
            with equal probability. If False, every member uses the mean of
            the models ("storyline" mode). Default is True.
        """
        self.fetch_assets()

        self._set_global_projection(global_projection)
        self.sample_spatial = sample_spatial
        self.cmip_era = cmip_era.lower()

        if patterns_dir is None:
            cmip_str = self.cmip_era + "-patterns"
            self.patterns_dir = PATTERNS_DIR / cmip_str
        else:
            self.patterns_dir = Path(patterns_dir)

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
            Path(self.patterns_dir).glob("*/zos_regression_*85_*.nc"),
            key=lambda p: p.name,
        )

        if not slope_files:
            raise FileNotFoundError(
                f"No NetCDF slope files found in {self.patterns_dir}"
            )

        # Lazily load all files keeping metadata intact
        datasets = [
            xr.open_dataset(f, chunks={"lat": -1, "lon": -1})[
                "zos_zostoga_regression_slope"
            ]
            for f in slope_files
        ]

        # Concatenate along a new dimension (representing the ensemble/models)
        slopes_stack = xr.concat(datasets, dim="model").chunk({"model": -1})

        # Read land mask if present
        mask_files = sorted(
            Path(self.patterns_dir).glob("*/zos_mask_*85_*.nc"), key=lambda p: p.name
        )

        if mask_files:
            if len(slope_files) == len(mask_files):
                self.land_mask_present = True

                datasets_mask = [
                    xr.open_dataset(f, chunks={"lat": -1, "lon": -1})["zos_mask"]
                    for f in mask_files
                ]
                mask_stack = xr.concat(datasets_mask, dim="model").chunk({"model": -1})
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

    def _load_patterns(self, state: SpatialState) -> da.Array:
        """
        Load the CMIP regression patterns at the target grid or sites.

        Parameters
        ----------
        state: SpatialState | LocalState
            Target grid or sites.

        Returns
        -------
        da.Array
            Regression slopes (metres of local change per metre of global
            thermal expansion), shape (n_models, *spatial). Land points are
            NaN where land masks are provided.
        """
        coeffs_da, mask_da = self._load_CMIP6_slopes()

        if self.land_mask_present:  # apply land mask
            coeffs_da = coeffs_da.where(mask_da == 0.0)

        return self.extract_spatial(coeffs_da, state).data

    def project_separable(
        self, state: SpatialState, rng: np.random.Generator
    ) -> SeparableProjection:
        """
        Global thermal expansion of the sampled members times CMIP patterns.

        Parameters
        ----------
        state: SpatialState | LocalState
            Target grid or sites and sampled member indices.
        rng: np.random.Generator
            Random number generator for drawing a CMIP model per member if
            sample_spatial is True.

        Returns
        -------
        SeparableProjection
            Members (members, time) and patterns (n_models, *spatial).
        """
        patterns, pattern_index = self.choose_patterns(
            self._load_patterns(state), state, rng, self.sample_spatial
        )
        return SeparableProjection(
            temporal=self.sample_global_members(state),
            patterns=patterns,
            pattern_index=pattern_index,
        )


class SterodynamicCMIP5(SpatialComponent):
    """
    Placeholder for a sterodynamic SLR component based on CMIP5 patterns.

    Not implemented yet: constructing it raises ``NotImplementedError`` so that
    it cannot silently contribute zeros to a projection. Use
    :class:`SterodynamicCMIP6` instead.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "SterodynamicCMIP5 is a placeholder and is not implemented yet. "
            "Use SterodynamicCMIP6 instead."
        )

    @property
    def global_projection(self):
        raise NotImplementedError("SterodynamicCMIP5 is not implemented yet.")

    def project(self, state, rng) -> da.Array:
        raise NotImplementedError("SterodynamicCMIP5 is not implemented yet.")
