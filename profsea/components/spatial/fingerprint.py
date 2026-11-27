from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from profsea.components.core.base import SeparableProjection, SpatialComponent
from profsea.components.core.state import SpatialState

PROFSEA_DIR = Path(__file__).resolve().parents[2]
FP_DIR = PROFSEA_DIR / "profsea-assets" / "grd-fingerprints"

# Default fingerprint files for each component. Where a component has several
# GRD models, they are listed in the same order everywhere (Klemann, Slangen,
# Spada): members choose a model by position, so a member gets the same GRD
# model in every component (as in Palmer et al., 2020).
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
    """Regional pattern of a barystatic (land ice or land water) contribution.

    Multiplies each member's global contribution (e.g. Greenland mass loss, in
    metres of global mean sea level) by a GRD fingerprint: the local sea-level
    change per metre of global mean change, from the gravitational, rotational
    and solid-Earth deformation response to that mass change.
    """

    def __init__(
        self,
        global_projection: xr.DataArray,
        fingerprint_component: str,
        fingerprint_paths: str | Path | list[str | Path] = None,
        scaling_factor: float = 1.0,
        sample_spatial: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        global_projection: xr.DataArray
            Full global ensemble of this contribution from ``Global``, shape
            (climate_member, process_member, time), or (members, time).
        fingerprint_component: str
            Which default fingerprints to use, a key of ``FP_PATH_MAP`` (e.g.
            "glacier", "greenland", "wais"). Ignored for the defaults when
            ``fingerprint_paths`` is given.
        fingerprint_paths: str, Path, or list of str/Path, optional
            NetCDF files with fingerprints, one GRD model per file, each a
            DataArray with dimensions (lat, lon). Overrides the defaults.
        scaling_factor: float, optional
            Multiplier applied to the fingerprints (e.g. to convert units).
            Default is 1.0.
        sample_spatial: bool, optional
            If True, each member uses one of the fingerprints, chosen with the
            state's shared fingerprint fraction: a member uses the same GRD
            model (the same position in the file list) in every Fingerprint
            component, independently of which global member it is. If False,
            every member uses the mean of the fingerprints ("storyline" mode).
            Default is True.
        """

        self.fetch_assets()
        self._set_global_projection(global_projection)
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

    def _load_and_interpolate(self, state: SpatialState) -> da.Array:
        """
        Lazily load the fingerprints at the target grid or sites.

        Parameters
        ----------
        state: SpatialState | LocalState
            Target grid or sites.

        Returns
        -------
        da.Array
            Fingerprints in file order, scaled, shape (n_fingerprints, *spatial).
        """
        grids = []
        for path in self.fp_paths:
            fp_da = xr.open_dataarray(path, chunks={"lat": -1, "lon": -1})
            fp_interp = self.extract_spatial(fp_da, state)
            grids.append(fp_interp.data * self.scaling_factor)

        # Stack them into a 3D array: (n_fingerprints, lat, lon)
        return da.stack(grids, axis=0)

    def project_separable(
        self, state: SpatialState, rng: np.random.Generator
    ) -> SeparableProjection:
        """
        Global time series of the sampled members times their fingerprints.

        Parameters
        ----------
        state: SpatialState | LocalState
            Target grid or sites, sampled member indices and fingerprint fraction.
        rng: np.random.Generator
            Not used: fingerprint choices come from the shared fingerprint
            fraction, so they match across Fingerprint components.

        Returns
        -------
        SeparableProjection
            Members (members, time) and fingerprints (n_fingerprints, *spatial).
        """
        fingerprints = self._load_and_interpolate(state)
        patterns, pattern_index = self.choose_patterns(
            fingerprints,
            state,
            rng,
            self.sample_spatial,
            fraction=state.fingerprint_fraction,
        )
        return SeparableProjection(
            temporal=self.sample_global_members(state),
            patterns=patterns,
            pattern_index=pattern_index,
        )
