from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pdb

import dask.array as da
import numpy as np
import xarray as xr

from profsea.components.core.base import SpatialComponent
from profsea.components.core.state import ClimateState
from profsea.utils import sample_members_2D

logging.basicConfig(level=logging.WARNING)

PROFSEA_DIR = Path(__file__).resolve().parents[2]
PATTERNS_DIR = PROFSEA_DIR / "profsea-assets" / "cmip6-patterns"

fig_path = "/home/users/lesley.allison/sealevel/profsea/profsea-v3/figs/"

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
        sample_spatial: bool = True,  # LA changed from F to T
    ) -> None:
        """
        Parameters
        ----------
        global_projection: xr.DataArray
             A 2D array (members x years) of global projections to apply the fingerprints to.  [LA: In this case it is the expansion component only?]
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


    def sample_pattern_indices(
        self,
        rng: np.random.Generator,
        n_local_members: int,
    ) -> np.ndarray:
        """
        Sample one CMIP6 sterodynamic pattern for each local MC member.

        The returned indices should be generated once per Spatial.run()
        and reused for every spatial tile.

        This method will discover how many models are available.
        """
        coeffs_da, _ = self._load_CMIP6_slopes()

        n_patterns = coeffs_da.sizes["model"]

        return rng.choice(
            n_patterns,
            size=n_local_members,
            replace=True,
        )

    
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

        model_names = [Path(f).stem.split("_")[-1] for f in slope_files]
        
        
        # Lazily load all files keeping metadata intact and now also preserving the model names
        datasets = [xr.open_dataset(f, chunks={"lat": 45, "lon": 45})["zos_zostoga_regression_slope"].expand_dims(model=[Path(f).stem.split("_")[-1]]) for f in slope_files]

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
                    xr.open_dataset(f, chunks={"lat": 45, "lon": 45})["zos_mask"].expand_dims(model=[Path(f).stem.split("_")[-1]])
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

#    def _calc_expansion_contribution(
#        self, rng: np.random.Generator, state: ClimateState
#    ) -> da.Array:
    def _calc_expansion_contribution(
            self,
            rng: np.random.Generator,
            state: ClimateState,
            lat_slice: slice | None = None,
            lon_slice: slice | None = None,
    ) -> da.Array:
    # LA for localMC - allow _calc_expansion_contribution() to receice optional tile slice arguments
        
        """
        Calculate the thermal expansion contribution to the regional component of
        sea level rise.  --> LA: make this description more accurate.

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
        
        apply_mask=True
        plot_nwes=False
        
        # Select slope coefficients based on the MIP
        coeffs_da, mask_da = self._load_CMIP6_slopes()

        if apply_mask and self.land_mask_present:  # apply land mask
            coeffs_da = coeffs_da.where(mask_da == 0.0)

        # Get the data either at sites or on a grid
        #interp_da = self.extract_spatial(coeffs_da, state)
        # Temporary test of first 5 x 5 spatial tile for local MC (temporary because Spatial.run() will suuply these slices eventually)
#        interp_da = self.extract_spatial(
#            coeffs_da,
#            state,
#            lat_slice=slice(0, 5),
#            lon_slice=slice(0, 5),
#        )
        interp_da = self.extract_spatial(
            coeffs_da,
            state,
            lat_slice=lat_slice,
            lon_slice=lon_slice,
        )   # LA for local MC -> pass spatial tile slice to extract_spatial


        coeffs = interp_da.data
        spatial_shape = coeffs.shape[1:]  # either (lat, lon) or (site,)


        #-------------------------------------------------------------------
        # Plotting maps of regression coeffients for all models
        n_models = interp_da.sizes["model"]
        
        if plot_nwes:
            NW_Eur_Shelf_da = interp_da.sel(lon=slice(-15, 15), lat=slice(40, 70))
            plot_data = NW_Eur_Shelf_da
        else:
            plot_data = interp_da
        
        # Choose subplot layout
        ncols = 5
        nrows = int(np.ceil(n_models / ncols))

        # Common colour scale across all models
        vmin = -2.0
        vmax = 2.0

        fig, axs = plt.subplots(
            nrows,
            ncols,
            figsize=(3*ncols, 2.5*nrows),
            constrained_layout=True
        )

        axs = np.ravel(axs)
        

        
        for i in range(n_models):
            im = plot_data[i].plot(
                ax=axs[i],
                add_colorbar=False,
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu_r"
            )
            axs[i].axis("off")
            axs[i].set_title(str(interp_da.model.values[i]),fontsize=7,pad=2)
            fig.subplots_adjust(wspace=0.02, hspace=0.15)
            
        # Hide unused axes
        for ax in axs[n_models:]:
            ax.set_visible(False)

        # One shared colourbar
        fig.colorbar(
            im,
            ax=axs[:n_models],
            shrink=0.8,
            label="(zos + zostoga)/zostoga Regression coefficient"
        )

        if plot_nwes:
            if apply_mask:
                figname = fig_path+"CMIP6_NWES_reg_coeffs.png"
            else:
                figname = fig_path+"CMIP6_NWES_reg_coeffs_before_mask.png"
        else:
            if apply_mask:
                figname = fig_path+"CMIP6_reg_coeffs.png"
            else:
                figname = fig_path+"CMIP6_reg_coeffs_before_mask.png"

        fig.savefig(figname, dpi=300, bbox_inches="tight")
    
        plt.show()
        #-------------------------------------------------------------------


        if self.sample_spatial:
            
           ## Randomly sample n_members CMIP6 model coefficient fields, with replacement.
           ## LA:  Here it is returning a 3D field, which has dimensions (model, lat, lon).
           ##      In the original version, size=state.n_members (which is the number of percentiles
           ##      that are specified for output (ie, [5, 17, 50, 83, 95]),
           #rand_samples = rng.choice(
            #    coeffs.shape[0], size=state.n_members, replace=True
            #)
            
            # Randomly sample CMIP6 model coefficient fields, with replacement.
            # LA: update for local MC. Use n_local_members for the size of the local Monte Carlo.
            #rand_samples = rng.choice(
            #    coeffs.shape[0],
            #    size=state.n_local_members,
            #    replace=True,
            #)
            #return coeffs[rand_samples, :, :]
            if state.sterodynamic_pattern_indices is None:
                raise RuntimeError(
                    "Sterodynamic pattern indices have not been sampled."
                )
            return coeffs[state.sterodynamic_pattern_indices,:,:,]  # each Monte Carlo member has a spatially coherent CMIP6 pattern
        else:
            # Calc pattern ensemble mean
            mean_coeff = da.nanmean(coeffs, axis=0)
            # LA:  Here is is returning a 3D field which is the 2D multi model mean of the regression coefficients
            #      broadcast to have an extra "member" dimension.
            #      Updated to use n_local_members instead of n_members (the local Monte Carlo instead of 5 percentiles).
            return da.broadcast_to(
                mean_coeff,
                (state.n_local_members, *spatial_shape),
            )



        
#    def project(self, state: ClimateState, rng) -> np.ndarray:
    def project(
            self,
            state: ClimateState,
            rng,
            lat_slice: slice | None = None,
            lon_slice: slice | None = None,
    ) -> np.ndarray:
        # LA for localMC, added optional tile arguments 
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
#        # Calculate percentiles locally without mutating self
#        # HERE THE 450000 member global ensemble is being reduced to percentiles
#        if state.output_percentiles is not None:
#            current_projection = sample_members_2D(
#                self.global_projection, state.output_percentiles
#            )
#        else:
#            current_projection = self.global_projection

        # LA: for the local Monte Carlo route, select the same global ensemble
        # rows that were chosen once in Spatial.run().
        if state.global_member_indices is not None:
            current_projection = self.global_projection[
                state.global_member_indices, :
            ]
        elif state.output_percentiles is not None:
            # Retain the old percentile-representative pathway as a fallback. Probably want to delete this eventually.
            current_projection = sample_members_2D(
                self.global_projection,
                state.output_percentiles,
            )
        else:
            current_projection = self.global_projection

            
        #expansion_contribution = self._calc_expansion_contribution(rng, state)
        expansion_contribution = self._calc_expansion_contribution(
            rng,
            state,
            lat_slice=lat_slice,
            lon_slice=lon_slice,
        )   # LA for localMC --> pass tile slices to _calc_expansion_contribution
        
        #-----------------------------------------------------------------------------
        # Compute and plot the (currently 5) members of expansion_contribution...

        # Skip this plotting step to avoid computing whole 100,000-member array

        plot_expansion_contribution = False
        
        if plot_expansion_contribution:
            
            arr = expansion_contribution[:5].compute()   # as it was a lazy dask array before now  (added the [:5] duing local MC code changes)

            # Same colour scale for all members
            vmin = -3
            vmax = 3

            fig, axs = plt.subplots(
                2, 3,
                figsize=(12, 6),
                constrained_layout=True
            )

            axs = axs.ravel()

            for i in range(5):
                im = axs[i].imshow(
                    arr[i],
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="RdBu_r",
                    aspect="auto"
                )

                axs[i].set_title(f"Member {i}")
                axs[i].set_xticks([])
                axs[i].set_yticks([])

            # Hide unused sixth panel
            axs[5].axis("off")

            # One colourbar shared by all panels
            fig.colorbar(
                im,
                ax=axs[:5].tolist(),
                orientation="vertical",
                label="Expansion contribution"
            )

            figname = fig_path+"expansion_contribution_near_end_of_project_SterodynamicCMIP6_class.png"
            fig.savefig(figname, dpi=300, bbox_inches="tight")

            plt.close(fig)

        result = self.broadcast_spatiotemporal(current_projection, expansion_contribution)
        # result should be a dask array with shape (state.n_local_members, state.n_years, len(state.grid_lats), len(state.grid_lons))

        return result


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
