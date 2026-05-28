import functools

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import truncnorm

from profsea.components.core.base import Component
from profsea.components.core.global_model import ClimateState

class Glacier(Component):

    def __init__(self, glaciermip: int=2):
        self.glaciermip = glaciermip
    
    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """Project glacier contribution to GMSLR.

        Returns
        -------
        glacier: np.ndarray
            Glacier contribution to GMSLR.
        """

        # glaciermip -- False => AR5 parameters, 1 => fit to Hock et al. (2019),
        #   2 => fit to Marzeion et al. (2020)
        dmzdtref = 0.95  # mm yr-1 in Marzeion's CMIP5 ensemble mean for AR5 ref period
        dmz = (
            dmzdtref * (state.endofhistory - 1996) * 1e-3
        )  # m from glacier at start wrt AR5 ref period
        glmass = 412.0 - 96.3  # initial glacier mass, used to set a limit, from Tab 4.2
        glmass = 1e-3 * glmass  # m SLE

        if self.glaciermip == 1:
            glparm = [
                dict(name="SLA2012", factor=3.39, exponent=0.722, cvgl=0.15),
                dict(name="MAR2012", factor=4.35, exponent=0.658, cvgl=0.13),
                dict(name="GIE2013", factor=3.57, exponent=0.665, cvgl=0.13),
                dict(name="RAD2014", factor=6.21, exponent=0.648, cvgl=0.17),
                dict(name="GloGEM", factor=2.88, exponent=0.753, cvgl=0.13),
            ]
        elif self.glaciermip == 2:
            glparm = [
                dict(name="GLIMB",   factor=3.70, exponent=0.662, cvgl=0.206),
                dict(name="GloGEM",  factor=4.08, exponent=0.716, cvgl=0.161),
                dict(name="JULES",   factor=5.50, exponent=0.564, cvgl=0.188),
                dict(name="Mar-12",  factor=4.89, exponent=0.651, cvgl=0.141),
                dict(name="OGGM",    factor=4.26, exponent=0.715, cvgl=0.164),
                dict(name="RAD2014", factor=5.18, exponent=0.709, cvgl=0.135),
                dict(name="WAL2001", factor=2.66, exponent=0.730, cvgl=0.206),
            ]
        elif not self.glaciermip:
            glparm = [
                dict(name="Marzeion", factor=4.96, exponent=0.685, cvgl=0.20),
                dict(name="Radic",    factor=5.45, exponent=0.676, cvgl=0.20),
                dict(name="Slangen",  factor=3.44, exponent=0.742, cvgl=0.20),
                dict(name="Giesen",   factor=3.02, exponent=0.733, cvgl=0.20),
            ]
        else:
            raise KeyError("glaciermip must be False (AR5 parameters), 1 (Hock et al., 2019), or 2 (Marzeion et al., 2020)")

        ngl = len(glparm)
        r = rng.standard_normal(state.num_members)[:, np.newaxis, np.newaxis]
        glacier = np.full((state.num_members, state.nt, state.nyr), np.nan)

        r_per_model = state.num_members // ngl
        r_remainder = state.num_members % ngl

        # Precompute mgl and zgl for all glacier methods
        mgl_all = np.array(
            [
                self._project_glacier1(
                    state.T_int_med, glparm[igl]["factor"], glparm[igl]["exponent"]
                )
                for igl in range(ngl)
            ]
        )
        zgl_all = np.array(
            [
                self._project_glacier1(
                    state.T_int_ens, glparm[igl]["factor"], glparm[igl]["exponent"]
                )
                for igl in range(ngl)
            ]
        )
        cvgl_all = np.array(
            [glparm[igl]["cvgl"] for igl in range(ngl)]
        )

        # Make an ensemble of projections for each method
        current_ensemble_idx = 0
        for igl in range(ngl):
            mgl = mgl_all[igl]
            zgl = zgl_all[igl]
            cvgl = cvgl_all[igl]

            num_reals_for_model_i = (
                r_per_model + 1 if igl < r_remainder else r_per_model
            )
            ifirst = current_ensemble_idx
            ilast = current_ensemble_idx + num_reals_for_model_i

            glacier[ifirst:ilast, ...] = zgl + (mgl * r[ifirst:ilast] * cvgl)
            current_ensemble_idx = ilast

        glacier += dmz
        np.clip(glacier, None, glmass, out=glacier)

        glacier = glacier.reshape(glacier.shape[0] * glacier.shape[1], glacier.shape[2])
        
        return glacier

    def _project_glacier1(self, T_int: np.ndarray, factor: float, 
                          exponent: float) -> np.ndarray:
        """Project glacier contribution by one glacier method.

        Parameters
        ----------
        T_int: np.ndarray
            Time-integral temperature anomaly timeseries.
        factor: float
            Factor for the glacier method.
        exponent: float
            Exponent for the glacier method.

        Returns
        -------
        np.ndarray
            Projection of glacier contribution.
        """
        scale = 1e-3  # mm to m
        
        return scale * factor * (np.where(T_int < 0, 0, T_int) ** exponent)
