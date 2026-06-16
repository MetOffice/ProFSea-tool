from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from profsea.components.core.base import Component
from profsea.components.core.global_model import ClimateState
from profsea.components.core.time_projection import time_projection


@functools.lru_cache(maxsize=1)
def load_greenland_calibration():
    """Loads the CSV once and keeps it in memory."""
    path = Path(__file__).parents[2] / "aux_data" / "ISMIP_GIS_calibration.csv"
    return pd.read_csv(path)


class GreenlandAR6(Component):
    def __init__(self):
        self.df = load_greenland_calibration()

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """Project Greenland ice-sheet contribution to GMSLR.
        This follows the IPCC AR6 methodology as closely as possible.
        Projections are relative to 1996-2014 baseline.

        Returns
        -------
        np.ndarray
            Total GIS contribution to GMSLR.
        """
        tas = state.T_ens
        if tas.ndim > 2:
            tas = np.squeeze(tas)
        if tas.ndim == 1:
            tas = np.expand_dims(tas, axis=0)

        nt = tas.shape[0]
        time_delta = np.arange(state.nyr)

        df = self.df
        n_models = len(df)

        model_indices = rng.integers(0, n_models, size=(nt, state.num_members))

        # Extract parameters and reshape to 3D: (nt, nm, 1)
        b0 = df["b0"].values[model_indices][:, :, None]
        b1 = df["b1"].values[model_indices][:, :, None]
        b2 = df["b2"].values[model_indices][:, :, None]
        b3 = df["b3"].values[model_indices][:, :, None]
        b4 = df["b4"].values[model_indices][:, :, None]
        b5 = df["b5"].values[model_indices][:, :, None]

        # GIS trend values taken from FACTS GitHub repo
        trend_mean = 0.19
        trend_std = 0.1

        # Calculate trend contribution distribution
        a_bound = (0.0 - trend_mean) / trend_std
        b_bound = (99999.9 - trend_mean) / trend_std  # Or just np.inf
        trend = truncnorm.ppf(
            rng.random((nt, state.num_members)),
            a=a_bound,
            b=b_bound,
            loc=trend_mean,
            scale=trend_std,
        )
        trend_sle = (trend[:, :, None] * time_delta[None, None, :]) * 1e-3

        tas_3d = tas[:, None, :]

        # Calculate GIS contribution rate
        dsle = (
            b0
            + (b1 * tas_3d)
            + (b2 * tas_3d**2)
            + (b3 * tas_3d**3)
            + (b4 * time_delta[None, None, :])
            + (b5 * time_delta[None, None, :] ** 2)
        )

        # Now integrate
        sle_ens = np.cumsum(dsle, axis=2) * 1e-3  # convert from mm to m

        sle_ens += trend_sle

        # Persist 2100 rate of changeg
        if state.end_yr >= 2100:
            idx_2100 = 94
            rate = np.diff(sle_ens, axis=2)[:, :, idx_2100 - 1]
            sle_ens[:, :, idx_2100 + 1 :] = sle_ens[:, :, idx_2100 : idx_2100 + 1] + (
                rate[:, :, None] * time_delta[None, None, 1 : state.nyr - idx_2100]
            )

        sle_ens = sle_ens.reshape((state.num_members * state.nt, state.nyr))
        return sle_ens


class GreenlandSMBAR5(Component):
    """
    AR5 Greenland SMB contribution to GMSLR.
    """

    def __init__(self):
        self.fgreendyn = 0.5
        self.dgreen = (3.21 - 0.30) * 1e-3
        self.mSLEoGt = 1e12 / 3.61e14 * 1e-3

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """Project Greenland SMB contribution to GMSLR.

        Parameters
        ----------
        state: ClimateState
            State object containing relevant information for the projection.
        rng: np.random.Generator
            Random number generator.

        Returns
        -------
        greensmb: np.ndarray
            Greenland SMB contribution to GMSLR.

        """
        dtgreen = -0.146  # Delta_T of Greenland ref period wrt AR5 ref period
        fnlogsd = 0.4  # random methodological error of the log factor
        febound = [1, 1.15]  # bounds of uniform pdf of SMB elevation feedback factor

        # random log-normal factor
        fn = np.exp(rng.standard_normal(state.num_members) * fnlogsd)
        # elevation feedback factor
        fe = rng.random(state.num_members) * (febound[1] - febound[0]) + febound[0]
        ff = fn * fe

        ztgreen = state.T_ens - dtgreen

        greensmb = ff[:, np.newaxis, np.newaxis] * self._fettweis(ztgreen)

        if state.palmer_method and state.end_yr > state.endofAR5:
            greensmb[:, :, 95:] = greensmb[:, :, 94:95]

        greensmb = np.cumsum(greensmb, axis=-1)

        greensmb += (1 - self.fgreendyn) * self.dgreen

        greensmb = greensmb.reshape(
            greensmb.shape[0] * greensmb.shape[1], greensmb.shape[2]
        )
        return greensmb

    def _fettweis(self, ztgreen: np.ndarray) -> np.ndarray:
        """Calculate Greenland SMB in m yr-1 SLE from global mean temperature
        anomaly, using Eq 2 of Fettweis et al. (2013).

        Parameters
        ----------
        ztgreen: np.ndarray
            Global mean temperature anomaly.

        Returns
        -------
        np.ndarray
            Greenland SMB in m yr-1 SLE.
        """
        return (
            71.5 * ztgreen + 20.4 * (ztgreen**2) + 2.8 * (ztgreen**3)
        ) * self.mSLEoGt


class GreenlandDynAR5(Component):
    """
    AR5 Greenland ice-sheet dynamics contribution to GMSLR.

    NOTE: This is not scenario independent. It will run with either rcp85/ssp585 related projections,
    or will default to a temperature independent projection based on AR5.

    This is based on Jonathan Gregory's AR5 implmentation,
    which can be found at https://github.com/JonathanGregory/ar5gmslr

    """

    def __init__(
        self,
    ):
        self.fgreendyn = 0.5
        self.dgreen = (3.21 - 0.30) * 1e-3

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """Project Greenland rapid ice-sheet dynamics contribution to GMSLR.

        Parameters
        ----------
        state: ClimateState
            State object containing relevant information for the projection.
        rng: np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Greenland rapid ice-sheet dynamics contribution to GMSLR.
        """
        # For SMB+dyn during 2005-2010 Table 4.6 gives 0.63+-0.17 mm yr-1 (5-95% range)
        # For dyn at 2100 Chapter 13 gives [20,85] mm for rcp85, [14,63] mm otherwise
        if state.scenario in ["rcp85", "ssp585"]:
            finalrange = [0.020, 0.085]
        else:
            finalrange = [0.014, 0.063]
        return (
            time_projection(
                state, 0.63 * self.fgreendyn, 0.17 * self.fgreendyn, finalrange, rng
            )
            + self.fgreendyn * self.dgreen
        )
