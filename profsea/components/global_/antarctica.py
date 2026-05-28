from pathlib import Path
from rich.progress import track
import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import norm
import xarray as xr

from profsea.components.core.base import Component
from profsea.components.core.global_model import ClimateState
from profsea.components.core.time_projection import time_projection


class AntarcticaISMIP6:
    """
    ISMIP6 2300 Antarctic ice-sheet emulator with two-timescale response.

    This emulation of the ISMIP6 ice-sheet model ensemble aims to capture
    slow, fast and drift responses of the Antarctic ice sheet to GMST.
    The slow response is modelled as the impulse response to temperature
    forcing, convolved with two exponential decay kernels representing
    different ice-sheet response timescales, depending on the region being modelled.
    The fast response is modelled as a direct proportionality to the integrated
    temperature anomaly, while the drift term captures any linear time-dependent
    trends not explained by the temperature forcing.

    Parameters provided are for the WAIS, EAIS and AIS Peninsula regions, since each has
    different response characteristics to warming.
    """

    def __init__(self, params_path: Path, samples: int = 1000):
        self.param_ds = xr.load_dataset(params_path)
        self.n_samples = samples
        self.n_models = self.param_ds.coords["model"].shape[0]

    def _impulse_response_term(
        self,
        tas: np.ndarray,
        tau1: float,
        tau2: float,
        gamma: float,
        params: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Computes the slow response term using convolution with two exponential decay kernels.
        Parameters
        ----------
        tas : np.ndarray
            Time series of global mean surface air temperature anomalies (shape: n_time).
        tau1 : float
            Timescale of the first response component (in years).
        tau2 : float
            Timescale of the second response component (in years).
        gamma : float
            Exponent for the temperature forcing term.
        params : np.ndarray
            Array of shape (n_samples, 4) containing [alpha1, alpha2, beta, drift] parameters for each sample.
        dt : float
            Time step in years.
        Returns
        -------
        np.ndarray
            Slow response term for each sample (shape: n_samples x n_time).
        """
        n_time = tas.shape[0]

        # Two slow coeffs
        alphas1 = params[:, 0]
        alphas2 = params[:, 1]

        # Base forcing term to power of gamma, but preserving sign
        forcing_base = np.sign(tas) * (np.abs(tas) ** gamma)

        # Impulse response function for timescale 1
        decay_factors1 = np.exp(-np.arange(n_time) * dt / tau1) * (dt / tau1)
        rate_delayed1 = fftconvolve(forcing_base, decay_factors1, mode="full")[:n_time]
        t_conv1 = np.cumsum(rate_delayed1, axis=0) * dt
        term_slow1 = alphas1[:, None] * t_conv1[None, :]

        # Impulse response function for timescale 2
        decay_factors2 = np.exp(-np.arange(n_time) * dt / tau2) * (dt / tau2)
        rate_delayed2 = fftconvolve(forcing_base, decay_factors2, mode="full")[:n_time]
        t_conv2 = np.cumsum(rate_delayed2, axis=0) * dt
        term_slow2 = alphas2[:, None] * t_conv2[None, :]

        return term_slow1 + term_slow2  # Slow terms are linearly combined

    def predict(
        self, tas: np.ndarray, dt: float = 1.0, display_progress=True, seed=None
    ) -> np.ndarray:
        """
        Projects AIS response using empirical additive bootstrapping.
        Parameters
        ----------
        tas : np.ndarray
            Time series of global mean surface air temperature anomalies (shape: n_time).
        dt : float, optional
            Time step in years (default: 1.0).
        display_progress : bool, optional
            Whether to show a progress bar during prediction (default: True).
        seed : int or None, optional
            Random seed for reproducibility (default: None).
        Returns
        -------
        np.ndarray
            Projected AIS contributions (shape: n_models x n_samples x n_time).
        """
        tas = np.squeeze(tas)
        n_time = len(tas)
        physical_time = np.arange(n_time) * dt

        # RNG
        rng = np.random.default_rng(seed)

        # Integrated temperature term
        tas_int = np.cumsum(tas) * dt

        # Output shape: (n_models, n_samples, n_time)
        all_preds = np.zeros((self.n_models, self.n_samples, n_time))

        # Shape assumed: (n_models, n_training_scenarios, 4)
        all_residuals = self.param_ds.param_residuals.values
        n_train_scenarios = all_residuals.shape[1]

        for model in track(
            range(self.n_models),
            description="Projecting AIS response... ",
            disable=not display_progress,
        ):
            # Unpack model-specific parameters
            tau1 = float(self.param_ds.tau1[model].values)
            tau2 = float(self.param_ds.tau2[model].values)
            gamma = float(self.param_ds.gamma[model].values)

            # Unpack general parameters [alpha1, alpha2, beta, drift]
            general_p = self.param_ds.general_params[model].values

            # Sample random parameter residuals for this model
            rand_indices = rng.integers(0, n_train_scenarios, size=self.n_samples)
            sampled_residuals = all_residuals[model, rand_indices, :]

            # Add sampled parameter residuals to general parameters
            total_params = general_p + sampled_residuals

            # Slow response term
            term_slow = self._impulse_response_term(
                tas, tau1, tau2, gamma, total_params, dt
            )

            # Fast response term
            betas = total_params[:, 2]
            term_fast = betas[:, None] * tas_int[None, :]

            # Drift term
            drift_coeffs = total_params[:, 3]
            term_drift = drift_coeffs[:, None] * physical_time[None, :]

            all_preds[model, :, :] = term_fast + term_slow + term_drift

        return all_preds


class AntarcticaDynAR5(Component):
    """
    AR5 Antarctic ice-dynamics response to warming, as a function of
    cumulative emissions or scenario. Following the implementation
    as given by Jonathan Gregory's ar5gmslr (https://github.com/JonathanGregory/ar5gmslr).

    Requires cumulative emissions total to be specified for a given scenario,
    or will default to scenario-based regression, based on rcp scenarios.
    """

    def __init__(
        self, d_ant: float = (2.37 + 0.13) * 1e-3, cum_emissions_total: float = None
    ):
        self.d_ant = d_ant
        self.cum_emissions_total = cum_emissions_total

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """Project Antarctic rapid ice-sheet dynamics contribution to GMSLR.

        Parameters
            ----------
            fraction: np.ndarray
            Random numbers for the dynamic contribution.

        Returns
        -------
        np.ndarray
            Antarctic rapid ice-sheet dynamics contribution to GMSLR.
        """
        # This is a naive solution to calculating the AntDyn contribution
        # for any given scenario. Basically linear regressions through existing data
        # to find rough relationship between cumulative emissions and AntDyn contribution.
        if self.cum_emissions_total:
            upper = (0.000110 * self.cum_emissions_total) + 0.375  # in metres
            lower = (1.363e-05 * self.cum_emissions_total) + 0.0392  # in metres
            final = [lower, upper]
        else:
            lcoeff = dict(
                rcp26=[-2.881, 0.923, 0.000],
                rcp45=[-2.676, 0.850, 0.000],
                rcp60=[-2.660, 0.870, 0.000],
                rcp85=[-2.399, 0.860, 0.000],
            )
            lcoeff = lcoeff[state.scenario]

            ascale = norm.ppf(state.fraction)
            final = np.exp(lcoeff[2] * ascale**2 + lcoeff[1] * ascale + lcoeff[0])
            final = final.reshape(state.num_members, state.nt)
        return (
            time_projection(state, 0.41, 0.20, final, rng, fraction=state.fraction)
            + self.d_ant
        )


class AntarcticaSMBAR5(Component):
    """
    AR5 Antarctic SMB contribution to GMSLR, as a function of global
    mean surface temperature change.

    Following the implementation as given by Jonathan Gregory's ar5gmslr
    (https://github.com/JonathanGregory/ar5gmslr).
    """

    def __init__(self):
        # Conversion factor for Gt to m SLE
        self.mSLEoGt = 1e12 / 3.61e14 * 1e-3

    def project(
        self,
        state: ClimateState,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Project Antarctic SMB contribution to GMSLR.

        Parameters
        ----------
        T_int_ens: np.ndarray
            Ensemble of time-integral temperature anomaly timeseries.
        fraction: np.ndarray
            Random numbers for the SMB-dynamic feedback.

        Returns
        -------
        antsmb: np.ndarray
            Antarctic SMB contribution to GMSLR.
        """
        # The following are [mean,SD]
        pcoK = [5.1, 1.5]  # % change in Ant SMB per K of warming from G&H06
        KoKg = [1.1, 0.2]  # ratio of Antarctic warming to global warming from G&H06

        # Generate a distribution of products of the above two factors
        pcoKg = (pcoK[0] + rng.standard_normal([state.num_members, state.nt]) * pcoK[1]) * (
            KoKg[0] + rng.standard_normal([state.num_members, state.nt]) * KoKg[1]
        )
        meansmb = 1923  # model-mean time-mean 1979-2010 Gt yr-1 from 13.3.3.2
        moaoKg = (
            -pcoKg * 1e-2 * meansmb * self.mSLEoGt
        )  # m yr-1 of SLE per K of global warming

        if state.fraction is None:
            fraction = rng.random((state.num_members, state.nt))
        elif state.fraction.size != state.num_members * state.nt:
            raise ValueError("fraction is the wrong size")
        else:
            fraction = state.fraction.reshape((state.num_members, state.nt))

        smax = 0.35  # max value of S in 13.SM.1.5
        ainterfactor = 1 - fraction * smax

        z = moaoKg * ainterfactor
        z = z[:, :, np.newaxis]
        antsmb = z * state.T_int_ens
        antsmb = antsmb.reshape(antsmb.shape[0] * antsmb.shape[1], antsmb.shape[2])
        return antsmb
