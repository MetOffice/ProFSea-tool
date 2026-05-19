from pathlib import Path
from rich.progress import track
import numpy as np
from scipy.signal import fftconvolve
import xarray as xr


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
        t_arr = np.arange(n_time)

        # Two slow coeffs
        alphas1 = params[:, 0]
        alphas2 = params[:, 1]

        # Base forcing term to power of gamma, but preserving sign
        forcing_base = np.sign(tas) * (np.abs(tas) ** gamma)

        # Impulse response function for timescale 1
        decay_factors1 = (t_arr * dt / tau1**2) * np.exp(-t_arr * dt / tau1) * dt
        rate_delayed1 = fftconvolve(forcing_base, decay_factors1, mode="full")[:n_time]
        t_conv1 = np.cumsum(rate_delayed1, axis=0) * dt
        term_slow1 = alphas1[:, None] * t_conv1[None, :]

        # Impulse response function for timescale 2
        decay_factors2 = (t_arr * dt / tau2**2) * np.exp(-t_arr * dt / tau2) * dt
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
            # drift_coeffs = total_params[:, 3]
            # term_drift = drift_coeffs[:, None] * physical_time[None, :]

            all_preds[model, :, :] = term_fast + term_slow

        return all_preds
