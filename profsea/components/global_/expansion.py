from __future__ import annotations

import numpy as np

from profsea.components.core.base import Component
from profsea.components.core.state import ClimateState
from profsea.utils import check_shapes


class ThermalExpansion(Component):
    """
    Parameters and Attributes
    -------------------------
    data_input: np.ndarray
        Array of ocean heat content (or thermosteric sea level) change values.
    distribution_scaler: float
        Controls distribution of sensitivity of thermosteric SLR to ocean heat content change.
    OHC_change: bool
        If True, assumes input is ocean heat content change.
        If False, assumes input is the global-mean sea level change.
    """

    def __init__(
        self,
        data_input: np.ndarray,
        distribution_scaler: float = 1.0,
        OHC_change: bool = True,
    ):

        self.data_input = data_input
        self.distribution_scaler = distribution_scaler
        self.OHC_change = OHC_change

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        self.data_input = np.asarray(self.data_input, dtype=state.dtype)

        # check the shape here
        check_shapes(self.data_input, state.nyr)

        # Ensure data_input is 2D
        if self.data_input.ndim > 2:
            self.data_input = np.squeeze(self.data_input)
        if self.data_input.ndim == 1:
            self.data_input = np.expand_dims(self.data_input, axis=0)

        if self.OHC_change:
            # Sensitivity of thermosteric SLR to ocean heat content change
            # From Turner et al. (2023)
            mean_eff = 0.113
            std_eff = 0.013 * self.distribution_scaler

            exp_efficiency = (
                rng.normal(
                    loc=mean_eff, scale=std_eff, size=(state.nt, state.num_members)
                )
                * 1e-24
            ).astype(state.dtype)  # m/YJ

            ohc_3d = self.data_input[:, None, :]
            # Efficiency shape: (nt, num_members, 1)
            exp_efficiency_3d = exp_efficiency[:, :, None]

            expansion = ohc_3d * exp_efficiency_3d

        else:
            expansion = np.broadcast_to(
                self.data_input[:, None, :], (state.nt, state.num_members, state.nyr)
            )

        return expansion.reshape(state.num_members * state.nt, state.nyr)
