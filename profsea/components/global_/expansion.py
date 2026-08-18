from __future__ import annotations

import numpy as np

from profsea.components.core.base import Component
from profsea.components.core.state import ClimateState
from profsea.utils import check_shapes


class ThermalExpansion(Component):
    """
    Parameters and Attributes
    -------------------------
    OHC_change: np.ndarray
        Array of ocean heat content change values.
    exp_efficiency: float
        Sensitivity of thermosteric SLR to ocean heat content change.
    """

    def __init__(self, OHC_change: np.ndarray, distribution_scaler: float = 1.0):
        self.OHC_change = OHC_change
        self.distribution_scaler = distribution_scaler

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        self.OHC_change = np.asarray(self.OHC_change, dtype=state.dtype)

        # check the shape here (climate_member, time)
        check_shapes(self.OHC_change, state.n_years)

        # Ensure OHC_change is 2D
        if self.OHC_change.ndim > 2:
            self.OHC_change = np.squeeze(self.OHC_change)
        if self.OHC_change.ndim == 1:
            self.OHC_change = np.expand_dims(self.OHC_change, axis=0)

        # Sensitivity of thermosteric SLR to ocean heat content change
        # From Turner et al. (2023)
        mean_eff = 0.113
        std_eff = 0.013 * self.distribution_scaler

        exp_efficiency = (
            rng.normal(
                loc=mean_eff, scale=std_eff, size=(state.nt, state.num_members)
            )  # (climate_member, process_member)
            * 1e-24
        ).astype(state.dtype)  # m/YJ

        ohc_3d = self.OHC_change[:, None, :]  # (climate_member, 1, time)
        # Efficiency shape: (nt, num_members, 1)
        exp_efficiency_3d = exp_efficiency[
            :, :, None
        ]  # (climate_member, process_member, 1)

        expansion = ohc_3d * exp_efficiency_3d  # (climate_member, process_member, time)

        return expansion
