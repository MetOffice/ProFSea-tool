import numpy as np

from profsea.components.core.base import Component
from profsea.components.core.state import ClimateState
from profsea.utils import check_shapes, sample_members_2D


class ThermalExpansion(Component):
    """
    Parameters & Attributes
    ----------
    OHC_change: np.ndarray
        Array of ocean heat content change values.
    exp_efficiency: float
        Sensitivity of thermosteric SLR to ocean heat content change.
    """
    def __init__(self, OHC_change: np.ndarray, distribution_scaler: float = 1.0):
        self.OHC_change = OHC_change
        self.distribution_scaler = distribution_scaler

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        # check the shape here
        check_shapes(self.OHC_change, state.nyr)
        # Sensitivity of thermosteric SLR to ocean heat content change
        # From Turner et al. (2023)
        exp_efficiency = (
            rng.normal(loc=0.113, scale=0.013, size=state.nt)[:, None] * 1e-24
        )  # m/YJ
        z = rng.standard_normal(state.nt) * self.distribution_scaler

        if state.nt > 1: # when input_ensemble = True
            therm_med = sample_members_2D(self.OHC_change, [50]) * exp_efficiency
            therm_std = np.std(self.OHC_change, axis=0) * exp_efficiency
        else:
            therm_med = self.OHC_change * exp_efficiency
            therm_std = therm_med * 0. # dummary variable

        therm_ens = z[:, np.newaxis] * therm_std + therm_med
        expansion = np.tile(therm_ens, (state.num_members, 1))
        
        return expansion.reshape(state.num_members * state.nt, state.nyr)
