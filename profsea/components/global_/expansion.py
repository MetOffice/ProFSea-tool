from __future__ import annotations

import numpy as np
import pdb

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
        If False, assumes input is thermosteric sea level change (e.g., zostoga) and skips OHC→SLR conversion.
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


    # Define a function called project that takes a ClimateState object and a random number generator
    #  and returns a np array    
    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        self.data_input = np.asarray(self.data_input, dtype=state.dtype)

        # check the shape here (climate_member, time)
        check_shapes(self.data_input, state.nyr)

        # Ensure data_input is 2D
        if self.data_input.ndim > 2:
            self.data_input = np.squeeze(self.data_input)
        if self.data_input.ndim == 1:
            self.data_input = np.expand_dims(self.data_input, axis=0)

        if self.OHC_change:
            # Sensitivity of thermosteric SLR to ocean heat content change
            # From IPCC AR6 WG1 Chapter 9: https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-9/#9.2.4.1
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


#class ThermalExpansion_via_zostoga(Component):
#    """
#    Parameters and Attributes
#    -------------------------
#        GTE: np.array
#        Array of global ocean thermal expansion (eg, zostoga)
#    """

#    # Initialise the object and store GTE_change as part of it to be used by other functions in the class:
#    def __init__(self, GTE_change: np.ndarray):
#        print("entered init")
#        self.GTE_change = GTE_change

#    # Define a function called project that takes a ClimateState object and a random number generator
#    #  and returns a np array:
#    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
#        print("entered project")
        

#        # check the shape here
#        check_shapes(self.GTE_change, state.nyr)

#        # Ensure GTE_change is 2D
#        if self.GTE_change.ndim > 2:
#            self.GTE_change = np.squeeze(self.GTE_change)
#        if self.GTE_change.ndim == 1:
#            self.GTE_change = np.expand_dims(self.GTE_change, axis=0)

        
#        return self.GTE_change

#    if state.nt > 1:
#        therm_med = sample_members_2D(self.zostoga_change, [50])
#        therm_std = np.std(self.zostoga_change, axis=0)
#    else:
#        therm_med = self.zostoga_change
#        therm_std = 0.0 * therm_med

        
#    z = rng.standard_normal(state.nt)

#    therm_ens = (
#        z[:, np.newaxis] * therm_std
#        + therm_med
#    )

