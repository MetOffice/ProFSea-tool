from __future__ import annotations

import numpy as np

from profsea.components.core.base import Component
from profsea.components.core.global_model import ClimateState


class Glacier(Component):
    def __init__(self, glaciermip: int = 2):
        self.glaciermip = glaciermip

    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        """Project glacier contribution to GMSLR.

        Returns
        -------
        glacier: np.ndarray
            Glacier contribution to GMSLR.
        """
        tas = state.T_ens
        if tas.ndim > 2:
            tas = np.squeeze(tas)
        if tas.ndim == 1:
            tas = np.expand_dims(tas, axis=0)

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
                dict(name="GLIMB", factor=3.70, exponent=0.662, cvgl=0.206),
                dict(name="GloGEM", factor=4.08, exponent=0.716, cvgl=0.161),
                dict(name="JULES", factor=5.50, exponent=0.564, cvgl=0.188),
                dict(name="Mar-12", factor=4.89, exponent=0.651, cvgl=0.141),
                dict(name="OGGM", factor=4.26, exponent=0.715, cvgl=0.164),
                dict(name="RAD2014", factor=5.18, exponent=0.709, cvgl=0.135),
                dict(name="WAL2001", factor=2.66, exponent=0.730, cvgl=0.206),
            ]
        elif not self.glaciermip:
            glparm = [
                dict(name="Marzeion", factor=4.96, exponent=0.685, cvgl=0.20),
                dict(name="Radic", factor=5.45, exponent=0.676, cvgl=0.20),
                dict(name="Slangen", factor=3.44, exponent=0.742, cvgl=0.20),
                dict(name="Giesen", factor=3.02, exponent=0.733, cvgl=0.20),
            ]
        else:
            raise KeyError(
                "glaciermip must be False (AR5 parameters), 1 (Hock et al., 2019), or 2 (Marzeion et al., 2020)"
            )

        ngl = len(glparm)
        model_indices = rng.integers(0, ngl, size=(state.nt, state.num_members))

        base_factors = np.array([p["factor"] for p in glparm])
        base_exponents = np.array([p["exponent"] for p in glparm])
        base_cvgls = np.array([p["cvgl"] for p in glparm])

        factors = base_factors[model_indices][:, :, None]
        exponents = base_exponents[model_indices][:, :, None]
        cvgls = base_cvgls[model_indices][:, :, None]

        r = rng.standard_normal((state.nt, state.num_members))[:, :, None]

        T_int_ens_3d = state.T_int_ens[:, None, :]
        # Median shape: (1, 1, nyr)
        T_int_med_3d = state.T_int_med[None, None, :]

        zgl = self._project_glacier1(T_int_ens_3d, factors, exponents)

        # Passes (1, 1, nyr) + (nt, nm, 1) -> Returns (nt, nm, nyr)
        mgl = self._project_glacier1(T_int_med_3d, factors, exponents)

        # 6. Apply variance and clip using 3D matrix math
        glacier = zgl + (mgl * r * cvgls)
        glacier += dmz
        np.clip(glacier, None, glmass, out=glacier)

        # 7. Flatten to standard 2D output for easy summation
        return glacier.reshape(state.nt * state.num_members, state.nyr)

    def _project_glacier1(
        self, T_int: np.ndarray, factor: np.ndarray, exponent: np.ndarray
    ) -> np.ndarray:
        """Project glacier contribution by one glacier method."""
        scale = 1e-3  # mm to m
        # np.where works perfectly with n-dimensional broadcasting
        return scale * factor * (np.where(T_int < 0, 0, T_int) ** exponent)
