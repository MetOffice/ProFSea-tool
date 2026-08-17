import numpy as np
import pytest
import xarray as xr

from profsea.components.core.state import ClimateState
from profsea.components.global_.antarctica import (
    AntarcticaDynAR5,
    AntarcticaISMIP6,
    AntarcticaSMBAR5,
)


def get_dummy_state(
    T_change_val: float = 1.0,
    scenario: str = "rcp45",
    fraction: np.ndarray | None = None,
) -> ClimateState:
    """Helper to generate a small ClimateState with constant temperature."""
    T_ens = np.ones((2, 4), dtype=np.float32) * T_change_val
    T_int_ens = np.cumsum(T_ens, axis=1)
    T_int_med = np.cumsum(np.median(T_ens, axis=0))

    if fraction is None:
        fraction = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)

    return ClimateState(
        scenario=scenario,
        T_ens=T_ens,
        T_int_ens=T_int_ens,
        T_int_med=T_int_med,
        fraction=fraction,
        palmer_method=True,
        endofAR5=2100,
        endofhistory=2006,
        end_yr=2010,
        nyr=4,
        nt=2,
        num_members=2,
    )


def get_dummy_ismip6() -> AntarcticaISMIP6:
    """Create an ISMIP6 component without reading auxiliary NetCDF files."""
    antarctica = AntarcticaISMIP6.__new__(AntarcticaISMIP6)

    antarctica.param_ds = xr.Dataset(
        data_vars={
            "tau1": ("model", np.array([2.0, 3.0], dtype=np.float32)),
            "tau2": ("model", np.array([5.0, 6.0], dtype=np.float32)),
            "gamma": ("model", np.array([1.0, 1.5], dtype=np.float32)),
            "general_params": (
                ("model", "parameter"),
                np.array(
                    [
                        [0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                    ],
                    dtype=np.float32,
                ),
            ),
            "param_residuals": (
                ("model", "training_scenario", "parameter"),
                np.zeros((2, 2, 3), dtype=np.float32),
            ),
        },
        coords={
            "model": [0, 1],
            "training_scenario": [0, 1],
            "parameter": ["alpha1", "alpha2", "beta"],
        },
    )

    antarctica.n_models = 2

    return antarctica


def test_ismip6_invalid_region():
    """Invalid Antarctic calibration regions should be rejected."""
    with pytest.raises(ValueError, match="Invalid region calibration"):
        AntarcticaISMIP6("invalid")


def test_ismip6_zero_temperature_zero_projection():
    """No temperature anomaly should produce no ISMIP6 contribution."""
    antarctica = get_dummy_ismip6()
    state = get_dummy_state(T_change_val=0.0)
    rng = np.random.default_rng(42)

    projection = antarctica.project(state, rng)

    assert projection.shape == (state.nt, state.num_members, state.nyr)
    np.testing.assert_allclose(projection, 0.0, atol=1e-7)


def test_ismip6_projection_shape():
    """ISMIP6 output should contain one timeseries per ensemble member."""
    antarctica = get_dummy_ismip6()
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    projection = antarctica.project(state, rng)

    assert projection.shape == (
        state.nt,
        state.num_members,
        state.T_ens.shape[1],
    )


def test_ismip6_reproducible_with_same_seed():
    """Random parameter sampling should be deterministic for a fixed RNG seed."""
    antarctica = get_dummy_ismip6()
    state = get_dummy_state()

    projection1 = antarctica.project(state, np.random.default_rng(42))
    projection2 = antarctica.project(state, np.random.default_rng(42))

    np.testing.assert_allclose(projection1, projection2)


def test_ismip6_precomputation_matches_impulse_response():
    """Vectorized delayed-rate calculation should match the scalar implementation."""
    antarctica = get_dummy_ismip6()
    state = get_dummy_state()

    tas = state.T_ens
    dt = 1.0

    cum_rate1, cum_rate2 = antarctica._precompute_delayed_rates(
        tas,
        dt,
        state,
    )

    model_idx = 0
    trajectory_idx = 0

    tau1 = antarctica.param_ds.tau1.values[model_idx]
    tau2 = antarctica.param_ds.tau2.values[model_idx]
    gamma = antarctica.param_ds.gamma.values[model_idx]
    params = antarctica.param_ds.general_params.values[model_idx]

    expected = antarctica._impulse_response_term(
        tas[trajectory_idx],
        tau1,
        tau2,
        gamma,
        params,
        dt,
        state,
    )

    alpha1, alpha2, _ = params

    actual = (
        alpha1 * cum_rate1[model_idx, trajectory_idx]
        + alpha2 * cum_rate2[model_idx, trajectory_idx]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_antarctica_dyn_invalid_scenario():
    """Unknown scenarios should fail when cumulative emissions are not supplied."""
    antarctica = AntarcticaDynAR5()
    state = get_dummy_state(scenario="invalid")
    rng = np.random.default_rng(42)

    with pytest.raises(KeyError):
        antarctica.project(state, rng)


def test_antarctica_dyn_cumulative_emissions_bypasses_scenario(monkeypatch):
    """Explicit cumulative emissions should avoid the RCP scenario lookup."""
    antarctica = AntarcticaDynAR5(cum_emissions_total=1000.0)
    state = get_dummy_state(scenario="invalid")

    captured = {}

    def mock_time_projection(
        state,
        median,
        uncertainty,
        final,
        rng,
        fraction=None,
    ):
        captured["final"] = final
        return np.zeros((state.nt, state.num_members, state.nyr))

    monkeypatch.setattr(
        "profsea.components.global_.antarctica.time_projection",
        mock_time_projection,
    )

    antarctica.project(state, np.random.default_rng(42))

    expected_lower = (1.363e-05 * 1000.0) + 0.0392
    expected_upper = (0.000110 * 1000.0) + 0.375

    np.testing.assert_allclose(
        captured["final"],
        [expected_lower, expected_upper],
    )


def test_antarctica_dyn_adds_d_ant(monkeypatch):
    """The background Antarctic contribution should be added to the projection."""
    d_ant = 0.01
    antarctica = AntarcticaDynAR5(d_ant=d_ant)
    state = get_dummy_state()

    def mock_time_projection(*args, **kwargs):
        return np.zeros((state.nt, state.num_members, state.nyr))

    monkeypatch.setattr(
        "profsea.components.global_.antarctica.time_projection",
        mock_time_projection,
    )

    projection = antarctica.project(state, np.random.default_rng(42))

    np.testing.assert_allclose(projection, d_ant)


def test_antarctica_smb_wrong_fraction_size():
    """SMB projection should reject a fraction array of the wrong size."""
    antarctica = AntarcticaSMBAR5()
    state = get_dummy_state(fraction=np.array([0.2, 0.4, 0.6], dtype=np.float32))
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError, match="fraction is the wrong size"):
        antarctica.project(state, rng)


def test_antarctica_smb_projection_shape():
    """SMB output should contain one timeseries per ensemble member."""
    antarctica = AntarcticaSMBAR5()
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    projection = antarctica.project(state, rng)

    assert projection.shape == (
        state.nt,
        state.num_members,
        state.T_int_ens.shape[1],
    )


def test_antarctica_smb_zero_temperature_zero_projection():
    """Zero integrated temperature should give zero SMB contribution."""
    antarctica = AntarcticaSMBAR5()
    state = get_dummy_state(T_change_val=0.0)
    rng = np.random.default_rng(42)

    projection = antarctica.project(state, rng)

    np.testing.assert_allclose(projection, 0.0, atol=1e-7)


def test_antarctica_smb_reproducible_with_same_seed():
    """SMB stochastic sampling should be deterministic for a fixed seed."""
    antarctica = AntarcticaSMBAR5()
    state = get_dummy_state()

    projection1 = antarctica.project(
        state,
        np.random.default_rng(42),
    )
    projection2 = antarctica.project(
        state,
        np.random.default_rng(42),
    )

    np.testing.assert_allclose(projection1, projection2)
