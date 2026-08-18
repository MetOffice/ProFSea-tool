import numpy as np

from profsea.components.core.state import ClimateState
from profsea.components.global_.expansion import ThermalExpansion


def get_dummy_state(
    *,
    n_years: int = 4,
    nt: int = 2,
    num_members: int = 3,
) -> ClimateState:
    """Helper to generate a small ClimateState."""
    T_ens = np.ones((nt, n_years), dtype=np.float32)
    T_int_ens = np.cumsum(T_ens, axis=1)
    T_int_med = np.cumsum(np.median(T_ens, axis=0))

    return ClimateState(
        scenario="ssp245",
        T_ens=T_ens,
        T_int_ens=T_int_ens,
        T_int_med=T_int_med,
        fraction=np.linspace(0.1, 0.9, nt * num_members, dtype=np.float32),
        palmer_method=True,
        endofAR5=2100,
        endofhistory=2006,
        end_yr=2010,
        n_years=n_years,
        nt=nt,
        num_members=num_members,
    )


def test_expansion_ohc_change_true_scales_input_with_efficiency():
    """When OHC_change is True, output should be OHC multiplied by sampled efficiency."""
    state = get_dummy_state(nt=2, num_members=3, n_years=4)

    data_input = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
        dtype=np.float32,
    )

    expansion = ThermalExpansion(
        data_input=data_input,
        distribution_scaler=0.0,
        OHC_change=True,
    )

    projection = expansion.project(state, np.random.default_rng(42))

    expected_efficiency = np.float32(0.113e-24)
    expected = (data_input[:, None, :] * expected_efficiency).astype(np.float32)
    expected = np.broadcast_to(
        expected,
        (state.nt, state.num_members, state.n_years),
    )

    assert projection.shape == (state.nt, state.num_members, state.n_years)
    np.testing.assert_allclose(projection, expected)


def test_expansion_ohc_change_false_broadcasts_input_without_scaling():
    """When OHC_change is False, input should be broadcast to all members unchanged."""
    state = get_dummy_state(nt=2, num_members=3, n_years=4)

    data_input = np.array(
        [
            [0.01, 0.02, 0.03, 0.04],
            [0.10, 0.20, 0.30, 0.40],
        ],
        dtype=np.float32,
    )

    expansion = ThermalExpansion(
        data_input=data_input,
        OHC_change=False,
    )

    projection = expansion.project(state, np.random.default_rng(42))

    expected = np.broadcast_to(
        data_input[:, None, :],
        (state.nt, state.num_members, state.n_years),
    )

    assert projection.shape == (state.nt, state.num_members, state.n_years)
    np.testing.assert_allclose(projection, expected)
