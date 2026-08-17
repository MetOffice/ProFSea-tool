import numpy as np
import pytest

from profsea.components.core.state import ClimateState
from profsea.components.global_.glacier import Glacier


def get_dummy_state(
    T_change_val: float = 1.0,
    *,
    nyr: int = 4,
    nt: int = 2,
    num_members: int = 2,
) -> ClimateState:
    """Helper to generate a state object with constant temperature."""
    T_ens = np.ones((nt, nyr), dtype=np.float32) * T_change_val
    T_int_ens = np.cumsum(T_ens, axis=1)
    T_int_med = np.cumsum(np.median(T_ens, axis=0))

    return ClimateState(
        scenario="test",
        T_ens=T_ens,
        T_int_ens=T_int_ens,
        T_int_med=T_int_med,
        fraction=np.linspace(
            0.1,
            0.9,
            nt * num_members,
            dtype=np.float32,
        ),
        palmer_method=True,
        endofAR5=2100,
        endofhistory=2006,
        end_yr=2010,
        nyr=nyr,
        nt=nt,
        num_members=num_members,
    )


def test_glacier_parameter_validation():
    """Invalid GlacierMIP selections should raise a clear error."""
    glacier = Glacier(glaciermip=3)
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    with pytest.raises(
        KeyError,
        match="glaciermip must be False",
    ):
        glacier.project(state, rng)


@pytest.mark.parametrize("glaciermip", [False, 1, 2])
def test_glacier_valid_parameter_sets(glaciermip):
    """All documented GlacierMIP parameter options should produce projections."""
    glacier = Glacier(glaciermip=glaciermip)
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    projection = glacier.project(state, rng)

    assert projection.shape == (
        state.nt * state.num_members,
        state.nyr,
    )


def test_glacier_projection_shape():
    """Projection should contain one timeseries per trajectory/member pair."""
    glacier = Glacier(glaciermip=2)
    state = get_dummy_state(
        nt=3,
        num_members=4,
    )
    rng = np.random.default_rng(42)

    projection = glacier.project(state, rng)

    assert projection.shape == (
        state.nt * state.num_members,
        state.nyr,
    )


def test_glacier_reproducible_with_same_seed():
    """Model selection and random variance should be reproducible with a fixed seed."""
    glacier = Glacier(glaciermip=2)
    state = get_dummy_state()

    projection1 = glacier.project(
        state,
        np.random.default_rng(42),
    )
    projection2 = glacier.project(
        state,
        np.random.default_rng(42),
    )

    np.testing.assert_allclose(projection1, projection2)


def test_glacier_mass_limit():
    """Projected glacier contribution should not exceed total available glacier mass."""
    glacier = Glacier(glaciermip=2)

    extreme_state = get_dummy_state(1000.0)
    rng = np.random.default_rng(42)

    projection = glacier.project(extreme_state, rng)

    max_sle = (412.0 - 96.3) * 1e-3

    assert np.all(projection <= max_sle)


def test_glacier_negative_integrated_temperature_is_clipped_to_zero():
    """Negative integrated temperature should not produce negative glacier melt."""
    glacier = Glacier(glaciermip=2)
    state = get_dummy_state(-1.0)

    T_int = np.array(
        [[[-1.0, -2.0, -3.0, -4.0]]],
        dtype=np.float32,
    )
    factor = np.array([[[4.0]]], dtype=np.float32)
    exponent = np.array([[[0.7]]], dtype=np.float32)

    result = glacier._project_glacier1(
        T_int,
        factor,
        exponent,
        state,
    )

    np.testing.assert_allclose(result, 0.0)


def test_glacier_project_glacier1_matches_formula():
    """The glacier helper should implement the documented scaling relation."""
    glacier = Glacier(glaciermip=2)
    state = get_dummy_state()

    T_int = np.array(
        [[[0.0, 1.0, 4.0, 9.0]]],
        dtype=np.float32,
    )
    factor = np.array([[[2.0]]], dtype=np.float32)
    exponent = np.array([[[0.5]]], dtype=np.float32)

    expected = (
        1e-3
        * factor
        * np.array(
            [[[0.0, 1.0, 2.0, 3.0]]],
            dtype=np.float32,
        )
    )

    result = glacier._project_glacier1(
        T_int,
        factor,
        exponent,
        state,
    )

    np.testing.assert_allclose(result, expected)


def test_glacier_project_glacier1_broadcasts_parameters():
    """Model parameters should broadcast correctly across trajectories and time."""
    glacier = Glacier(glaciermip=2)
    state = get_dummy_state()

    T_int = np.ones((2, 1, 4), dtype=np.float32)

    factor = np.array(
        [
            [[2.0], [3.0]],
            [[4.0], [5.0]],
        ],
        dtype=np.float32,
    )

    exponent = np.ones_like(factor)

    result = glacier._project_glacier1(
        T_int,
        factor,
        exponent,
        state,
    )

    assert result.shape == (2, 2, 4)

    np.testing.assert_allclose(
        result[:, :, 0],
        np.array(
            [
                [0.002, 0.003],
                [0.004, 0.005],
            ],
            dtype=np.float32,
        ),
    )


def test_glacier_adds_reference_period_offset():
    """Projection should include the historical glacier offset dmz."""
    glacier = Glacier(glaciermip=2)

    # Zero integrated temperature removes the temperature-driven component.
    state = get_dummy_state(0.0)
    rng = np.random.default_rng(42)

    projection = glacier.project(state, rng)

    dmzdtref = 0.95
    expected_dmz = dmzdtref * (state.endofhistory - 1996) * 1e-3

    np.testing.assert_allclose(
        projection,
        expected_dmz,
        atol=1e-7,
    )


def test_glacier_accepts_one_dimensional_temperature():
    """A single temperature trajectory should be accepted when state.nt is one."""
    glacier = Glacier(glaciermip=2)
    state = get_dummy_state(
        nt=1,
        num_members=2,
    )

    state.T_ens = np.ones(state.nyr, dtype=np.float32)
    state.T_int_ens = np.cumsum(
        state.T_ens,
    )[None, :]
    state.T_int_med = np.cumsum(state.T_ens)

    projection = glacier.project(
        state,
        np.random.default_rng(42),
    )

    assert projection.shape == (
        state.num_members,
        state.nyr,
    )


def test_glacier_output_uses_state_dtype():
    """Projection output should preserve the configured ClimateState dtype."""
    glacier = Glacier(glaciermip=2)
    state = get_dummy_state()
    state.dtype = np.float32

    projection = glacier.project(
        state,
        np.random.default_rng(42),
    )

    assert projection.dtype == np.float32
