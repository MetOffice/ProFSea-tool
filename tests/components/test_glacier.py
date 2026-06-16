import numpy as np
import pytest

from profsea.components.core.state import ClimateState
from profsea.components.global_.glacier import Glacier


def get_dummy_state(T_change_val: float) -> ClimateState:
    """Helper to generate a state object with a constant temperature."""
    T_ens = np.ones((2, 4)) * T_change_val
    T_int_ens = np.cumsum(T_ens, axis=1)
    T_int_med = np.cumsum(np.median(T_ens, axis=0))

    return ClimateState(
        scenario="test",
        T_ens=T_ens,
        T_int_ens=T_int_ens,
        T_int_med=T_int_med,
        fraction=np.random.rand(10),
        palmer_method=True,
        endofAR5=2100,
        endofhistory=2006,
        end_yr=2010,
        nyr=4,
        nt=2,
        num_members=2,
    )


def test_glacier_parameter_validation():
    # Attempting to initialize with an invalid MIP index
    glacier = Glacier(glaciermip=3)
    state = get_dummy_state(1.0)
    rng = np.random.default_rng(42)

    with pytest.raises(KeyError, match="glaciermip must be False"):
        glacier.project(state, rng)


def test_glacier_mass_limit():
    glacier = Glacier(glaciermip=2)

    # Force a massive, physically impossible temperature spike (1000 degrees)
    # This ensures the mathematical scaling tries to exceed the physical ice volume
    extreme_state = get_dummy_state(1000.0)
    rng = np.random.default_rng(42)

    projection = glacier.project(extreme_state, rng)

    # Calculate the hard limit:
    max_sle = (412.0 - 96.3) * 1e-3

    # Assert no value in the projection exceeds the hard mass limit
    assert np.all(projection <= max_sle)
