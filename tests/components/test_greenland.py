import numpy as np
import pandas as pd

from profsea.components.core.state import ClimateState
from profsea.components.global_.greenland import (
    GreenlandAR6,
    GreenlandDynAR5,
    GreenlandSMBAR5,
)


def get_dummy_state(
    T_change_val: float = 1.0,
    scenario: str = "rcp45",
    *,
    n_years: int = 4,
    end_yr: int = 2010,
    palmer_method: bool = True,
) -> ClimateState:
    """Helper to generate a small ClimateState with constant temperature."""
    T_ens = np.ones((2, n_years), dtype=np.float32) * T_change_val
    T_int_ens = np.cumsum(T_ens, axis=1)
    T_int_med = np.cumsum(np.median(T_ens, axis=0))

    return ClimateState(
        scenario=scenario,
        T_ens=T_ens,
        T_int_ens=T_int_ens,
        T_int_med=T_int_med,
        fraction=np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
        palmer_method=palmer_method,
        endofAR5=2100,
        endofhistory=2006,
        end_yr=end_yr,
        n_years=n_years,
        nt=2,
        num_members=2,
    )


def get_dummy_ar6() -> GreenlandAR6:
    """Create a Greenland AR6 component without reading the calibration CSV."""
    greenland = GreenlandAR6.__new__(GreenlandAR6)

    greenland.df = pd.DataFrame(
        {
            "b0": [1.0, 2.0],
            "b1": [0.5, 0.25],
            "b2": [0.0, 0.0],
            "b3": [0.0, 0.0],
            "b4": [0.0, 0.0],
            "b5": [0.0, 0.0],
        }
    )

    return greenland


def test_greenland_ar6_projection_shape():
    """AR6 output should contain one timeseries per ensemble member."""
    greenland = get_dummy_ar6()
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    projection = greenland.project(state, rng)

    assert projection.shape == (
        state.nt,
        state.num_members,
        state.n_years,
    )


def test_greenland_ar6_reproducible_with_same_seed():
    """Model and trend sampling should be deterministic for a fixed RNG seed."""
    greenland = get_dummy_ar6()
    state = get_dummy_state()

    projection1 = greenland.project(
        state,
        np.random.default_rng(42),
    )
    projection2 = greenland.project(
        state,
        np.random.default_rng(42),
    )

    np.testing.assert_allclose(projection1, projection2)


def test_greenland_ar6_zero_coefficients_only_leave_trend():
    """With zero emulator coefficients, the AR6 result should only contain trend."""
    greenland = GreenlandAR6.__new__(GreenlandAR6)

    greenland.df = pd.DataFrame(
        {
            "b0": [0.0],
            "b1": [0.0],
            "b2": [0.0],
            "b3": [0.0],
            "b4": [0.0],
            "b5": [0.0],
        }
    )

    state = get_dummy_state(T_change_val=0.0)
    projection = greenland.project(
        state,
        np.random.default_rng(42),
    )

    # Trend is multiplied by time_delta, so the first timestep must be zero.
    np.testing.assert_allclose(projection[:, :, 0], 0.0)

    # The truncated trend distribution is non-negative, so projections should
    # not decrease when all emulator coefficients are zero.
    assert np.all(np.diff(projection, axis=2) >= 0.0)


def test_greenland_ar6_accepts_one_dimensional_temperature():
    """A 1D temperature trajectory should be promoted to a 2D ensemble."""
    greenland = get_dummy_ar6()
    state = get_dummy_state()

    state.T_ens = np.ones(state.n_years, dtype=np.float32)
    state.nt = 1

    projection = greenland.project(
        state,
        np.random.default_rng(42),
    )

    assert projection.shape == (
        state.nt,
        state.num_members,
        state.n_years,
    )


def test_greenland_ar6_persists_2100_rate():
    """After 2100, AR6 should continue using the 2100 rate of change."""
    n_years = 100
    greenland = get_dummy_ar6()
    state = get_dummy_state(
        n_years=n_years,
        end_yr=2105,
    )

    projection = greenland.project(
        state,
        np.random.default_rng(42),
    )

    idx_2100 = 94

    rate_2100 = projection[:, :, idx_2100] - projection[:, :, idx_2100 - 1]
    rate_after = projection[:, :, idx_2100 + 1] - projection[:, :, idx_2100]

    np.testing.assert_allclose(
        rate_after,
        rate_2100,
        rtol=1e-5,
        atol=1e-7,
    )


def test_greenland_smb_fettweis_zero_at_zero_anomaly_offset():
    """The Fettweis parameterisation should be zero for zero input anomaly."""
    greenland = GreenlandSMBAR5()

    result = greenland._fettweis(np.zeros((2, 4), dtype=np.float32))

    np.testing.assert_allclose(result, 0.0)


def test_greenland_smb_fettweis_matches_formula():
    """The Fettweis helper should implement the documented polynomial."""
    greenland = GreenlandSMBAR5()

    temperature = np.array([1.0, 2.0], dtype=np.float32)

    expected = (
        71.5 * temperature + 20.4 * temperature**2 + 2.8 * temperature**3
    ) * greenland.mSLEoGt

    actual = greenland._fettweis(temperature)

    np.testing.assert_allclose(actual, expected)


def test_greenland_smb_projection_shape():
    """SMB output should contain one timeseries per ensemble member."""
    greenland = GreenlandSMBAR5()
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    projection = greenland.project(state, rng)

    assert projection.shape == (
        state.nt,
        state.num_members,
        state.n_years,
    )


def test_greenland_smb_reproducible_with_same_seed():
    """SMB stochastic factors should be deterministic for a fixed seed."""
    greenland = GreenlandSMBAR5()
    state = get_dummy_state()

    projection1 = greenland.project(
        state,
        np.random.default_rng(42),
    )
    projection2 = greenland.project(
        state,
        np.random.default_rng(42),
    )

    np.testing.assert_allclose(projection1, projection2)


def test_greenland_smb_includes_dynamic_baseline():
    """SMB projection should include the non-dynamic share of dgreen."""
    greenland = GreenlandSMBAR5()

    # ztgreen = T_ens - (-0.146), so setting T_ens to -0.146 makes
    # the Fettweis SMB term exactly zero.
    state = get_dummy_state(T_change_val=-0.146)
    rng = np.random.default_rng(42)

    projection = greenland.project(state, rng)

    expected = (1 - greenland.fgreendyn) * greenland.dgreen

    np.testing.assert_allclose(
        projection,
        expected,
        atol=1e-7,
    )


def test_greenland_smb_palmer_method_freezes_post_ar5_rate():
    """Palmer mode should hold the annual SMB contribution constant after 2100."""
    greenland = GreenlandSMBAR5()

    state = get_dummy_state(
        T_change_val=1.0,
        n_years=100,
        end_yr=2105,
        palmer_method=True,
    )

    projection = greenland.project(
        state,
        np.random.default_rng(42),
    )

    # project() freezes the annual contribution from index 95 onward
    # before applying cumulative sum. Therefore subsequent increments
    # should be equal.
    increment_95 = projection[:, :, 95] - projection[:, :, 94]
    increment_96 = projection[:, :, 96] - projection[:, :, 95]

    np.testing.assert_allclose(
        increment_95,
        increment_96,
        rtol=1e-5,
        atol=1e-7,
    )


def test_greenland_dyn_uses_rcp85_range(monkeypatch):
    """RCP8.5 should use the high Greenland dynamics final range."""
    greenland = GreenlandDynAR5()
    state = get_dummy_state(scenario="rcp85")

    captured = {}

    def mock_time_projection(
        state,
        median,
        uncertainty,
        final,
        rng,
        fraction=None,
    ):
        captured["median"] = median
        captured["uncertainty"] = uncertainty
        captured["final"] = final
        return np.zeros((state.nt, state.num_members, state.n_years))

    monkeypatch.setattr(
        "profsea.components.global_.greenland.time_projection",
        mock_time_projection,
    )

    greenland.project(
        state,
        np.random.default_rng(42),
    )

    assert captured["final"] == [0.020, 0.085]
    assert captured["median"] == 0.63 * greenland.fgreendyn
    assert captured["uncertainty"] == 0.17 * greenland.fgreendyn


def test_greenland_dyn_uses_ssp585_range(monkeypatch):
    """SSP5-8.5 should use the same high dynamics range as RCP8.5."""
    greenland = GreenlandDynAR5()
    state = get_dummy_state(scenario="ssp585")

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
        return np.zeros((state.nt, state.num_members, state.n_years))

    monkeypatch.setattr(
        "profsea.components.global_.greenland.time_projection",
        mock_time_projection,
    )

    greenland.project(
        state,
        np.random.default_rng(42),
    )

    assert captured["final"] == [0.020, 0.085]


def test_greenland_dyn_uses_default_range_for_other_scenarios(monkeypatch):
    """Non-RCP8.5 scenarios should use the standard dynamics range."""
    greenland = GreenlandDynAR5()
    state = get_dummy_state(scenario="rcp45")

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
        return np.zeros((state.nt, state.num_members, state.n_years))

    monkeypatch.setattr(
        "profsea.components.global_.greenland.time_projection",
        mock_time_projection,
    )

    greenland.project(
        state,
        np.random.default_rng(42),
    )

    assert captured["final"] == [0.014, 0.063]


def test_greenland_dyn_adds_dynamic_baseline(monkeypatch):
    """Greenland dynamics should add its fraction of the historical baseline."""
    greenland = GreenlandDynAR5()
    state = get_dummy_state()

    def mock_time_projection(*args, **kwargs):
        return np.zeros((state.nt, state.num_members, state.n_years))

    monkeypatch.setattr(
        "profsea.components.global_.greenland.time_projection",
        mock_time_projection,
    )

    projection = greenland.project(
        state,
        np.random.default_rng(42),
    )

    expected = greenland.fgreendyn * greenland.dgreen

    np.testing.assert_allclose(projection, expected)
