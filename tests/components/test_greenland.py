import numpy as np
import pandas as pd
import pytest
import xarray as xr

from profsea.components.core.observations import Observations
from profsea.components.core.state import ClimateState
from profsea.components.global_.greenland import (
    GreenlandAR6,
    GreenlandDynAR5,
    GreenlandSMBAR5,
)


@pytest.fixture
def observations(monkeypatch):
    """Provide observations without reading external observational files."""
    obs = Observations(
        baseline_start=1995,
        baseline_end=2005,
    )

    years = np.arange(1900, 2201)

    dataset = xr.Dataset(
        {
            "sle": ("time", np.zeros(years.size)),
            "sle_unc": ("time", np.zeros(years.size)),
            "rate": ("time", np.zeros(years.size)),
            "rate_unc": ("time", np.zeros(years.size)),
        },
        coords={"time": years},
    )

    monkeypatch.setattr(
        obs,
        "greenland",
        lambda **kwargs: dataset,
    )

    return obs


def get_dummy_state(
    T_change_val: float = 1.0,
    scenario: str = "rcp45",
    *,
    start_yr: int = 2006,
    end_yr: int = 2010,
    baseline_start: int = 1995,
    baseline_end: int = 2005,
    palmer_method: bool = True,
) -> ClimateState:
    """Helper to generate a small ClimateState with constant temperature."""
    n_years = end_yr - start_yr
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
        start_yr=start_yr,
        end_yr=end_yr,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
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


def test_greenland_ar6_projection_shape(observations):
    """AR6 output should contain one timeseries per ensemble member."""
    greenland = get_dummy_ar6()
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    projection = greenland.project(state, rng, observations)

    assert projection.shape == (
        state.nt,
        state.num_members,
        state.n_years,
    )


def test_greenland_ar6_reproducible_with_same_seed(observations):
    """Model and trend sampling should be deterministic for a fixed RNG seed."""
    greenland = get_dummy_ar6()
    state = get_dummy_state()

    projection1 = greenland.project(
        state,
        np.random.default_rng(42),
        observations
    )
    projection2 = greenland.project(
        state,
        np.random.default_rng(42),
        observations
    )

    np.testing.assert_allclose(projection1, projection2)


def test_greenland_ar6_zero_coefficients_only_leave_trend(
        monkeypatch,
        observations
    ):
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

    initial_sle = 0.01

    def mock_greenland_observations(component="total", **kwargs):
        assert component == "total"

        return xr.Dataset(
            {
                "sle": ("time", [initial_sle]),
                "rate": ("time", [0.0]),
                "rate_unc": ("time", [0.0]),
            },
            coords={
                "time": [state.start_yr - 1],
            },
        )

    monkeypatch.setattr(
        observations,
        "greenland",
        mock_greenland_observations,
    )

    projection = greenland.project(
        state,
        np.random.default_rng(42),
        observations,
    )

    # Trend is multiplied by time_delta, so the first timestep must be zero.
    np.testing.assert_allclose(projection[:, :, 0], initial_sle)

    # The truncated trend distribution is non-negative, so projections should
    # not decrease when all emulator coefficients are zero.
    assert np.all(np.diff(projection, axis=2) >= 0.0)


def test_greenland_ar6_accepts_one_dimensional_temperature(observations):
    """A 1D temperature trajectory should be promoted to a 2D ensemble."""
    greenland = get_dummy_ar6()
    state = get_dummy_state()

    state.T_ens = np.ones(state.n_years, dtype=np.float32)
    state.nt = 1

    projection = greenland.project(
        state,
        np.random.default_rng(42),
        observations
    )

    assert projection.shape == (
        state.nt,
        state.num_members,
        state.n_years,
    )


def test_greenland_ar6_persists_2100_rate(observations):
    """After 2100, AR6 should continue using the 2100 rate of change."""
    greenland = get_dummy_ar6()
    state = get_dummy_state(
        start_yr=2006,
        end_yr=2105,
    )

    projection = greenland.project(
        state,
        np.random.default_rng(42),
        observations
    )

    idx_2100 = 2100 - state.start_yr

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


def test_greenland_smb_projection_shape(observations):
    """SMB output should contain one timeseries per ensemble member."""
    greenland = GreenlandSMBAR5()
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    projection = greenland.project(
        state,
        rng,
        observations
    )

    assert projection.shape == (
        state.nt,
        state.num_members,
        state.n_years,
    )


def test_greenland_smb_reproducible_with_same_seed(observations):
    """SMB stochastic factors should be deterministic for a fixed seed."""
    greenland = GreenlandSMBAR5()
    state = get_dummy_state()

    projection1 = greenland.project(
        state,
        np.random.default_rng(42),
        observations
    )
    projection2 = greenland.project(
        state,
        np.random.default_rng(42),
        observations
    )

    np.testing.assert_allclose(projection1, projection2)


def test_greenland_smb_includes_dynamic_baseline(
        monkeypatch,
        observations
    ):
    """SMB projection should include the non-dynamic share of dgreen."""
    greenland = GreenlandSMBAR5()

    # ztgreen = T_ens - (-0.146), so setting T_ens to -0.146 makes
    # the Fettweis SMB term exactly zero.
    state = get_dummy_state(T_change_val=-0.146)
    rng = np.random.default_rng(42)

    smb_offset = 0.01

    def mock_greenland_observations(component="total", **kwargs):
        assert component == "smb"

        return xr.Dataset(
            {
                "sle": ("time", [smb_offset]),
                "rate": ("time", [0.0]),
                "rate_unc": ("time", [0.0]),
            },
            coords={
                "time": [state.start_yr - 1],
            },
        )

    monkeypatch.setattr(
        observations,
        "greenland",
        mock_greenland_observations,
    )

    projection = greenland.project(
        state,
        rng,
        observations,
    )

    np.testing.assert_allclose(
        projection,
        smb_offset,
        atol=1e-7,
    )


def test_greenland_smb_palmer_method_freezes_post_ar5_rate(observations):
    """Palmer mode should hold the annual SMB contribution constant after 2100."""
    greenland = GreenlandSMBAR5()

    state = get_dummy_state(
        T_change_val=1.0,
        end_yr=2105,
        palmer_method=True,
    )

    projection = greenland.project(
        state,
        np.random.default_rng(42),
        observations
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


def test_greenland_dyn_uses_rcp85_range(
    monkeypatch,
    observations,
):
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

        return np.zeros(
            (state.nt, state.num_members, state.n_years)
        )

    monkeypatch.setattr(
        "profsea.components.global_.greenland.time_projection",
        mock_time_projection,
    )

    dynamic_offset = 0.01
    dynamic_rate = 0.02
    dynamic_rate_unc = 0.03

    def mock_greenland_observations(
        component="total",
        baseline_start=None,
        baseline_end=None,
    ):
        assert component == "dyn"

        sle = (
            dynamic_offset
            if baseline_start == 1986 and baseline_end == 2005
            else 0.0
        )

        return xr.Dataset(
            {
                "sle": ("time", [sle]),
                "rate": ("time", [dynamic_rate]),
                "rate_unc": ("time", [dynamic_rate_unc]),
            },
            coords={
                "time": [state.start_yr - 1],
            },
        )

    monkeypatch.setattr(
        observations,
        "greenland",
        mock_greenland_observations,
    )

    greenland.project(
        state,
        np.random.default_rng(42),
        observations,
    )

    assert captured["final"] == [
        0.020 - dynamic_offset,
        0.085 - dynamic_offset,
    ]
    assert captured["median"] == dynamic_rate
    assert captured["uncertainty"] == dynamic_rate_unc


def test_greenland_dyn_uses_ssp585_range(
    monkeypatch,
    observations,
):
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

        return np.zeros(
            (state.nt, state.num_members, state.n_years)
        )

    monkeypatch.setattr(
        "profsea.components.global_.greenland.time_projection",
        mock_time_projection,
    )

    dynamic_offset = 0.01

    def mock_greenland_observations(
        component="total",
        baseline_start=None,
        baseline_end=None,
    ):
        assert component == "dyn"

        sle = (
            dynamic_offset
            if baseline_start == 1986 and baseline_end == 2005
            else 0.0
        )

        return xr.Dataset(
            {
                "sle": ("time", [sle]),
                "rate": ("time", [0.0]),
                "rate_unc": ("time", [0.0]),
            },
            coords={
                "time": [state.start_yr - 1],
            },
        )

    monkeypatch.setattr(
        observations,
        "greenland",
        mock_greenland_observations,
    )

    greenland.project(
        state,
        np.random.default_rng(42),
        observations,
    )

    assert captured["final"] == [
        0.020 - dynamic_offset,
        0.085 - dynamic_offset,
    ]


def test_greenland_dyn_uses_default_range_for_other_scenarios(
    monkeypatch,
    observations,
):
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

        return np.zeros(
            (state.nt, state.num_members, state.n_years)
        )

    monkeypatch.setattr(
        "profsea.components.global_.greenland.time_projection",
        mock_time_projection,
    )

    dynamic_offset = 0.01

    def mock_greenland_observations(
        component="total",
        baseline_start=None,
        baseline_end=None,
    ):
        assert component == "dyn"

        sle = (
            dynamic_offset
            if baseline_start == 1986 and baseline_end == 2005
            else 0.0
        )

        return xr.Dataset(
            {
                "sle": ("time", [sle]),
                "rate": ("time", [0.0]),
                "rate_unc": ("time", [0.0]),
            },
            coords={
                "time": [state.start_yr - 1],
            },
        )

    monkeypatch.setattr(
        observations,
        "greenland",
        mock_greenland_observations,
    )

    greenland.project(
        state,
        np.random.default_rng(42),
        observations,
    )

    np.testing.assert_allclose(
        captured["final"],
        [
            0.014 - dynamic_offset,
            0.063 - dynamic_offset,
        ],
    )


def test_greenland_dyn_adds_dynamic_baseline(
    monkeypatch,
    observations,
):
    """Greenland dynamics should add its historical initial condition."""
    greenland = GreenlandDynAR5()
    state = get_dummy_state()

    def mock_time_projection(*args, **kwargs):
        return np.zeros(
            (state.nt, state.num_members, state.n_years)
        )

    monkeypatch.setattr(
        "profsea.components.global_.greenland.time_projection",
        mock_time_projection,
    )

    initial_dynamic_sle = 0.01

    def mock_greenland_observations(
        component="total",
        baseline_start=None,
        baseline_end=None,
    ):
        assert component == "dyn"

        # The explicit 1986-2005 request is the AR5 re-referencing
        # dataset. Give it zero offset for this test.
        if baseline_start == 1986 and baseline_end == 2005:
            sle = 0.0
        else:
            # This is the normal observation request using the
            # Observations object's configured baseline.
            sle = initial_dynamic_sle

        return xr.Dataset(
            {
                "sle": ("time", [sle]),
                "rate": ("time", [0.0]),
                "rate_unc": ("time", [0.0]),
            },
            coords={
                "time": [state.start_yr - 1],
            },
        )

    monkeypatch.setattr(
        observations,
        "greenland",
        mock_greenland_observations,
    )

    projection = greenland.project(
        state,
        np.random.default_rng(42),
        observations,
    )

    np.testing.assert_allclose(
        projection,
        initial_dynamic_sle,
    )
