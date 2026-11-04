import numpy as np
import xarray as xr

from profsea.components.core.state import ClimateState
from profsea.components.global_.landwater import (
    LandwaterAR5,
    LandwaterAR6,
)


def get_dummy_state(
    *,
    nt: int = 2,
    num_members: int = 2,
    start_yr: int = 2006,
    end_yr: int = 2010,
    baseline_start: int = 1996,
    baseline_end: int = 2005,
) -> ClimateState:
    """Helper to generate a small ClimateState."""
    n_years = end_yr - start_yr
    T_ens = np.ones((nt, n_years), dtype=np.float32)
    T_int_ens = np.cumsum(T_ens, axis=1)
    T_int_med = np.cumsum(np.median(T_ens, axis=0))

    return ClimateState(
        scenario="ssp245",
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
        start_yr=start_yr,
        end_yr=end_yr,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        n_years=n_years,
        nt=nt,
        num_members=num_members,
    )


def get_dummy_landwater_obs(
    state,
    *,
    sle=0.0,
    sle_unc=0.0,
):
    """Create a minimal land-water observational dataset."""
    return xr.Dataset(
        {
            "sle": (
                "time",
                np.array([sle], dtype=state.dtype),
            ),
            "sle_unc": (
                "time",
                np.array([sle_unc], dtype=state.dtype),
            ),
        },
        coords={
            "time": [state.start_yr - 1],
        },
        attrs={
            "baseline_start": state.baseline_start,
            "baseline_end": state.baseline_end,
            "reference": "Synthetic test data",
        },
    )


def get_dummy_ar6() -> LandwaterAR6:
    """Create a LandwaterAR6 component without loading the auxiliary NetCDF."""
    landwater = LandwaterAR6.__new__(LandwaterAR6)

    years = np.arange(2005, 2301)

    # Two synthetic projection samples in millimetres.
    sea_level_change = np.stack(
        [
            np.arange(len(years), dtype=np.float32),
            np.arange(len(years), dtype=np.float32) * 2,
        ]
    )

    landwater.lw_ds = xr.Dataset(
        {
            "sea_level_change": (
                ("samples", "years"),
                sea_level_change,
            )
        },
        coords={
            "samples": [0, 1],
            "years": years,
        },
    )

    return landwater


def test_landwater_ar6_projection_shape():
    """AR6 output should contain one timeseries per ensemble member."""
    landwater = get_dummy_ar6()
    state = get_dummy_state()
    rng = np.random.default_rng(42)

    projection = landwater.project(state, rng)

    assert projection.shape == (
        state.nt,
        state.num_members,
        state.n_years,
    )


def test_landwater_ar6_reproducible_with_same_seed():
    """Sample selection should be deterministic for a fixed RNG seed."""
    landwater1 = get_dummy_ar6()
    landwater2 = get_dummy_ar6()
    state = get_dummy_state()

    projection1 = landwater1.project(
        state,
        np.random.default_rng(42),
    )
    projection2 = landwater2.project(
        state,
        np.random.default_rng(42),
    )

    np.testing.assert_allclose(projection1, projection2)


def test_landwater_ar6_converts_mm_to_metres():
    """AR6 projections should be converted from millimetres to metres."""
    landwater = LandwaterAR6.__new__(LandwaterAR6)

    years = np.arange(2005, 2301)

    landwater.lw_ds = xr.Dataset(
        {
            "sea_level_change": (
                ("samples", "years"),
                np.arange(len(years), dtype=np.float32)[None, :] * 1000.0,
            )
        },
        coords={
            "samples": [0],
            "years": years,
        },
    )

    state = get_dummy_state()
    rng = np.random.default_rng(42)

    projection = landwater.project(state, rng)

    np.testing.assert_allclose(
        projection[..., 1] - projection[..., 0],
        1.0,
    )


def test_landwater_ar6_is_relative_to_projection_start(monkeypatch):
    """AR6 projection should be re-referenced to the projection start year."""
    landwater = LandwaterAR6.__new__(LandwaterAR6)

    years = np.arange(2005, 2301)
    values = np.arange(len(years), dtype=np.float32)

    landwater.lw_ds = xr.Dataset(
        {
            "sea_level_change": (
                ("samples", "years"),
                values[np.newaxis, :],
            )
        },
        coords={
            "samples": [0],
            "years": years,
        },
    )

    state = get_dummy_state(
        nt=1,
        num_members=1,
    )

    initial_sle = 0.01

    def mock_read_landwater_obs(state, **kwargs):
        return get_dummy_landwater_obs(
            state,
            sle=initial_sle,
        )

    monkeypatch.setattr(
        "profsea.components.global_.landwater.read_landwater_obs",
        mock_read_landwater_obs,
    )

    projection = landwater.project(
        state,
        np.random.default_rng(42),
    )
    projection -= initial_sle

    # Added an extra bracket layer to make this a 3D array (1, 1, 4)
    expected = (
        np.array(
            [[[0.0, 1.0, 2.0, 3.0]]],
            dtype=np.float32,
        )
        * 1e-3
    )

    np.testing.assert_allclose(projection, expected, atol=1e-7)


def test_landwater_ar6_samples_only_existing_projection_members(monkeypatch):
    """All sampled projections should correspond to rows from the input dataset."""
    landwater = get_dummy_ar6()

    state = get_dummy_state(
        nt=3,
        num_members=4,
    )

    initial_sle = 0.01

    def mock_read_landwater_obs(state, **kwargs):
        return get_dummy_landwater_obs(
            state,
            sle=initial_sle,
        )

    monkeypatch.setattr(
        "profsea.components.global_.landwater.read_landwater_obs",
        mock_read_landwater_obs,
    )

    projection = landwater.project(
        state,
        np.random.default_rng(42),
    )
    projection -= initial_sle

    # The two source projections are:
    # sample 0 -> [0, 1, 2, 3] mm
    # sample 1 -> [0, 2, 4, 6] mm
    expected_sample_0 = np.array([0, 1, 2, 3]) * 1e-3
    expected_sample_1 = np.array([0, 2, 4, 6]) * 1e-3

    # Flatten the (climate, process) dims so we iterate strictly over 1D time-series rows
    for row in projection.reshape(-1, projection.shape[-1]):
        assert np.allclose(row, expected_sample_0) or np.allclose(
            row, expected_sample_1
        )


def test_landwater_ar6_preserves_state_dtype():
    """AR6 projection should use the ClimateState dtype."""
    landwater = get_dummy_ar6()
    state = get_dummy_state()

    state.dtype = np.float32

    projection = landwater.project(
        state,
        np.random.default_rng(42),
    )

    assert projection.dtype == np.float32


def test_landwater_ar5_uses_expected_parameters(monkeypatch):
    """AR5 should pass the documented start rate and final range."""
    landwater = LandwaterAR5()
    state = get_dummy_state()

    captured = {}

    def mock_time_projection(
        state,
        start_mean,
        start_uncertainty,
        final,
        rng,
        nfinal=None,
    ):
        captured["start_mean"] = start_mean
        captured["start_uncertainty"] = start_uncertainty
        captured["final"] = final
        captured["nfinal"] = nfinal

        return np.zeros(
            (state.nt, state.num_members, state.n_years),
            dtype=state.dtype,
        )

    monkeypatch.setattr(
        "profsea.components.global_.landwater.time_projection",
        mock_time_projection,
    )

    start_rate_mean = 0.5
    ar5_start_offset = 0.02

    years = np.arange(
        state.start_yr - 13,
        state.start_yr + 5,
    )

    # Construct an observational time series with a 0.5 mm/yr trend.
    sle = (years - years[0]) * start_rate_mean * 1e-3
    sle_unc = np.full(years.size, 0.001)

    def mock_read_landwater_obs(
        state,
        baseline_start=None,
        baseline_end=None,
    ):
        obs_sle = sle.copy()

        if baseline_start == 1986 and baseline_end == 2005:
            # Ensure the SLE at start_yr - 1 gives the required
            # AR5 start offset.
            idx = np.where(years == state.start_yr - 1)[0][0]
            obs_sle += ar5_start_offset - obs_sle[idx]

        return xr.Dataset(
            {
                "sle": ("time", obs_sle),
                "sle_unc": ("time", sle_unc),
            },
            coords={"time": years},
        )

    monkeypatch.setattr(
        "profsea.components.global_.landwater.read_landwater_obs",
        mock_read_landwater_obs,
    )

    landwater.project(
        state,
        np.random.default_rng(42),
    )

    np.testing.assert_allclose(
        captured["start_mean"],
        start_rate_mean,
    )
    np.testing.assert_allclose(
        captured["final"],
        [
            -0.01 - ar5_start_offset,
            0.09 - ar5_start_offset,
        ],
    )


def test_landwater_ar5_uses_twenty_year_final_average(monkeypatch):
    """AR5 final projection should represent the 2081-2100 mean."""
    landwater = LandwaterAR5()
    state = get_dummy_state()

    captured = {}

    def mock_time_projection(
        state,
        start_mean,
        start_uncertainty,
        final,
        rng,
        nfinal=None,
    ):
        captured["nfinal"] = nfinal

        return np.zeros(
            (state.nt, state.num_members, state.n_years),
            dtype=state.dtype,
        )

    monkeypatch.setattr(
        "profsea.components.global_.landwater.time_projection",
        mock_time_projection,
    )

    landwater.project(
        state,
        np.random.default_rng(42),
    )

    assert captured["nfinal"] == 20


def test_landwater_ar5_returns_time_projection_result(monkeypatch):
    """AR5 should return the time-projection output unchanged."""
    landwater = LandwaterAR5()
    state = get_dummy_state()

    expected = np.arange(
        state.nt * state.num_members * state.n_years,
        dtype=np.float32,
    ).reshape(
        state.nt,
        state.num_members,
        state.n_years,
    )

    def mock_time_projection(*args, **kwargs):
        return expected

    initial_sle = 0.01
    def mock_read_landwater_obs(state, **kwargs):
        years = np.arange(
            state.start_yr - 13,
            state.start_yr + 5,
        )

        # Simple linear series centred so that SLE at start_yr - 1
        # is initial_sle.
        sle = initial_sle + (years - (state.start_yr - 1)) * 0.0005

        return xr.Dataset(
            {
                "sle": ("time", sle),
                "sle_unc": (
                    "time",
                    np.full(years.size, 0.001),
                ),
            },
            coords={"time": years},
        )

    monkeypatch.setattr(
        "profsea.components.global_.landwater.read_landwater_obs",
        mock_read_landwater_obs,
    )
    monkeypatch.setattr(
        "profsea.components.global_.landwater.time_projection",
        mock_time_projection,
    )

    projection = landwater.project(
        state,
        np.random.default_rng(42),
    )

    np.testing.assert_allclose(projection, expected + initial_sle, atol=1e-7)
