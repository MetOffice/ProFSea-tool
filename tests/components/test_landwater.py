import numpy as np
import xarray as xr

from profsea.components.core.state import ClimateState
from profsea.components.global_.landwater import (
    LandwaterAR5,
    LandwaterAR6,
)


def get_dummy_state(
    *,
    nyr: int = 4,
    nt: int = 2,
    num_members: int = 2,
    end_yr: int = 2010,
) -> ClimateState:
    """Helper to generate a small ClimateState."""
    T_ens = np.ones((nt, nyr), dtype=np.float32)
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
        end_yr=end_yr,
        nyr=nyr,
        nt=nt,
        num_members=num_members,
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
        state.nyr,
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
                np.ones((1, len(years)), dtype=np.float32) * 1000.0,
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

    np.testing.assert_allclose(projection, 1.0)


def test_landwater_ar6_skips_first_interpolated_year():
    """Projection should begin from the second year in the interpolated series."""
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
        nyr=4,
    )

    projection = landwater.project(
        state,
        np.random.default_rng(42),
    )

    # Added an extra bracket layer to make this a 3D array (1, 1, 4)
    expected = (
        np.array(
            [[[1.0, 2.0, 3.0, 4.0]]],
            dtype=np.float32,
        )
        * 1e-3
    )

    np.testing.assert_allclose(projection, expected)


def test_landwater_ar6_samples_only_existing_projection_members():
    """All sampled projections should correspond to rows from the input dataset."""
    landwater = get_dummy_ar6()

    state = get_dummy_state(
        nt=3,
        num_members=4,
    )

    projection = landwater.project(
        state,
        np.random.default_rng(42),
    )

    # The two source projections are:
    # sample 0 -> [1, 2, 3, 4] mm
    # sample 1 -> [2, 4, 6, 8] mm
    expected_sample_0 = np.array([1, 2, 3, 4]) * 1e-3
    expected_sample_1 = np.array([2, 4, 6, 8]) * 1e-3

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
            (state.nt, state.num_members, state.nyr),
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

    assert captured["start_mean"] == 0.38
    assert captured["start_uncertainty"] == 0.49 - 0.38
    assert captured["final"] == [-0.01, 0.09]


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
            (state.nt, state.num_members, state.nyr),
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
        state.nt * state.num_members * state.nyr,
        dtype=np.float32,
    ).reshape(
        state.nt,
        state.num_members,
        state.nyr,
    )

    def mock_time_projection(*args, **kwargs):
        return expected

    monkeypatch.setattr(
        "profsea.components.global_.landwater.time_projection",
        mock_time_projection,
    )

    projection = landwater.project(
        state,
        np.random.default_rng(42),
    )

    np.testing.assert_array_equal(projection, expected)
