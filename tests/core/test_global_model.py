import numpy as np
import pytest
import xarray as xr

from profsea.components.core.base import Component
from profsea.components.core.global_model import Global
from profsea.components.core.observations import Observations
from profsea.components.core.state import ClimateState


class MockGlobalComponent(Component):
    def project(
        self, state: ClimateState, rng: np.random.Generator, observations: Observations
    ) -> np.ndarray:
        # Update shape to (climate_members, process_members, time)
        return np.ones((state.nt, state.num_members, state.n_years), dtype=state.dtype)


def test_calculate_drivers_math():
    # 2 members, 3 time steps
    T_change = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])

    global_model = Global(
        components={},
        start_yr=2006,
        end_yr=2008,
        baseline_start=1986,
        baseline_end=2005,
        nt=2,
    )
    T_ens, T_int_ens, T_int_med = global_model._calculate_drivers(T_change)

    # Assert time integrals (cumulative sum over axis 1)
    np.testing.assert_array_equal(T_int_ens, [[1.0, 3.0, 6.0], [3.0, 7.0, 12.0]])

    # Median of T_change across members is [2.0, 3.0, 4.0]
    # Cumsum of that median is [2.0, 5.0, 9.0]
    np.testing.assert_array_equal(T_int_med, [2.0, 5.0, 9.0])


def test_run_orchestration():
    components = {"mock1": MockGlobalComponent(), "mock2": MockGlobalComponent()}

    # nt=2 (climate members), num_members=3 (process members), n_years=4 (2010-2006)
    global_model = Global(
        components=components,
        start_yr=2006,
        end_yr=2010,
        baseline_start=1995,
        baseline_end=2014,
        nt=2,
        num_members=3,
    )

    # Shape: (nt=2, n_years=4)
    T_change = np.zeros((2, 4))
    results = global_model.run(scenario="ssp119", T_change=T_change)

    assert "mock1" in results
    assert "mock2" in results

    # Assert the new 3D matrix shape
    assert results["mock1"].shape == (2, 3, 4)


def test_save_components(tmp_path):
    # tmp_path is a built-in pytest fixture that creates a temporary directory
    global_model = Global(
        components={},
        start_yr=2001,
        end_yr=2010,
        baseline_start=1986,
        baseline_end=2014,
    )

    # Create dummy result matching the new 3D dimensional structure
    components = {
        "mock_comp": xr.DataArray(
            np.random.rand(2, 3, 4), dims=["climate_member", "process_member", "time"]
        )
    }

    # Save the output to the temporary directory
    global_model.save_components(
        components,
        output_dir=str(tmp_path),
        scenario_name="test",
        output_prefix="global",
        output_format="netcdf",
    )

    expected_file = tmp_path / "test_global.nc"
    assert expected_file.exists()

    # Verify the contents of the NetCDF
    ds = xr.open_dataset(expected_file)
    assert "mock_comp" in ds.data_vars
    assert ds["mock_comp"].shape == (2, 3, 4)
    assert list(ds.dims) == ["climate_member", "process_member", "time"]


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64],
)
def test_return_types(dtype):
    components = {
        "mock": MockGlobalComponent(),
    }

    global_model = Global(
        components=components,
        start_yr=2006,
        end_yr=2101,
        baseline_start=1990,
        baseline_end=2000,
        nt=2,
        num_members=3,
        dtype=dtype,
    )

    T_change = np.zeros(
        (2, 95),
        dtype=dtype,
    )

    results = global_model.run(
        scenario="rcp26",
        T_change=T_change,
    )

    data = results["mock"]

    assert isinstance(data, xr.DataArray)
    assert data.dtype == dtype
    assert data.shape == (2, 3, 95)
