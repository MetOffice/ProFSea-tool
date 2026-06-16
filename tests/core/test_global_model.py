import numpy as np
import xarray as xr

from profsea.components.core.base import Component
from profsea.components.core.global_model import Global
from profsea.components.core.state import ClimateState


class MockGlobalComponent(Component):
    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
        # Just return an array of 1s with the correct shape
        return np.ones((state.nt * state.num_members, state.nyr))


def test_calculate_drivers_math():
    # 2 members, 3 time steps
    T_change = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])

    global_model = Global(components={}, end_yr=2008, nt=2)
    T_ens, T_int_ens, T_int_med = global_model._calculate_drivers(T_change)

    # Assert time integrals (cumulative sum over axis 1)
    np.testing.assert_array_equal(T_int_ens, [[1.0, 3.0, 6.0], [3.0, 7.0, 12.0]])

    # Median of T_change across members is [2.0, 3.0, 4.0]
    # Cumsum of that median is [2.0, 5.0, 9.0]
    np.testing.assert_array_equal(T_int_med, [2.0, 5.0, 9.0])


def test_run_orchestration():
    components = {"mock1": MockGlobalComponent(), "mock2": MockGlobalComponent()}

    # 2 time series, 3 members each -> output shape should be (6, nyr)
    global_model = Global(components=components, end_yr=2010, nt=2, num_members=3)

    # Shape: (nt=2, nyr=4)
    T_change = np.zeros((2, 4))
    results = global_model.run(scenario="ssp119", T_change=T_change)

    assert "mock1" in results
    assert "mock2" in results
    assert results["mock1"].shape == (6, 4)


def test_save_components(tmp_path):
    # tmp_path is a built-in pytest fixture that creates a temporary directory
    global_model = Global(components={}, end_yr=2010)

    # Create dummy result: 5 members, 4 years
    components = {"mock_comp": np.random.rand(5, 4)}

    # Save the output to the temporary directory
    global_model.save_components(
        components, output_dir=str(tmp_path), scenario_name="test"
    )

    expected_file = tmp_path / "test_global.nc"
    assert expected_file.exists()

    # Verify the contents of the NetCDF
    ds = xr.open_dataset(expected_file)
    assert "mock_comp" in ds.data_vars
    assert ds["mock_comp"].shape == (5, 4)
    assert list(ds.dims) == ["member", "time"]
