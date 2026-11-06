import numpy as np
import xarray as xr

from profsea.components.core.base import Component
from profsea.components.core.global_model import Global
from profsea.components.core.state import ClimateState


class MockGlobalComponent(Component):
    def project(self, state: ClimateState, rng: np.random.Generator) -> np.ndarray:
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


def test_return_types(monkeypatch):
    # Import a number of real components to test their return types
    from profsea.components.global_ import (
        AntarcticaDynAR5,
        AntarcticaISMIP6,
        AntarcticaSMBAR5,
        Glacier,
        GreenlandAR6,
        GreenlandDynAR5,
        GreenlandSMBAR5,
        LandwaterAR5,
        LandwaterAR6,
        ThermalExpansion,
    )

    # Mock observational readers so this test does not depend on external
    years = np.arange(1900, 2201)

    def mock_read_antarctic_obs(
        baseline_start,
        baseline_end,
        component="total",
        region=None,
    ):
        return xr.Dataset(
            {
                "sle": ("time", np.zeros(years.size)),
                "sle_unc": ("time", np.full(years.size, 0.001)),
                "rate": ("time", np.zeros(years.size)),
                "rate_unc": ("time", np.full(years.size, 0.001)),
            },
            coords={"time": years},
        )

    def mock_read_greenland_obs(
        baseline_start,
        baseline_end,
        component="total",
    ):
        return xr.Dataset(
            {
                "sle": ("time", np.zeros(years.size)),
                "sle_unc": ("time", np.full(years.size, 0.001)),
                "rate": ("time", np.zeros(years.size)),
                "rate_unc": ("time", np.full(years.size, 0.001)),
            },
            coords={"time": years},
        )

    def mock_read_glacier_obs(
        baseline_start,
        baseline_end,
    ):
        return xr.Dataset(
            {
                "sle": ("time", np.zeros(years.size)),
                "sle_unc": ("time", np.full(years.size, 0.001)),
                "rate": ("time", np.zeros(years.size)),
                "rate_unc": ("time", np.full(years.size, 0.001)),
            },
            coords={"time": years},
        )

    def mock_read_landwater_obs(
        baseline_start,
        baseline_end,
    ):
        return xr.Dataset(
            {
                "sle": ("time", np.zeros(years.size)),
                "sle_unc": ("time", np.full(years.size, 0.001)),
            },
            coords={"time": years},
        )

    monkeypatch.setattr(
        "profsea.components.global_.antarctica.read_antarctic_obs",
        mock_read_antarctic_obs,
    )
    monkeypatch.setattr(
        "profsea.components.global_.greenland.read_greenland_obs",
        mock_read_greenland_obs,
    )
    monkeypatch.setattr(
        "profsea.components.global_.glacier.read_glacier_obs",
        mock_read_glacier_obs,
    )
    monkeypatch.setattr(
        "profsea.components.global_.landwater.read_landwater_obs",
        mock_read_landwater_obs,
    )

    tas = np.zeros((2, 95))  # 2 trajectories, 95 years
    ohc = np.zeros((2, 95))

    components = {
        "AntarcticaISMIP6": AntarcticaISMIP6(region="wais"),
        "AntarcticaSMBAR5": AntarcticaSMBAR5(),
        "AntarcticaDynAR5": AntarcticaDynAR5(),
        "GreenlandAR6": GreenlandAR6(),
        "GreenlandDynAR5": GreenlandDynAR5(),
        "GreenlandSMBAR5": GreenlandSMBAR5(),
        "Glacier": Glacier(),
        "LandwaterAR5": LandwaterAR5(),
        "LandwaterAR6": LandwaterAR6(),
        "ThermalExpansion": ThermalExpansion(data_input=ohc),
    }

    global_model_float64 = Global(
        components=components,
        start_yr=2006,
        end_yr=2101,
        baseline_start=1990,
        baseline_end=2000,
        nt=2,
        num_members=3,
        dtype=np.float64,
    )
    results = global_model_float64.run(scenario="rcp26", T_change=tas)

    for comp_name, data in results.items():
        assert isinstance(data, xr.DataArray)
        assert data.dtype == np.float64
        # Assert that the real components correctly return the 3D structure
        assert data.shape == (2, 3, 95)

    global_model_float32 = Global(
        components=components,
        start_yr=2006,
        end_yr=2101,
        baseline_start=1990,
        baseline_end=2014,
        nt=2,
        num_members=3,
        dtype=np.float32,
    )
    results = global_model_float32.run(scenario="rcp26", T_change=tas)

    for comp_name, data in results.items():
        assert isinstance(data, xr.DataArray)
        assert data.dtype == np.float32
        assert data.shape == (2, 3, 95)
