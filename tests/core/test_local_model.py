from unittest.mock import patch

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from profsea.components.core.base import SpatialComponent
from profsea.components.core.local_model import Local


class DummyState:
    """Mock state object to hold target coordinates and ensemble dimensions."""

    def __init__(self, lats, lons):
        self.target_lats = np.atleast_1d(lats)
        self.target_lons = np.atleast_1d(lons)
        self.nt = 2
        self.num_members = 3
        self.n_years = 4


class MockLocalComponent(SpatialComponent):
    @property
    def global_projection(self):
        # 3D: (climate_member, process_member, time)
        return np.zeros((2, 3, 4))

    def project(self, state, rng):
        # 4D for site projections: (climate_member, process_member, time, site)
        return np.ones(
            (state.nt, state.num_members, state.n_years, len(state.target_lats))
        )


class TestLazyLocalExtraction:
    @pytest.fixture
    def lazy_grid(self) -> xr.DataArray:
        """
        Creates a predictable 3x4 global grid with known values and NaNs.
        Latitudes: -10, 0, 10
        Longitudes: -170, -90, 0, 90
        """
        lats = np.array([-10.0, 0.0, 10.0])
        lons = np.array([-170.0, -90.0, 0.0, 90.0])

        # Predictable values 1 through 12
        data = np.arange(1, 13, dtype=float).reshape(3, 4)

        # Inject NaNs to test coastal weighting
        data[0, 2] = np.nan  # lat=-10, lon=0
        data[2, 3] = np.nan  # lat=10,  lon=90

        dask_data = da.from_array(data, chunks=(3, 2))

        return xr.DataArray(
            dask_data, dims=["lat", "lon"], coords={"lat": lats, "lon": lons}
        )

    @pytest.fixture
    def component(self):
        return MockLocalComponent()

    def test_preserves_lazy_evaluation(self, lazy_grid, component):
        """Ensure the extraction operations do not eagerly compute the Dask array."""
        state = DummyState(lats=[5.0], lons=[-45.0])
        result = component.extract_spatial(lazy_grid, state)
        assert isinstance(result.data, da.Array), (
            "Extraction triggered eager evaluation!"
        )

    def test_standard_bilinear_interpolation(self, lazy_grid, component):
        """Test a clean extraction in the center of 4 valid nodes."""
        state = DummyState(lats=[5.0], lons=[-45.0])
        result = component.extract_spatial(lazy_grid, state).compute()
        np.testing.assert_allclose(result.values, [8.5])

    def test_coastal_nan_renormalization(self, lazy_grid, component):
        """Test extraction where one corner of the bounding box is NaN."""
        state = DummyState(lats=[-5.0], lons=[-45.0])
        result = component.extract_spatial(lazy_grid, state).compute()
        np.testing.assert_allclose(result.values, [5.0])

    def test_exact_node_strike(self, lazy_grid, component):
        """Ensure zero-division safeguards work when hitting a coordinate exactly."""
        state = DummyState(lats=[0.0], lons=[-90.0])
        result = component.extract_spatial(lazy_grid, state).compute()
        np.testing.assert_allclose(result.values, [6.0])

    def test_zonal_periodicity_wrapping(self, lazy_grid, component):
        """Test extraction across the -180/180 antimeridian line."""
        state = DummyState(lats=[0.0], lons=[180.0])
        result = component.extract_spatial(lazy_grid, state).compute()
        np.testing.assert_allclose(result.values, [5.3])

    def test_all_nan_handling(self, lazy_grid, component):
        """Ensure regions completely surrounded by NaNs yield NaN."""
        lazy_grid *= np.nan  # Make the entire grid NaN
        state = DummyState(lats=[5.0], lons=[45.0])
        result = component.extract_spatial(lazy_grid, state).compute()
        assert np.isnan(result.values[0])

    def test_missing_spatial_attributes(self, lazy_grid, component):
        """Ensure a ValueError is raised if the state lacks valid spatial targets."""

        class DummyStateEmpty:
            pass  # A truly empty state object

        state = DummyStateEmpty()
        with pytest.raises(
            ValueError, match="State object is missing required spatial attributes"
        ):
            component.extract_spatial(lazy_grid, state)


class TestLocalMetadataGeneration:
    @patch("profsea.components.core.local_model.fetch_zenodo_fingerprints")
    def test_local_arr_to_xr_metadata_with_percentiles(self, mock_fetch):
        """Metadata generator should produce 3D structure when percentiles are active."""
        components = {"mock_comp": MockLocalComponent()}
        locations = {"Newlyn": (50.1, -5.5)}
        local = Local(
            components=components,
            locations=locations,
            end_year=2010,
            output_percentiles=[5, 50, 95],
        )

        # Mocking 3D output: (percentile, time, site)
        arr = da.zeros((3, 4, 1))
        xr_dict = local._arr_to_xr({"mock_comp": arr})

        da_out = xr_dict["mock_comp"]
        assert list(da_out.dims) == ["percentile", "time", "site"]
        assert len(da_out.percentile) == 3

    @patch("profsea.components.core.local_model.fetch_zenodo_fingerprints")
    def test_local_arr_to_xr_metadata_no_percentiles(self, mock_fetch):
        """Metadata generator should produce 4D tensor when percentiles are None."""
        components = {"mock_comp": MockLocalComponent()}
        locations = {"Newlyn": (50.1, -5.5)}
        local = Local(
            components=components,
            locations=locations,
            end_year=2010,
            output_percentiles=None,
        )

        # Local model should read the 2x3 shape from the dummy component's global projection
        assert local.nt == 2
        assert local.num_members == 3

        # Mocking 4D output: (climate_member, process_member, time, site)
        arr = da.zeros((2, 3, 4, 1))
        xr_dict = local._arr_to_xr({"mock_comp": arr})

        da_out = xr_dict["mock_comp"]
        assert list(da_out.dims) == ["climate_member", "process_member", "time", "site"]
        assert len(da_out.climate_member) == 2
        assert len(da_out.process_member) == 3
