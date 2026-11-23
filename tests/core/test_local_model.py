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
        return da.ones(
            (state.num_spatial_members, state.n_years, len(state.target_lats))
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
        """Full output has one local Monte Carlo axis."""
        components = {"mock_comp": MockLocalComponent()}
        locations = {"Newlyn": (50.1, -5.5)}
        local = Local(
            components=components,
            locations=locations,
            end_year=2010,
            output_percentiles=None,
            num_spatial_members=7,
        )

        assert local.num_output_members == 7

        da_out = local.run(member_seed=1)["mock_comp"]
        assert list(da_out.dims) == ["member", "time", "site"]
        assert da_out.shape == (7, 4, 1)
        # Labels survive the land mask and record each member's global origin
        assert da_out.climate_member.shape == da_out.process_member.shape == (7,)
        assert sorted(np.bincount(da_out.climate_member.values)) == [3, 4]


@pytest.mark.parametrize("percentiles", [None, []])
@patch("profsea.components.core.local_model.fetch_zenodo_fingerprints")
def test_run_full_spatial_member_ensemble(mock_fetch, percentiles):
    """Without percentiles, Local returns every member on a member axis, plus the total."""
    model = Local(
        components={"custom": MockLocalComponent()},
        locations={"Newlyn": (50.1, -5.5), "Aberdeen": (57.1, -2.1)},
        end_year=2010,
        output_percentiles=percentiles,
        num_spatial_members=7,
    )
    results = model.run(member_seed=42)
    assert set(results) == {"custom", "total_rsl"}
    assert results["custom"].dims == ("member", "time", "site")
    assert results["custom"].shape == (7, 4, 2)
    assert isinstance(results["custom"].data, da.Array)
    np.testing.assert_array_equal(results["custom"].compute(), np.ones((7, 4, 2)))
    np.testing.assert_array_equal(
        results["total_rsl"].compute(), results["custom"].compute()
    )


class PartlyMaskedComponent(MockLocalComponent):
    """Member 0 is NaN at every site (as when it draws a model that is masked
    there); the other members are valid."""

    def project(self, state, rng):
        values = np.ones(
            (state.num_spatial_members, state.n_years, len(state.target_lats))
        )
        values[0] = np.nan
        return da.from_array(values)


@pytest.mark.parametrize("percentiles", [None, [5, 50, 95]])
@patch("profsea.components.core.local_model.fetch_zenodo_fingerprints")
def test_universal_mask_does_not_drop_sites_for_one_masked_member(
    mock_fetch, percentiles
):
    """A site missing data for one member only is kept; it is not treated as land."""
    model = Local(
        components={"custom": PartlyMaskedComponent()},
        locations={"Newlyn": (50.1, -5.5)},
        end_year=2010,
        output_percentiles=percentiles,
        num_spatial_members=7,
    )
    result = model.run(member_seed=1)["custom"].compute()

    # Only member 0 is missing; the site itself must not be masked
    if percentiles is None:
        assert np.isnan(result.isel(member=0)).all()
        assert np.isfinite(result.isel(member=slice(1, None))).all()
    else:
        assert np.isfinite(result).all()
