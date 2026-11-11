import pytest
import xarray as xr

from profsea.components.core.observations import Observations


@pytest.fixture
def observations():
    return Observations(
        baseline_start=1995,
        baseline_end=2014,
    )


@pytest.fixture
def dummy_obs():
    return xr.Dataset({"sle": ("time", [0.0, 1.0, 2.0])})


def test_greenland_uses_default_baseline(
    monkeypatch,
    observations,
    dummy_obs,
):
    captured = {}

    def mock_read_greenland_obs(
        baseline_start,
        baseline_end,
        component,
    ):
        captured["baseline_start"] = baseline_start
        captured["baseline_end"] = baseline_end
        captured["component"] = component

        return dummy_obs

    monkeypatch.setattr(
        "profsea.components.core.observations.read_greenland_obs",
        mock_read_greenland_obs,
    )

    result = observations.greenland()

    assert captured == {
        "baseline_start": 1995,
        "baseline_end": 2014,
        "component": "total",
    }
    assert result is dummy_obs


def test_greenland_allows_baseline_override(
    monkeypatch,
    observations,
    dummy_obs,
):
    captured = {}

    def mock_read_greenland_obs(
        baseline_start,
        baseline_end,
        component,
    ):
        captured["baseline_start"] = baseline_start
        captured["baseline_end"] = baseline_end
        captured["component"] = component
        return dummy_obs

    monkeypatch.setattr(
        "profsea.components.core.observations.read_greenland_obs",
        mock_read_greenland_obs,
    )

    observations.greenland(
        baseline_start=1986,
        baseline_end=2005,
        component="dyn",
    )

    assert captured == {
        "baseline_start": 1986,
        "baseline_end": 2005,
        "component": "dyn",
    }


def test_antarctica_passes_component_and_region(
    monkeypatch,
    observations,
    dummy_obs,
):
    captured = {}

    def mock_read_antarctic_obs(
        baseline_start,
        baseline_end,
        component,
        region,
    ):
        captured["baseline_start"] = baseline_start
        captured["baseline_end"] = baseline_end
        captured["component"] = component
        captured["region"] = region
        return dummy_obs

    monkeypatch.setattr(
        "profsea.components.core.observations.read_antarctic_obs",
        mock_read_antarctic_obs,
    )

    observations.antarctica(
        component="dyn",
        region="EAIS",
    )

    assert captured == {
        "baseline_start": 1995,
        "baseline_end": 2014,
        "component": "dyn",
        "region": "EAIS",
    }


def test_observations_are_cached(
    monkeypatch,
    observations,
    dummy_obs,
):
    calls = 0

    def mock_read_greenland_obs(
        baseline_start,
        baseline_end,
        component,
    ):
        nonlocal calls
        calls += 1
        return dummy_obs

    monkeypatch.setattr(
        "profsea.components.core.observations.read_greenland_obs",
        mock_read_greenland_obs,
    )

    first = observations.greenland()
    second = observations.greenland()

    assert calls == 1
    assert first is second


def test_different_requests_are_cached_separately(
    monkeypatch,
    observations,
    dummy_obs,
):
    calls = 0

    def mock_read_greenland_obs(
        baseline_start,
        baseline_end,
        component,
    ):
        nonlocal calls
        calls += 1
        return dummy_obs.copy()

    monkeypatch.setattr(
        "profsea.components.core.observations.read_greenland_obs",
        mock_read_greenland_obs,
    )

    observations.greenland(component="total")
    observations.greenland(component="dyn")
    observations.greenland(component="dyn")

    assert calls == 2


@pytest.mark.parametrize(
    ("method", "reader"),
    [
        ("landwater", "read_landwater_obs"),
        ("glacier", "read_glacier_obs"),
        ("thermal_expansion", "read_thermal_expansion_obs"),
        ("gmsl", "read_GMSL_obs"),
    ],
)
def test_simple_observation_methods_use_baseline(
    monkeypatch,
    observations,
    dummy_obs,
    method,
    reader,
):
    captured = {}

    def mock_reader(baseline_start, baseline_end):
        captured["baseline_start"] = baseline_start
        captured["baseline_end"] = baseline_end
        return dummy_obs

    monkeypatch.setattr(
        f"profsea.components.core.observations.{reader}",
        mock_reader,
    )

    result = getattr(observations, method)()

    assert captured == {
        "baseline_start": 1995,
        "baseline_end": 2014,
    }
    assert result is dummy_obs
