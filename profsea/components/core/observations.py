import xarray as xr

from profsea.utils import (
    read_antarctic_obs,
    read_glacier_obs,
    read_GMSL_obs,
    read_greenland_obs,
    read_landwater_obs,
    read_thermal_expansion_obs,
)


class Observations:
    """Manage observational data used to initialise sea-level projections.

    Parameters
    ----------
    baseline_start : int
        First year of the period used to define the sea-level baseline.
    baseline_end : int
        Last year of the period used to define the sea-level baseline.

    Notes
    -----
    Observational datasets are cached after first access and reused for
    subsequent requests.
    """

    def __init__(
        self,
        baseline_start: int,
        baseline_end: int,
    ):
        self.baseline_start = baseline_start
        self.baseline_end = baseline_end
        self._cache = {}

    def antarctica(
        self,
        component: str = "total",
        region: str | None = None,
        baseline_start: int | None = None,
        baseline_end: int | None = None,
    ) -> xr.Dataset:

        if baseline_start is None:
            baseline_start = self.baseline_start

        if baseline_end is None:
            baseline_end = self.baseline_end

        key = (
            "antarctica",
            component,
            region,
            baseline_start,
            baseline_end,
        )

        if key not in self._cache:
            self._cache[key] = read_antarctic_obs(
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                component=component,
                region=region,
            )

        return self._cache[key]

    def greenland(
        self,
        component: str = "total",
        baseline_start: int | None = None,
        baseline_end: int | None = None,
    ) -> xr.Dataset:

        if baseline_start is None:
            baseline_start = self.baseline_start

        if baseline_end is None:
            baseline_end = self.baseline_end

        key = (
            "greenland",
            component,
            baseline_start,
            baseline_end,
        )

        if key not in self._cache:
            self._cache[key] = read_greenland_obs(
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                component=component,
            )

        return self._cache[key]

    def landwater(
        self,
        baseline_start: int | None = None,
        baseline_end: int | None = None,
    ) -> xr.Dataset:

        if baseline_start is None:
            baseline_start = self.baseline_start

        if baseline_end is None:
            baseline_end = self.baseline_end

        key = ("landwater", baseline_start, baseline_end)

        if key not in self._cache:
            self._cache[key] = read_landwater_obs(
                baseline_start=baseline_start,
                baseline_end=baseline_end,
            )

        return self._cache[key]

    def glacier(
        self,
        baseline_start: int | None = None,
        baseline_end: int | None = None,
    ) -> xr.Dataset:

        if baseline_start is None:
            baseline_start = self.baseline_start

        if baseline_end is None:
            baseline_end = self.baseline_end

        key = ("glacier", baseline_start, baseline_end)

        if key not in self._cache:
            self._cache[key] = read_glacier_obs(
                baseline_start=baseline_start,
                baseline_end=baseline_end,
            )

        return self._cache[key]

    def thermal_expansion(
        self,
        baseline_start: int | None = None,
        baseline_end: int | None = None,
    ) -> xr.Dataset:

        if baseline_start is None:
            baseline_start = self.baseline_start

        if baseline_end is None:
            baseline_end = self.baseline_end

        key = ("thermal_expansion", baseline_start, baseline_end)

        if key not in self._cache:
            self._cache[key] = read_thermal_expansion_obs(
                baseline_start=baseline_start,
                baseline_end=baseline_end,
            )

        return self._cache[key]

    def gmsl(
        self,
        baseline_start: int | None = None,
        baseline_end: int | None = None,
    ) -> xr.Dataset:

        if baseline_start is None:
            baseline_start = self.baseline_start

        if baseline_end is None:
            baseline_end = self.baseline_end

        key = ("gmsl", baseline_start, baseline_end)

        if key not in self._cache:
            self._cache[key] = read_GMSL_obs(
                baseline_start=baseline_start,
                baseline_end=baseline_end,
            )

        return self._cache[key]
