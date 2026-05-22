from pathlib import Path
import dask.array as da
import xarray as xr
import numpy as np

from profsea.components.core.base import SpatialComponent
from profsea.components.core.state import SpatialState
from profsea.utils import interpolate_to_grid

class GIA(SpatialComponent):
    """
    Handles Glacial Isostatic Adjustment.
    GIA accumulates linearly over time and has no global warming driver input.
    """
    def __init__(
        self, 
        gia_path: str, 
        baseline_yrs: tuple = (1986, 2005)
    ):
        self.gia_path = Path(gia_path)
        self.baseline_yrs = baseline_yrs
        
        if not self.gia_path.exists():
            raise FileNotFoundError(f"Clean GIA NetCDF missing: {self.gia_path}")
            
        # Dummy property required by the base Spatial architecture
        self._global_projection = da.zeros((1, 1))

    @property
    def global_projection(self):
        return self._global_projection

    def project(self, state: SpatialState, rng: np.random.Generator) -> da.Array:
        # 1. Load the NetCDF lazily and interpolate to the target user grid
        gia_da = xr.open_dataarray(self.gia_path, chunks={"lat": 45, "lon": 45})
        interp_da = interpolate_to_grid(gia_da, state.grid_lats, state.grid_lons)
        
        # Shape is now (n_models, target_lat, target_lon)
        gia_rates = interp_da.data
        n_models = gia_rates.shape[0]

        # 2. Calculate the accumulation time vector (mm/yr to m/yr)
        midyr = (self.baseline_yrs[1] - self.baseline_yrs[0] + 1) * 0.5 + self.baseline_yrs[0]
        Tdelta = 2006 - midyr
        unit_series = (np.arange(state.n_years) + Tdelta) * 0.001
        
        # 3. Randomly sample the GIA models for each Monte Carlo member
        rgiai = rng.integers(n_models, size=state.n_members)
        selected_gia = gia_rates[rgiai, :, :] # Shape: (members, lat, lon)
        
        # 4. Multiply accumulation time by spatial rates
        # (years) * (members, lat, lon) -> broadcasts to (members, years, lat, lon)
        spatial_projection = unit_series[None, :, None, None] * selected_gia[:, None, :, :]
        
        return spatial_projection