from dataclasses import dataclass
import numpy as np


@dataclass
class ClimateState:
    """ClimateState context object to hold relevant state information for SLR projections."""

    scenario: str
    T_ens: np.ndarray
    T_int_ens: np.ndarray
    T_int_med: np.ndarray
    fraction: np.ndarray  # shared correlation array
    palmer_method: bool  # whether to use Palmer method for time projection
    endofAR5: int
    endofhistory: int
    end_yr: int
    nyr: int
    nt: int
    num_members: int


@dataclass
class SpatialState:
    """SpatialState context object to hold relevant state information for spatial SLR projections."""

    grid_lats: np.ndarray
    grid_lons: np.ndarray
    n_years: int
    n_members: int
    grid_interpolation: str
    output_percentiles: list[int] | np.ndarray
    baseline_yrs: tuple[int, int]


@dataclass
class LocalState:
    """LocalState context object to hold relevant state information for local SLR projections."""

    target_lats: list[float]
    target_lons: list[float]
    n_years: int
    n_members: int
    interpolation_method: str
    output_percentiles: list[int] | np.ndarray
    baseline_yrs: tuple[int, int]
