from __future__ import annotations

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
    start_yr: int
    end_yr: int
    baseline_start: int
    baseline_end: int
    n_years: int
    nt: int
    num_members: int
    dtype: np.dtype = np.float32


@dataclass
class SpatialState:
    """Everything a spatial component needs for one gridded run of ``Spatial``.

    Built by the model and passed to each component's ``project_separable``.
    ``grid_interpolation`` is recorded but not yet applied (regridding is
    always linear).
    """

    grid_lats: np.ndarray
    grid_lons: np.ndarray
    n_years: int
    num_spatial_members: int
    grid_interpolation: str
    endofhistory: int
    output_percentiles: list[int] | np.ndarray
    baseline_yrs: tuple[int, int]
    dtype: np.dtype = np.float32
    # Indices of the sampled members in the flattened (climate, process) global
    # ensemble, shared by all components so members stay consistent.
    member_indices: np.ndarray | None = None
    # Uniform [0, 1) draw per member for choosing GRD fingerprints, shared by
    # Fingerprint components (one GRD model per member) and independent
    # of member_indices.
    fingerprint_fraction: np.ndarray | None = None
    # Chunk plan for (members, time, *spatial) arrays, set by the model.
    time_chunk: int = -1
    spatial_chunks: tuple[int, ...] = (-1,)


@dataclass
class LocalState:
    """Everything a spatial component needs for one site run of ``Local``.

    Built by the model and passed to each component's ``project_separable``.
    ``interpolation_method`` is recorded but not yet applied (sites use a
    land-aware four-cell interpolation).
    """

    target_lats: list[float]
    target_lons: list[float]
    n_years: int
    num_spatial_members: int
    interpolation_method: str
    endofhistory: int
    output_percentiles: list[int] | np.ndarray
    baseline_yrs: tuple[int, int]
    dtype: np.dtype = np.float32
    # Indices of the sampled members in the flattened (climate, process) global
    # ensemble, shared by all components so members stay consistent.
    member_indices: np.ndarray | None = None
    # Uniform [0, 1) draw per member for choosing GRD fingerprints, shared by
    # Fingerprint components (one GRD model per member) and independent
    # of member_indices.
    fingerprint_fraction: np.ndarray | None = None
    # Chunk plan for (members, time, *spatial) arrays, set by the model.
    time_chunk: int = -1
    spatial_chunks: tuple[int, ...] = (-1,)
