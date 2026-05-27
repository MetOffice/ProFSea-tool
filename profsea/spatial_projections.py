"""
Copyright (c) 2023, Met Office
All rights reserved.
"""
import glob
import pickle
import json
import os
from pathlib import Path
import warnings

import dask.array as da
from netCDF4 import Dataset
import numpy as np
from rich.console import Console
from rich.progress import track
import xarray as xr

from profsea.config import settings
from profsea.directories import read_dir
from profsea.emulator import Global
from profsea.utils import sample_members_2D, interpolate
from profsea.slr_pkg import choose_montecarlo_dir

console = Console()
warnings.filterwarnings("ignore")

def calc_baseline_period(yrs: np.array) -> float:
    """
    Calculate the offset between projection start and AR5 baseline midpoint.

    Parameters
    ----------
    yrs : numpy.ndarray
        Projection years.

    Returns
    -------
    float
        Difference between the first projection year and the midpoint of the
        1986-2005 baseline period.
    """
    byr1 = 1986.
    byr2 = 2005.

    console.log("Baseline period = ", byr1, "to", byr2)
    midyr = (byr2 - byr1 + 1) * 0.5 + byr1
    return yrs[0] - midyr


def calc_future_sea_level(scenario: str) -> None:
    """
    Calculate and save future spatial sea-level projections for a scenario.

    Parameters
    ----------
    scenario : str
        Emissions scenario identifier.

    Returns
    -------
    None
        Outputs are written to configured NetCDF files.
    """
    # Set the UKCP*18* random seed so results are reproducible
    np.random.seed(18)

    # Directory of Monte Carlo time series for new projections
    mcdir = choose_montecarlo_dir()

    # Specify the sea level components to include. The GIA contribution is
    # calculated separately.
    components = ['expansion', 'antdyn', 'antsmb', 'greenland',
                  'glacier', 'landwater']

    # Select dimensions from sample file, [time, realisation]
    sample = np.load(os.path.join(settings["baseoutdir"],settings["experiment_name"],
                                  settings['emulator_settings']['gmslr_output_dir'], f'{scenario}_expansion.npy'))
    
    nesm = sample.shape[0] # also number of samples to make
    nyrs = sample.shape[1]
    
    yrs = np.arange(2006, 2006 + nyrs)
    console.log(f"Running with {nesm} ensemble members")

    grid_path = os.path.join(
        settings["cmipinfo"]["sealevelbasedir"], 
        "ACCESS-CM2/zos_regression_ssp245_ACCESS-CM2.npy")
    grid_sample = np.load(grid_path)
    array_dims = [nesm, nesm, nyrs, grid_sample.shape[0], grid_sample.shape[1]]

    console.log(
        "INFO: This module expects sterodynamic patterns on half-integer grids:\n"
        "\tlat: (-89.5, ..., 89.5)\n"
        "\tlon: (-179.5, ..., 179.5)")

    # Get random samples of global and regional sea level components
    calculate_sl_components(mcdir, components, scenario, yrs, array_dims)


def calc_gia_contribution(
        yrs: np.array, nyrs: int, nsmps: int, 
        scenario: str) -> None:
    """
    Calculate and save glacial isostatic adjustment (GIA) regional projections.

    Parameters
    ----------
    yrs : numpy.ndarray
        Projection years.
    nyrs : int
        Number of years in each projection time series.
    nsmps : int
        Number of samples to generate.
    scenario : str
        Emissions scenario identifier.

    Returns
    -------
    None
        GIA projections are written to a NetCDF output file.
    """
    console.log('Calculating GIA contribution...')
    nGIA, GIA_vals = read_gia_estimates()
    Tdelta = calc_baseline_period(yrs)

    # Unit series of mm/yr expressed as m/yr
    unit_series = (np.arange(nyrs) + Tdelta) * 0.001
    GIA_unit_series = np.ones([nsmps, nyrs]) * unit_series

    # rgiai is an array of random GIA indices the size of the sample years
    rgiai = np.random.randint(nGIA, size=nsmps)

    GIA_T = da.from_array(GIA_unit_series)
    GIA_vals = da.from_array(GIA_vals)
    GIA_series = GIA_T[:, :, None, None] * GIA_vals[rgiai, None, :, :]

    file_header = '_'.join(['gia', scenario, "projection", 
                    f"{settings['projection_end_year']}"])
    sealev_ddir = os.path.join(settings["baseoutdir"],settings["experiment_name"],
                               settings['emulator_settings']['spatial_output_dir'])

    # Save data in netcdf format (Assuming first dimension is percentile, but can be more general percentile/ensemble)
    xr_dataArray = xr.DataArray(
        GIA_series, 
        dims=["samples", "time", "lat", "lon"], 
        coords={
            "samples": np.arange(nsmps),
            "time": np.arange(2006, GIA_series.shape[1] + 2006),
            "lat": np.arange(-90, 90) + 0.5, 
            "lon": np.arange(0, 360) + 0.5})
    xr_dataArray.attrs["units"] = "m"
    xr_dataArray.attrs["long_name"] = "Regional GIA sea-level projections"
    ds = xr_dataArray.to_dataset(name='gia')

    ds.attrs["source"] = "ProFSea-Climate v0.1"

    R_file = '_'.join([file_header, 'regional']) + '.nc'
    encoding = {'gia': {"zlib": True, "complevel": 5, "dtype": "float32"}}
    ds.to_netcdf(os.path.join(sealev_ddir, R_file), encoding=encoding, compute=True)
    
    del GIA_series


def calc_expansion_contribution(
        scenario: str, nsmps: int) -> da.array:
    """
    Sample thermal expansion coefficients for regional projections.

    Parameters
    ----------
    scenario : str
        Emissions scenario identifier.
    nsmps : int
        Number of samples to draw.

    Returns
    -------
    dask.array.Array
        Sampled expansion coefficients on the target grid.
    """
    # Select slope coefficients based on the MIP
    if settings["emulator_settings"]["emulator_mode"]:
        if settings["cmipinfo"]["mip"].lower() == "cmip6":
            coeffs = load_CMIP6_slopes('ssp585')
            coeffs = da.roll(coeffs, 180, axis=2)
        else:
            coeffs = load_CMIP5_slope_coeffs('rcp85')
    else:
        coeffs = load_CMIP5_slope_coeffs(scenario)

    rand_samples = np.random.choice(
        coeffs.shape[0], size=nsmps, replace=True)               
    rand_coeffs = coeffs[rand_samples, :, :]
    return rand_coeffs


def calc_landwater_contribution(data: dict, lats: int, lons: int) -> da.array:
    """
    Interpolate and re-grid landwater contribution fields.

    Parameters
    ----------
    data : dict
        Landwater input data used for interpolation.
    lats : int
        Number of latitude points in the target grid.
    lons : int
        Number of longitude points in the target grid.

    Returns
    -------
    dask.array.Array
        Landwater contribution values on the target grid.
    """
    landwater_vals = interpolate(data, lats, lons)
    landwater_vals = da.roll(landwater_vals, 180, axis=1)
    return landwater_vals


def calc_fingerprint_contributions(
    FPlist: list, comp: str, lats: int, lons: int) -> da.array:
    """
    Interpolate and stack fingerprints for a selected component.

    Parameters
    ----------
    FPlist : list
        List of fingerprint dictionaries.
    comp : str
        Component key to extract from each fingerprint dictionary.
    lats : int
        Number of latitude points in the target grid.
    lons : int
        Number of longitude points in the target grid.

    Returns
    -------
    dask.array.Array
        Stacked fingerprint values for the selected component.
    """
    # Initiate an empty list for fingerprint values
    fp_vals = []
    for FP_dict in FPlist:
        # Interpolate values to target lat/lon
        val = FP_dict[comp]
        val = interpolate(val, lats, lons)
        fp_vals.append(val)

    fp_vals = da.stack(fp_vals, axis=0)
    return fp_vals


def calc_greenland_fingerprint_ar6(lats: int, lons: int) -> da.array:
    """Load and prepare the AR6 Greenland fingerprint.

    Parameters
    ----------
    lats : int
        Number of latitude points in the target grid.
    lons : int
        Number of longitude points in the target grid.

    Returns
    -------
    dask.array.Array
        Greenland fingerprint on the target grid.
    """
    # Load in the fingerprint
    fp_path = Path(settings["fingerprints"]) / "greenland_ar6.nc"
    fp_ds = xr.open_dataset(fp_path, chunks={})

    # Interpolate to (180, 360) grid
    fp_vals = fp_ds.fp.interp(
        lat=np.linspace(-90, 90, lats, endpoint=False) + 0.5, 
        lon=np.linspace(0, 360, lons, endpoint=False) + 0.5, 
        method="linear").data * 1000  # convert mm to m SLE per m GMSLR

    # Flip vertically and roll by 180 degrees
    fp_vals = da.flip(fp_vals)
    fp_vals = da.roll(fp_vals, 180, axis=1)
    return fp_vals


def save_projections(
        montecarlo_R: da.array, component: str, scenario: str, percentile: da.array) -> None:
    """
    Save regional sea-level projections for one component to NetCDF.

    Parameters
    ----------
    montecarlo_R : dask.array.Array
        Regional sea-level projections.
    component : str
        Sea-level component name.
    scenario : str
        Emissions scenario identifier.
    percentile : dask.array.Array or numpy.ndarray
        Percentiles represented in the first dimension.

    Returns
    -------
    None
        Data are written to a component-specific NetCDF file.
    """
    sealev_ddir = os.path.join(settings["baseoutdir"],settings["experiment_name"],
                               settings['emulator_settings']['spatial_output_dir'])
    file_header = '_'.join([component, scenario, "projection", 
                            f"{settings['projection_end_year']}"])

    # Save data in netcdf format (Assuming first dimension is percentile, but can be more general percentile/ensemble)
    xr_dataArray = xr.DataArray(montecarlo_R, dims=["percentile", "time", "lat", "lon"], 
                                coords={"percentile": percentile, 
                                        "time": np.arange(2006, montecarlo_R.shape[1] + 2006),
                                        "lat": np.arange(-90, 90) + 0.5, "lon": np.arange(0, 360) + 0.5})
    xr_dataArray.attrs["units"] = "m"
    xr_dataArray.attrs["long_name"] = f"Regional {component} sea-level projections"
    ds = xr_dataArray.to_dataset(name=component)

    ds.attrs["source"] = "ProFSea-Climate v0.1"

    R_file = '_'.join([file_header, 'regional']) + '.nc'
    encoding = {component: {"zlib": True, "complevel": 5, "dtype": "float32"}}
    ds.to_netcdf(os.path.join(sealev_ddir, R_file), encoding=encoding, compute=True)


def calculate_sl_components(
        mcdir: str, components: list, scenario: str, 
        yrs: np.array, array_dims: list) -> None:
    """
    Calculate regional contributions for all selected sea-level components.

    Parameters
    ----------
    mcdir : str
        Directory containing Monte Carlo time series inputs.
    components : list
        Sea-level components to project.
    scenario : str
        Emissions scenario identifier.
    yrs : numpy.ndarray
        Projection years.
    array_dims : list
        Dimensions in the order [nesm, nsmps, nyrs, lats, lons].

    Returns
    -------
    None
        Component-specific NetCDF outputs are written to disk.
    """  
    # Numbers of ensemble members, samples, years
    nesm, nsmps, nyrs, lats, lons = array_dims
    nFPs, FPlist = load_fingerprints(components)
    resamples = np.random.choice(nesm, nsmps) # Preserve correlations across comps
    rfpi = np.random.randint(nFPs, size=nsmps)
    # Take the 0th, 25th, 50th, 75th and 100th percentiles
    output_percentiles = np.array([0, 25, 50, 75, 100])

    # Calculate GIA contribution and save it out
    calc_gia_contribution(yrs, nyrs, len(output_percentiles), scenario)

    for comp in track(components, description="Calculating components..."):
        montecarlo_R = da.zeros((nsmps, nyrs, lats, lons), dtype=np.float32) # (FPs applied) + GIA
        montecarlo_G = da.zeros((nsmps, nyrs, lats, lons), dtype=np.float32) # (no FPs applied)

        # Load global projections in for the component
        #mc_timeseries = np.load(os.path.join(mcdir, f'{scenario}_{comp}.npy'))
        mc_timeseries = np.load(os.path.join(settings["baseoutdir"],settings["experiment_name"],
                                             settings['emulator_settings']['gmslr_output_dir'],f'{scenario}_{comp}.npy'))
        sampled_mc = mc_timeseries[resamples, :nyrs]
        montecarlo_G[:, :] = da.from_array(sampled_mc[:, :, None, None], chunks="auto")

        if comp == "expansion":
            sampled_coeffs = calc_expansion_contribution(scenario, nsmps)
            montecarlo_R = montecarlo_G * sampled_coeffs[:, None, :, :]
            del sampled_coeffs

        elif comp == "landwater":
            landwater_vals = calc_landwater_contribution(FPlist[0]["landwater"], lats, lons)
            montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * landwater_vals[None, None, :, :]
            del landwater_vals

        elif comp == "greenland":
            greenland_fp = calc_greenland_fingerprint_ar6(lats, lons)
            montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * greenland_fp[None, None, :, :]

        else:
            fp_vals = calc_fingerprint_contributions(FPlist, comp, lats, lons)
            montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * fp_vals[rfpi][:, None, :, :]
            del fp_vals

        montecarlo_R = da.percentile(montecarlo_R, output_percentiles, axis=0)
        montecarlo_R = montecarlo_R.astype(np.float32)

        # Create the output sea level projections file directory and filename
        save_projections(montecarlo_R, comp, scenario, output_percentiles)


def get_projection_info(indir: str, scenario: str) -> tuple:
    """
    Read dimensions and years from a Monte Carlo NetCDF sample file.

    Parameters
    ----------
    indir : str
        Input directory containing Monte Carlo files.
    scenario : str
        Emissions scenario identifier.

    Returns
    -------
    tuple
        Tuple of (number of ensemble members, number of years, years array).
    """
    sample_file = f'{scenario}_exp.nc'
    f = Dataset(f'{indir}{sample_file}', 'r')
    nesm = f.dimensions['realization'].size
    t = f.variables['time']
    nyrs = t.size
    unit_str = t.units
    first_year = int(unit_str.split(' ')[2][:4])
    f.close()

    yrs = first_year + np.arange(nyrs)
    return nesm, nyrs, yrs


def load_CMIP5_slope_coeffs(scenario: str) -> np.ndarray:
    """
    Load CMIP5 slope coefficients for a selected scenario.

    Parameters
    ----------
    scenario : str
        Emissions scenario identifier.

    Returns
    -------
    numpy.ndarray
        Regression slope coefficients on the spatial grid.
    """
    # Read in the sea level regressions
    in_zosddir = read_dir()[2]
    filename = os.path.join(in_zosddir, 'zos_regression.npy')
    coeffs = np.load(filename)
    scenario_index = ['rcp26', 'rcp45', 'rcp85'].index(scenario)
    coeffs = coeffs[:, scenario_index, :, :]
    coeffs[np.isnan(coeffs)] = 0
    coeffs[coeffs > 999] = 0
    coeffs[coeffs < -999] = 0
    return coeffs


def load_CMIP6_slopes(scenario: str) -> np.ndarray:
    """
    Load CMIP6 slope coefficients for a selected scenario.

    Parameters
    ----------
    scenario : str
        Emissions scenario identifier.

    Returns
    -------
    dask.array.Array
        Stacked slope coefficients across available models.
    """
    # Read in the sea level regressions
    cmip6_dir = settings["cmipinfo"]["sealevelbasedir"]
    slope_files = glob.glob(cmip6_dir + f'/*/zos_regression_{scenario}_*.npy')

    def load_one_slope(f):
        return np.load(f, mmap_mode='r')

    # Create a list of lazy dask arrays
    lazy_slopes = [
        da.from_array(load_one_slope(f), chunks=(180, 360)) 
        for f in slope_files]
    slopes_stack = da.stack(lazy_slopes, axis=0)

    return slopes_stack


def read_gia_estimates() -> tuple:
    """
    Read and prepare pre-processed GIA estimate fields.

    Returns
    -------
    tuple
        Tuple of (number of GIA fields, array of GIA values).
    """
    gia_file = settings["giaestimates"]["global"]
    with open(gia_file, "rb") as ifp:
        GIA_dict = pickle.load(ifp, encoding='latin1') # Interp objects

    GIA_vals = []
    for key in list(GIA_dict.keys()):
        val = GIA_dict[key].values
        GIA_vals.append(val)

    nGIA = len(GIA_vals)
    GIA_vals = np.array(GIA_vals)

     # Sort out the crazy values in the 0th GIA array
    GIA_vals[0][GIA_vals[0] < -99999] = 0

    # AND shift them from -180, 180 to 0, 360
    GIA_vals = np.roll(GIA_vals, 180, axis=2)
    return nGIA, GIA_vals


def load_fingerprints(components: list) -> tuple:
    """
    Load component fingerprints from multiple fingerprint datasets.

    Parameters
    ----------
    components : list
        Sea-level components to load.

    Returns
    -------
    tuple
        Tuple of (number of fingerprint sets, list of fingerprint dictionaries).
    """
    # Create empty dictionaries for the Slangen, Spada and Klemann fingerprints
    # interpolator objects.
    slangen_FPs = {}
    spada_FPs = {}
    klemann_FPs = {}

    # Only 1 fingerprint for Landwater
    comp = "landwater"
    slangen_FPs[comp] = xr.open_dataarray(
        os.path.join(settings["fingerprints"],
        comp + "_slangen_nomask.nc"), chunks={})

    # Other FPs have multiple components
    components_todo = [
        c for c in components 
        if c not in ["expansion", "landwater", "greenland"]]
    for comp in components_todo:
        slangen_FPs[comp] = xr.open_dataarray(
            os.path.join(settings["fingerprints"],
            comp + "_slangen_nomask.nc"), chunks={})
        spada_FPs[comp] = xr.open_dataarray(
            os.path.join(settings["fingerprints"],
            comp + "_spada_nomask.nc"), chunks={})
        klemann_FPs[comp] = xr.open_dataarray(
            os.path.join(settings["fingerprints"],
            comp + "_klemann_nomask.nc"), chunks={})

    FPlist = [slangen_FPs, spada_FPs, klemann_FPs]
    nFPs = len(FPlist)
    return nFPs, FPlist


def calculate_global_components(scenario: str, palmer_method: bool) -> None:
    """
    Calculate global component projections using the GMSLR emulator.

    Parameters
    ----------
    scenario : str
        Scenario being simulated.
    palmer_method : bool
        Whether to apply the Palmer method beyond 2100.

    Returns
    -------
    None
        Global component outputs are saved to disk.
    """
    # Check inputs are correctly set up
    if not (os.path.exists(settings["scm_data"]["temperature"]) and 
            os.path.exists(settings["scm_data"]["ocean_heat_content"])):
        raise Exception(
            'SCM data paths (temperature and ocean heat content) must be '
            'correctly configured in user-settings.yml')

    if not os.path.exists(settings["scm_data"]["cumulative_emissions"]):
        raise Exception('Cumulative emissions path must be correctly configured '
                        'in user-settings.yml')

    if (settings["scm_data"]["temperature"].split(".")[-1] != 'nc' or
        settings["scm_data"]["ocean_heat_content"].split(".")[-1] != 'nc'):
        raise Exception('SCM data must be saved in NetCDF format.')

    percentiles = np.arange(101)
    # percentiles are hard coded. We could make it an user input in future updates.

    # Now run the simulations
    console.log(f'Projecting global components for {scenario} scenario...')
    T_change = xr.load_dataarray(settings["scm_data"]["temperature"])
    OHC_change = xr.load_dataarray(settings["scm_data"]["ocean_heat_content"])
    with open(settings["scm_data"]["cumulative_emissions"]) as f:
        cumulative_emissions = json.load(f)

    T_change = T_change.sel(scenario=scenario).data.T # (member, time)
    OHC_change = OHC_change.sel(scenario=scenario).data.T

    T_change = sample_members_2D(T_change, percentiles)
    OHC_change = sample_members_2D(OHC_change, percentiles)

    gmslr = Global(
        T_change,
        OHC_change,
        scenario,
        settings["projection_end_year"],
        palmer_method=palmer_method,
        input_ensemble=settings["emulator_settings"]["use_input_ensemble"],
        output_percentiles=percentiles,
        cum_emissions_total=cumulative_emissions[scenario])
    gmslr.project()

    console.log('Saving global components...')
    gmslr.save_components(
        os.path.join(settings["baseoutdir"],settings["experiment_name"],
                     settings['emulator_settings']['gmslr_output_dir']),
        scenario)


    console.log('Saved!\n')



def main():
    """
    Run end-to-end global and regional sea-level projection workflows.

    Returns
    -------
    None
        Projection files are generated in configured output directories.
    """
    console.log(f'\nProjecting out to: {settings["projection_end_year"]}\n')

    # Sort out paths
    Path(
        os.path.join(
            settings["baseoutdir"], 
            settings["experiment_name"])
    ).mkdir(parents=True, exist_ok=True)

    Path(
        os.path.join(
            settings["baseoutdir"],
            settings["experiment_name"],
            settings['emulator_settings']['gmslr_output_dir'])
    ).mkdir(parents=True, exist_ok=True)

    Path(
        os.path.join(
            settings["baseoutdir"],
            settings["experiment_name"],
            settings['emulator_settings']['spatial_output_dir'])
    ).mkdir(parents=True, exist_ok=True)

    # Extract site data from station list (e.g. tide gauge location) or
    # construct based on user input
    if settings["emulator_settings"]["emulator_mode"]:
        console.log('\nInitiating ProFSea emulator')
        if settings["projection_end_year"] > 2100:
            palmer_method = True
        else:
            palmer_method = False

        # Get the metadata of either the site location or tide gauge location
        for scenario in settings["emulator_settings"]["emulator_scenario"]:
            calculate_global_components(scenario, palmer_method)
            calc_future_sea_level(scenario)
    else:
        scenarios = ['rcp26', 'rcp45', 'rcp85']
        for scenario in scenarios:
            calc_future_sea_level(scenario)


if __name__ == '__main__':
    main()
