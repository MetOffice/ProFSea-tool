"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import os
import iris
import glob
import json
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from netCDF4 import Dataset
import xarray as xr
import dask.array as da

from profsea.config import settings
from profsea.slr_pkg import choose_montecarlo_dir  # found in __init.py__
from profsea.directories import read_dir, makefolder
from emulator import GMSLREmulator


def calc_baseline_period(sci_method, yrs):
    """
    Baseline years used for UKCP18 -- 1981-2000 (offset required)
    Baseline years used for IPCC AR5 and Palmer et al 2020 -- 1986-2005
    :param sci_method: scientific method used to select GIA estimates, GRD
    fingerprints and CMIP regression (based on user input)
    :param yrs: years of the projections
    :return: baseline years
    """
    if sci_method == 'global':
        byr1 = 1986.
        byr2 = 2005.
        # offset not required
        G_offset = 0.0
        print("Baseline period = ", byr1, "to", byr2)
    elif sci_method == 'UK':
        byr1 = 1981.
        byr2 = 2000.
        G_offset = 0.011
        print("Baseline period = ", byr1, "to", byr2)

    midyr = (byr2 - byr1 + 1) * 0.5 + byr1

    return yrs[0] - midyr, G_offset


def calc_future_sea_level(scenario):
    """
    Calculates future sea level at the given site and write to file.
    :param df: Data frame of all metadata (tide gauge or site specific) for
               each(all) location(s)
    :param site_loc: name of the site location
    :param scenario: emission scenario
    """
    print('running function calc_future_sea_level')


    # Set the UKCP*18* random seed so results are reproducible
    np.random.seed(18)

    # Directory of Monte Carlo time series for new projections
    mcdir = choose_montecarlo_dir()

    # Specify the sea level components to include. The GIA contribution is
    # calculated separately.
    components = ['exp', 'antdyn', 'antsmb', 'greendyn', 'greensmb',
                  'glacier', 'landwater']

    # Select dimensions from sample file, [time, realisation]
    sample = np.load(os.path.join(mcdir, f'{scenario}_exp.npy'))
    nesm = sample.shape[0]
    nyrs = sample.shape[1]
    yrs = np.arange(2007, 2007 + nyrs)

    grid_path = os.path.join(settings["baseoutdir"], "full_earth/data/zos_regression/zos_regression.npy")
    grid_sample = np.load(grid_path)[0, 0]

    # Determine the number of samples you wish to make CHOGOM
    # project = 100000, UKCP18 = 200000
    if nesm >= 200000:
        nsmps = 200000
    else:
        nsmps = nesm
        
    array_dims = [nesm, nsmps, nyrs, grid_sample.shape[0], grid_sample.shape[1]]

    # Get random samples of global and regional sea level components
    calculate_sl_components(mcdir, components, scenario, yrs, array_dims)


def calc_gia_contribution(sci_method, yrs, nyrs, nsmps):
    """
    Calculate the glacial isostatic adjustment (GIA) contribution to the
    regional component of sea level rise.
    Option to use the generic global GIA estimates or use the GIA estimates
    developed for UK as part of UKCP18.
    :param sci_method: scientific method used to select GIA estimates, GRD
    fingerprints and CMIP regression (based on user input)
    :param yrs: years of the projections
    :param nyrs: number of years in each projection time series
    :param nsmps: determine the number of samples
    :param coords: coordinates of location of interest
    :return: GIA estimates converted to mm/yr
    """
    print('running function calc_gia_contribution')

    nGIA, GIA_vals = read_gia_estimates(sci_method)

    Tdelta, G_offset = calc_baseline_period(sci_method, yrs)

    # Unit series of mm/yr expressed as m/yr
    unit_series = (np.arange(nyrs) + Tdelta) * 0.001
    GIA_unit_series = np.ones([5, nyrs]) * unit_series

    # rgiai is an array of random GIA indices the size of the sample years
    rgiai = np.random.randint(nGIA, size=5)

    GIA_T = da.from_array(GIA_unit_series)
    GIA_vals = da.from_array(GIA_vals)

    GIA_series = GIA_T[:, :, None, None] * GIA_vals[rgiai, None, :, :]

    return GIA_series, G_offset


def calculate_sl_components(mcdir, components, scenario, yrs, array_dims):
    """
    Calculates global and regional component part contributions to sea level
    change.
    :param mcdir: location of Monte Carlo time series for new projections
    :param components: sea level components
    :param scenario: emission scenario
    :param yrs: years of the projections
    :param array_dims: Array of nesm, nsmps and nyrs
        nesm --> Number of ensemble members in time series
        nsmps --> Determine the number of samples you wish to make
        nyrs --> Number of years in each projection time series
    :return: montecarlo_G (global contribution to sea level rise) and
    montecarlo_R (regional contribution to sea level change)
    """
    print('running function calculate_sl_components')
    
    # Numbers of ensemble members, samples, years
    nesm, nsmps, nyrs, lats, lons = array_dims

    # Scientific method
    sci_method = settings["sciencemethod"]

    # Set up fingerprint interpolators for each component of sea level change
    # Note that the first entry in the list is the Slangen, which includes the
    # land water estimate
    nFPs, FPlist = setup_FP_interpolators(components, sci_method)

    # Use the same resampling across all sea level components (to preserve
    # correlations)
    resamples = np.random.choice(nesm, nsmps)
    rfpi = np.random.randint(nFPs, size=nsmps)

    # The "offset_slopes" scales the global_offset variables into the different
    # sea level components.
    # These offsets exist because a baseline period of 1981-2000 is used,
    # rather than the AR5 baseline period of 1986-2005.
    offset_slopes = {'exp': 0.405,
                     'antnet': 0.095,
                     'antsmb': -0.025,
                     'antdyn': 0.120,
                     'greennet': 0.125,
                     'greensmb': 0.040,
                     'greendyn': 0.085,
                     'glacier': 0.270,
                     'landwater': 0.105}

    # Calculate the GIA contribution to the regional component of sea level
    # change
    GIA_series, G_offset = calc_gia_contribution(sci_method, yrs, nyrs, nsmps)

    GIA_series = GIA_series.compute()

    sealev_ddir = read_dir()[4]
    file_header = '_'.join(['gia', scenario, "projection", 
                    f"{settings['projection_end_year']}"])
    R_file = '_'.join([file_header, 'regional']) + '.npy'
    
    np.save(os.path.join(sealev_ddir, R_file), GIA_series)
    del GIA_series

    for cc, comp in enumerate(components):
        # Monte Carlo for regional values (FPs applied) + GIA
        montecarlo_R = da.zeros((nsmps, nyrs, lats, lons), dtype=np.float32)
        # Monte Carlo for Global values (no FPs applied)
        montecarlo_G = da.zeros((nsmps, nyrs, lats, lons), dtype=np.float32)

        print(f'cc = {cc:d}, comp = {comp}')
        offset = G_offset * offset_slopes[comp]

        if settings["emulator_settings"]["emulator_mode"]:
            # Input timeseries provided as numpy objects
            mc_timeseries = np.load(os.path.join(mcdir, f'{scenario}_{comp}.npy'))
            offset_mc = mc_timeseries[resamples, :nyrs] + offset
            montecarlo_G[:, :] = da.from_array(offset_mc[:, :, None, None])
        else:
            cube = iris.load_cube(os.path.join(mcdir, f'{scenario}_{comp}.nc'))
            offset_mc = cube.data[:nyrs, resamples] + offset
            montecarlo_G[:, :] = offset_mc[:, :, None, None]

        if comp == 'exp':
            if sci_method == 'global':
                if settings["emulator_settings"]["emulator_mode"]:
                    coeffs = load_CMIP5_slope_coeffs('rcp26')
                else:
                    coeffs = load_CMIP5_slope_coeffs(scenario)
                rand_samples = np.random.choice(coeffs.shape[0], size=nsmps,
                                               replace=True)                         
                rand_coeffs = coeffs[rand_samples, :, :]
                rand_coeffs = da.from_array(rand_coeffs)
            elif sci_method == 'UK':
                if settings["emulator_settings"]["emulator_mode"]:
                    coeffs, weights = load_CMIP5_slope_coeffs_UK('rcp85')
                else:
                    coeffs, weights = load_CMIP5_slope_coeffs_UK(scenario)
                rand_coeffs = np.random.choice(coeffs, size=nsmps,
                                               replace=True, p=weights)

            montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * rand_coeffs[:, None, :, :]
            del rand_coeffs

        elif comp == 'landwater':
            landwater_FP_interpolator = FPlist[0]['landwater']
            # Interpolate values
            vals = da.from_array(landwater_FP_interpolator.values.astype(np.float32).data)
            vals = np.roll(vals, 180, axis=1)
            montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * vals[None, None, :, :]
        else:
            # Initiate an empty list for fingerprint values
            FPvals = []
            for FP_dict in FPlist:
                # Interpolate values to target lat/lon
                val = FP_dict[comp].values
                if val.shape != (lats, lons):
                    original_da = xr.DataArray(
                        val,
                        coords=[("lat", np.linspace(90, -90, 360)), ("lon", np.linspace(-180, 180, 720, endpoint=False))],
                        name="v")
                    target_lat = np.linspace(90, -90, 180)
                    target_lon = np.linspace(-180, 180, 360, endpoint=False)
                    val = original_da.interp(lat=target_lat, lon=target_lon, method="linear").data
                    val = np.roll(val, 180, axis=1)
                else:
                    val = np.roll(val, 180, axis=1)
                    
                FPvals.append(val)
            FPvals = da.from_array(np.array(FPvals, dtype=np.float32))
            montecarlo_R[:, :, :, :] = montecarlo_G[:, :, :, :] * FPvals[rfpi][:, None, :, :]
            del FPvals

        # Take the 0th, 25th, 50th, 75th and 100th percentiles
        montecarlo_R = da.percentile(montecarlo_R, [0, 25, 50, 75, 100], axis=0)

        montecarlo_R = montecarlo_R.compute()

        # Create the output sea level projections file directory and filename
        sealev_ddir = read_dir()[4]
        file_header = '_'.join([comp, scenario, "projection", 
                                f"{settings['projection_end_year']}"])
        # G_file = '_'.join([file_header, 'global']) + '.npy'
        R_file = '_'.join([file_header, 'regional']) + '.npy'

        # Save the global and local projections
        makefolder(sealev_ddir)
        # montecarlo_R.to_netcdf(os.path.join(sealev_ddir, R_file))
        # np.save(os.path.join(sealev_ddir, G_file), montecarlo_G)
        np.save(os.path.join(sealev_ddir, R_file), montecarlo_R)

    return montecarlo_R


def create_FP_interpolator(datadir, dfile, method='linear'):
    """
    Generates a scipy Interpolator object from input NetCDF data of
    gravitational fingerprints (takes inputs of Latitude and Longitude).
    :param datadir: data directory
    :param dfile: data filename
    :param method: interpolation type --> 'linear' or 'nearest'
    :return: 2D Interpolator object
    """
    cube = iris.load_cube(os.path.join(datadir, dfile))
    lon = cube.coord('longitude').points
    lat = cube.coord('latitude').points

    # Define linear interpolator object:
    interp_object = RegularGridInterpolator((lat, lon), cube.data,
                                            method=method, bounds_error=True,
                                            fill_value=None)

    return interp_object


def get_projection_info(indir, scenario):
    """
    Read in the dimensions of the Monte-Carlo data. These files are all
    relative to midnight on 1st January 2007
    :param indir: directory of Monte Carlo time series for new projections
    :param scenario: emission scenarios to be considered
    :return: Number of ensemble members in time series, number of years in
    each projection time series and the years of the projections
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


def load_CMIP5_slope_coeffs(scenario):
    """
    Loads in the CMIP slope coefficients based on linear regression of
    'zos+zostoga' against 'zostoga' for the period 2005 to 2100.
    Some models are missing regression slopes for RCP2.6. If so, use RCP4.5
    values instead.
    :param site_loc: name of the site location
    :param scenario: emissions scenario
    :return: 1D array of regression coefficients
    """
    print('running function load_CMIP5_slope_coeffs')
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


def load_CMIP5_slope_coeffs_UK(scenario):
    """
    Loads in the CMIP5 slope coefficients developed for UKCP18
    :param scenario: emissions scenario
    :return: 1D array of slope coefficients and 1D array of weights
    """
    print('running function load_CMIP5_slope_coeffs_UK')
    in_zosdir_uk = settings["cmipinfo"]["slopecoeffsuk"]
    filename_uk = f'{scenario}_CMIP5_regress_coeffs_uk_mask_1.pickle'

    try:
        with open(os.path.join(in_zosdir_uk, filename_uk), 'rb') as f:
            data = pickle.load(f, encoding='latin1')['uk_mask_1']
    except FileNotFoundError:
        raise FileNotFoundError(filename_uk,
                                '- scenario selected does not exist')

    # Keys are: 'coeffs', 'models', 'weights'
    coeffs = data['coeffs']
    weights = data['weights']

    return coeffs, weights


def read_gia_estimates(sci_method):
    """
    Read in pre-processed interpolator objects of GIA estimates (Lambeck,
    ICE5G or UK specific ones)
    :param sci_method: scientific method used to select GIA estimates, GRD
    fingerprints and CMIP regression (based on user input)
    :param coords: latitude and longitude of tide gauge
    :return: length of GIA_vals and numpy array of pre-processed interpolator
    objects of GIA estimates
    """
    print('running function read_gia_estimates')
    # Directories containing GIA data (independent of scenario)
    if sci_method == 'global':
        gia_file = settings["giaestimates"]["global"]
    elif sci_method == 'UK':
        gia_file = settings["giaestimates"]["uk"]
    else:
        raise UnboundLocalError('The selected GIA estimate - ' +
                                f'{sci_method} - is not available')

    with open(gia_file, "rb") as ifp:
        GIA_dict = pickle.load(ifp, encoding='latin1')

    GIA_vals = []
    # The GIA_dict contains interpolator objects
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


def setup_FP_interpolators(components, sci_method):
    """
    Create 2D Interpolator objects for the Slangen, Spada and Klemann
    fingerprints
    :param components: list of sea level components
    :param sci_method: scientific method used to select GIA estimates, GRD
    fingerprints and CMIP regression (based on user input)
    :return nFPs: length of FPlist and interpolator objects of all sea level
    components
    """
    print('running function setup_FP_interpolators')

    # Directories for the Slangen, Spada and Klemann fingerprints
    slangendir = settings["fingerprints"]["slangendir"]
    spadadir = settings["fingerprints"]["spadadir"]
    klemanndir = settings["fingerprints"]["klemanndir"]

    # Create empty dictionaries for the Slangen, Spada and Klemann fingerprints
    # interpolator objects.
    slangen_FPs = {}
    spada_FPs = {}
    klemann_FPs = {}

    # Only 1 fingerprint for Landwater
    comp = 'landwater'
    slangen_FPs[comp] = create_FP_interpolator(slangendir,
                                               comp + '_slangen_nomask.nc')

    # Create interpolators for the remaining components. Expansion ('exp')
    # is global so no interpolation is needed.
    components_todo = [c for c in components if c not in ['exp', 'landwater']]
    for comp in components_todo:
        slangen_FPs[comp] = create_FP_interpolator(slangendir,
                                                   comp + '_slangen_nomask.nc')
        spada_FPs[comp] = create_FP_interpolator(spadadir,
                                                 comp + '_spada_nomask.nc')
        klemann_FPs[comp] = create_FP_interpolator(klemanndir,
                                                   comp + '_klemann_nomask.nc')

    if sci_method == 'UK':
        # Klemann fingerprints were not used in UKCP18
        FPlist = [slangen_FPs, spada_FPs]
    elif sci_method == 'global':
        FPlist = [slangen_FPs, spada_FPs, klemann_FPs]
    else:
        raise UnboundLocalError('The selected GRD fingerprint method - ' +
                                f'{sci_method} - is not available')

    nFPs = len(FPlist)

    return nFPs, FPlist


def read_csv_file(file_pattern: str, start_yr: int=2007, 
                  end_yr: int=settings["projection_end_year"]):
    file = glob.glob(
        os.path.join(
            settings["emulator_settings"]["emulator_input_dir"], 
            file_pattern))
    if not file:
        raise FileNotFoundError(f'File {file_pattern} not found')
    df = pd.read_csv(file[0])
    return df.loc[:, str(start_yr):str(end_yr)].to_numpy()


def main():
    """
    Reads in and calculates global and local (regional) sea level change
    (sum total), based on the different contributing factors e.g. thermal
    expansion, GIA and mass balance. Writes out the selected emissions scenario
    estimates of the various components and their sums.
    """
    print(f'\nProjecting out to: {settings["projection_end_year"]}\n')

    # Extract site data from station list (e.g. tide gauge location) or
    # construct based on user input
    if settings["emulator_settings"]["emulator_mode"]:
        print('\nInitiating ProFSea emulator')
        if settings["projection_end_year"] > 2100:
            palmer_method = True
        else:
            palmer_method = False
            
        makefolder(os.path.join(settings["baseoutdir"], 'emulator_output'))
        
        # Get the metadata of either the site location or tide gauge location
        for scenario in settings["emulator_settings"]["emulator_scenario"]:
            # print(f'Projecting {scenario}...')
            # T_change = read_csv_file(f'*{scenario}*_temperature*.csv')
            # OHC_change = read_csv_file(f'*{scenario}*_ocean_heat_content_change*.csv')

            # cum_emissions_file = 'cumulative_scenario_emissions.json'
            # if glob.glob(os.path.join(settings["emulator_settings"]["emulator_input_dir"], cum_emissions_file)):
            #     with open('ngfs_data/cumulative_scenario_emissions.json') as f:
            #         cum_emissions_total = json.load(f)[scenario]
            # else:
            #     raise FileNotFoundError('For any non-RCP scenario, a cumulative emissions total must be provided.')
            
            # gmslr = GMSLREmulator(
            #     T_change,
            #     OHC_change,
            #     scenario,
            #     os.path.join(settings["baseoutdir"], 'emulator_output'),
            #     settings["projection_end_year"],
            #     palmer_method=palmer_method,
            #     input_ensemble=settings["emulator_settings"]["use_input_ensemble"],
            #     cum_emissions_total=cum_emissions_total)
            # gmslr.project()
            # print('Saving components...')
            # gmslr.save_components(
            #     os.path.join(settings["baseoutdir"], 'emulator_output'),
            #     scenario)
            # print('Saved!\n')

            calc_future_sea_level(scenario)
    else:
        scenarios = ['rcp26', 'rcp45', 'rcp85']
        for scenario in scenarios:
            calc_future_sea_level(scenario)


if __name__ == '__main__':
    main()
