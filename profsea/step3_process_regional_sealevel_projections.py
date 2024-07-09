"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import os
import iris
import glob
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from netCDF4 import Dataset

from profsea.config import settings
from profsea.tide_gauge_locations import extract_site_info
from profsea.slr_pkg import abbreviate_location_name, choose_montecarlo_dir  # found in __init.py__
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


def calc_future_sea_level_at_site(df, site_loc, scenario):
    """
    Calculates future sea level at the given site and write to file.
    :param df: Data frame of all metadata (tide gauge or site specific) for
               each(all) location(s)
    :param site_loc: name of the site location
    :param scenario: emission scenario
    """
    print('running function calc_future_sea_level_at_site')
    # Get the location's latitude and longitude.
    loc_coords = [df.at[site_loc, 'Latitude'], df.at[site_loc, 'Longitude']]

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
    nesm = sample.shape[1]
    nyrs = sample.shape[0]
    yrs = np.arange(2007, 2007 + nyrs)

    # Determine the number of samples you wish to make CHOGOM
    # project = 100000, UKCP18 = 200000
    if nesm >= 200000:
        nsmps = 200000
    else:
        nsmps = nesm
        
    array_dims = [nesm, nsmps, nyrs]

    # Get random samples of global and regional sea level components
    montecarlo_G, montecarlo_R = calculate_sl_components(mcdir, components,
                                                         scenario, site_loc,
                                                         loc_coords, yrs,
                                                         array_dims)

    # Calculate the summary time series of global and local sea level
    # change, i.e. mean and percentile values.
    G_df, R_df = calculate_summary_timeseries(components, yrs, montecarlo_G,
                                              montecarlo_R)

    # Create the output sea level projections file directory and filename
    sealev_ddir = read_dir()[4]
    loc_abbrev = abbreviate_location_name(site_loc)
    file_header = '_'.join([loc_abbrev, scenario, "projection", 
                            f"{settings['projection_end_year']}"])
    G_file = '_'.join([file_header, 'global']) + '.csv'
    R_file = '_'.join([file_header, 'regional']) + '.csv'

    # Save the global and local projections
    makefolder(sealev_ddir)
    G_df.to_csv(os.path.join(sealev_ddir, G_file))
    R_df.to_csv(os.path.join(sealev_ddir, R_file))


def calc_gia_contribution(sci_method, yrs, nyrs, nsmps, coords):
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

    nGIA, GIA_vals = read_gia_estimates(sci_method, coords)

    Tdelta, G_offset = calc_baseline_period(sci_method, yrs)

    # Unit series of mm/yr expressed as m/yr
    unit_series = (np.arange(nyrs) + Tdelta) * 0.001
    GIA_unit_series = np.ones([nsmps, nyrs]) * unit_series

    # rgiai is an array of random GIA indices the size of the sample years
    rgiai = np.random.randint(nGIA, size=nsmps)

    GIA_series = GIA_unit_series.transpose() * GIA_vals[rgiai]

    print('GIA_vals (mm/yr) = ', GIA_series)

    return GIA_series, G_offset


def calculate_sl_components(mcdir, components, scenario, site_loc, loc_coords,
                            yrs, array_dims):
    """
    Calculates global and regional component part contributions to sea level
    change.
    :param mcdir: location of Monte Carlo time series for new projections
    :param components: sea level components
    :param scenario: emission scenario
    :param site_loc: name of the site location
    :param loc_coords: latitude and longitude of site
    :param yrs: years of the projections
    :param array_dims: Array of nesm, nsmps and nyrs
        nesm --> Number of ensemble members in time series
        nsmps --> Determine the number of samples you wish to make
        nyrs --> Number of years in each projection time series
    :return: montecarlo_G (global contribution to sea level rise) and
    montecarlo_R (regional contribution to sea level change)
    """
    print('running function calculate_sl_components')
    ncomps = len(components)

    # Numbers of ensemble members, samples, years
    nesm, nsmps, nyrs = array_dims

    # Coordinates of location of interest
    tlat, tlon = loc_coords

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

    # Monte Carlo for regional values (FPs applied) + GIA
    montecarlo_R = np.zeros((ncomps + 1, nyrs, nsmps))
    # Monte Carlo for Global values (no FPs applied)
    montecarlo_G = np.zeros((ncomps, nyrs, nsmps))

    # Calculate the GIA contribution to the regional component of sea level
    # change
    GIA_series, G_offset = calc_gia_contribution(sci_method, yrs, nyrs, nsmps,
                                                 loc_coords)
    montecarlo_R[-1, :, :] = GIA_series[:, :]

    for cc, comp in enumerate(components):
        print(f'cc = {cc:d}, comp = {comp}')

        offset = G_offset * offset_slopes[comp]

        if settings["emulator_settings"]["emulator_mode"]:
            # Input timeseries provided as numpy objects
            mc_timeseries = np.load(os.path.join(mcdir, f'{scenario}_{comp}.npy'))
            montecarlo_G[cc, :, :] = mc_timeseries[:nyrs, resamples] + offset
        else:
            cube = iris.load_cube(os.path.join(mcdir, f'{scenario}_{comp}.nc'))
            montecarlo_G[cc, :, :] = cube.data[:nyrs, resamples] + offset

        if comp == 'exp':
            if sci_method == 'global':
                if settings["emulator_settings"]["emulator_mode"]:
                    coeffs = load_CMIP5_slope_coeffs(site_loc, 'rcp85')
                else:
                    coeffs = load_CMIP5_slope_coeffs(site_loc, scenario)
                rand_coeffs = np.random.choice(coeffs, size=nsmps,
                                               replace=True)
            elif sci_method == 'UK':
                if settings["emulator_settings"]["emulator_mode"]:
                    coeffs, weights = load_CMIP5_slope_coeffs_UK('rcp85')
                else:
                    coeffs, weights = load_CMIP5_slope_coeffs_UK(scenario)
                rand_coeffs = np.random.choice(coeffs, size=nsmps,
                                               replace=True, p=weights)
            montecarlo_R[cc, :, :] = montecarlo_G[cc, :, :] * rand_coeffs
        elif comp == 'landwater':
            landwater_FP_interpolator = FPlist[0]['landwater']
            # Interpolate values to target lat/lon
            val = landwater_FP_interpolator([tlat, tlon])[0]
            montecarlo_R[cc, :, :] = montecarlo_G[cc, :, :] * val
        else:
            # Initiate an empty list for fingerprint values
            FPvals = []
            for FP_dict in FPlist:
                # Interpolate values to target lat/lon
                val = FP_dict[comp]([tlat, tlon])[0]
                FPvals.append(val)
            FPvals = np.array(FPvals)
            montecarlo_R[cc, :, :] = montecarlo_G[cc, :, :] * FPvals[rfpi]

    return montecarlo_G, montecarlo_R


def calculate_summary_timeseries(components, years, montecarlo_G,
                                 montecarlo_R):
    """
    Calculates summary timeseries of each sea level rise component, plus sums
    of the contributions from Greenland and Antarctica, and sum of all
    components.
    :param components: sea level components
    :param years: years of the projections
    :param montecarlo_G: global sea level components
    :param montecarlo_R: regional sea level components
    :return: DataFrame of global and regional summary timeseries of sea level
    """
    print('running function calculate_summary_timeseries')
    percentiles = [5, 10, 30, 33, 50, 67, 70, 90, 95]

    # Define lists to store the summary timeseries. They will be converted to
    # Pandas dataframes.
    R_list = []
    G_list = []

    for cc, _ in enumerate(components):
        # Calculate and store the global component time series
        cgout = np.percentile(montecarlo_G[cc, :, :], percentiles, axis=1)
        G_list.append(cgout.flatten(order='F'))

        # Calculate and store the regional component time series
        crout = np.percentile(montecarlo_R[cc, :, :], percentiles, axis=1)
        R_list.append(crout.flatten(order='F'))

    # Set up a multi-level index for the dataframes, consisting of the years
    # and percentiles.
    iterables = [years, percentiles]
    idx = pd.MultiIndex.from_product(iterables, names=['year', 'percentile'])
    G_df = pd.DataFrame(np.asarray(G_list).T, columns=components, index=idx)
    R_df = pd.DataFrame(np.asarray(R_list).T, columns=components, index=idx)
    R_df.rename(columns={"exp": "ocean"})

    # Store the regional GIA component time series
    ncomp = len(components)
    crout = np.percentile(montecarlo_R[ncomp, :, :], percentiles, axis=1)
    R_df['GIA'] = crout.flatten(order='F')

    # Add sums of contributions from Antarctica and Greenland to the saved
    # series. This needs to be calculated across the full montecarlo range not
    # a pandas addition.
    # calc_future_sea_level_at_site states: components = ['exp', 'antdyn',
    # 'antsmb', 'greendyn', 'greensmb', 'glacier', 'landwater']
    antnet_tmpg = np.percentile(
        montecarlo_G[1, :, :] + montecarlo_G[2, :, :],
        percentiles, axis=1).flatten(order='F')
    G_df['antnet'] = pd.DataFrame(antnet_tmpg, columns=['antnet'], index=idx)
    greennet_tmpg = np.percentile(
        montecarlo_G[3, :, :] + montecarlo_G[4, :, :],
        percentiles, axis=1).flatten(order='F')
    G_df['greennet'] = pd.DataFrame(
        greennet_tmpg, columns=['greennet'], index=idx)
    antnet_tmpr = np.percentile(
        montecarlo_R[1, :, :] + montecarlo_R[2, :, :],
        percentiles, axis=1).flatten(order='F')
    R_df['antnet'] = pd.DataFrame(antnet_tmpr, columns=['antnet'], index=idx)
    greennet_tmpr = np.percentile(
        montecarlo_R[3, :, :] + montecarlo_R[4, :, :],
        percentiles, axis=1).flatten(order='F')
    R_df['greennet'] = pd.DataFrame(
        greennet_tmpr, columns=['greennet'], index=idx)

    # Sum of all components (net sea level change)
    montecarlo_Gsum = np.sum(montecarlo_G, axis=0)
    montecarlo_Rsum = np.sum(montecarlo_R, axis=0)
    # Store the percentiles of the sums of all components to global and local
    # sea level change
    cgout = np.percentile(montecarlo_Gsum, percentiles, axis=1)
    crout = np.percentile(montecarlo_Rsum, percentiles, axis=1)
    G_df['sum'] = cgout.flatten(order='F')
    R_df['sum'] = crout.flatten(order='F')

    return G_df, R_df


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


def load_CMIP5_slope_coeffs(site_loc, scenario):
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
    loc_abbrev = abbreviate_location_name(site_loc)
    filename = os.path.join(in_zosddir, f'{loc_abbrev}_zos_regression.csv')
    df = pd.read_csv(filename, header=0)

    coeffs = df.loc[(df['Scenario'] == scenario)]['slope_05_00'].values

    # Some models are missing regression slopes for RCP2.6. If so, use RCP4.5
    # values instead
    if scenario == 'rcp26':
        rcp45_coeffs = df.loc[(df['Scenario'] ==
                               'rcp45')]['slope_05_00'].values
        msgi = np.where(np.isnan(coeffs))[0]
        coeffs[msgi] = rcp45_coeffs[msgi]

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


def read_gia_estimates(sci_method, coords):
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
    lat, lon = coords
    # The GIA_dict contains interpolator objects
    for key in list(GIA_dict.keys()):
        val = GIA_dict[key]([lat, lon])[0]
        GIA_vals.append(val)

    nGIA = len(GIA_vals)
    GIA_vals = np.array(GIA_vals)

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
    print(f'User specified site name(s) is (are):'
          f' {settings["siteinfo"]["sitename"]}')
    print(f'User specified lat(s) and lon(s) is (are): '
          f'{settings["siteinfo"]["sitelatlon"]}')
    if settings["siteinfo"]["sitelatlon"] == [[]]:
        print(f'    No lat lon specified - use tide gauge metadata if '
              f'available')
    print(f'User specified science method is: {settings["sciencemethod"]}')
    if {settings["cmipinfo"]["cmip_sea"]} == {'all'}:
        print('User specified all CMIP models')
    elif {settings["cmipinfo"]["cmip_sea"]} == {'marginal'}:
        print('User specified CMIP models for marginal seas only')

    print(f'\nProjecting out to: {settings["projection_end_year"]}\n')

    # Extract site data from station list (e.g. tide gauge location) or
    # construct based on user input
    df_site_data = extract_site_info(settings["tidegaugeinfo"]["source"],
                                     settings["tidegaugeinfo"]["datafq"],
                                     settings["siteinfo"]["region"],
                                     settings["siteinfo"]["sitename"],
                                     settings["siteinfo"]["sitelatlon"])

    if settings["emulator_settings"]["emulator_mode"]:
        print('\nInitiating ProFSea emulator')
        if settings["projection_end_year"] > 2100:
            palmer_method = True
        else:
            palmer_method = False
            
        makefolder(os.path.join(settings["baseoutdir"], 'emulator_output'))
        
        # Get the metadata of either the site location or tide gauge location
        for scenario in settings["emulator_settings"]["emulator_scenario"]:
            print(f'Projecting {scenario}...')
            T_change = read_csv_file(f'*{scenario}*_temperature*.csv')
            OHC_change = read_csv_file(f'*{scenario}*_ocean_heat_content_change*.csv')
            
            gmslr = GMSLREmulator(
                T_change,
                OHC_change,
                scenario,
                os.path.join(settings["baseoutdir"], 'emulator_output'),
                settings["projection_end_year"],
                palmer_method=palmer_method,
                input_ensemble=settings["emulator_settings"]["use_input_ensemble"]
            )
            gmslr.project()
            for loc_name in df_site_data.index.values:
                calc_future_sea_level_at_site(df_site_data, loc_name, scenario)
    else:
        scenarios = ['rcp26', 'rcp45', 'rcp85']
        for scenario in scenarios:
            for loc_name in df_site_data.index.values:
                calc_future_sea_level_at_site(df_site_data, loc_name, scenario)


if __name__ == '__main__':
    main()
