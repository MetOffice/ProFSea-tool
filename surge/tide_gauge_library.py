"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import linregress

from config import settings


def calc_baseline_sl(root_dir, data_region, loc_abbrev, years, annual_means,
                     baseline_years, min_overlapping_years=14):
    """
    Calculates or estimates the baseline sea levels, using observed or
    modelled values. If there are sufficient observed values, use those to
    estimate the baseline sea level, otherwise read in and extrapolate back
    from the RCP 2.6 projections, which are nearly linear.
    :param root_dir: base directory for all input and output data
    :param data_region: region (used for file paths)
    :param loc_abbrev: abbreviated name of tide gauge
    :param years: Numpy array of available years for requested tide gauge
    :param annual_means: annual mean sea level (mm)
    :param baseline_years: first and last year of the baseline period,
    should be 1981-2000
    :param min_overlapping_years: minimum number of years of the observed
    sea levels that must lie inside the baseline period
    :return: the baseline sea level and the difference between the observed
    period and the baseline
    """
    print('running function calc_baseline_sl')
    # Check which years from this gauge lie within the baseline period.
    common_years = find_common_years(years, baseline_years)
    nvalid = 0

    if len(common_years) >= min_overlapping_years:
        obs_amsl = annual_means[(years >= common_years[0]) &
                                (years <= common_years[-1])]
        # Check for number of valid data values
        nvalid = sum(~np.isnan(obs_amsl))

    if nvalid >= min_overlapping_years:
        # Calculate the mean sea level for the baseline using these
        # observations.
        baseline_sea_level = np.nanmean(obs_amsl)
        delta_sl = 0.0
    else:
        # (a) Get obs years
        # (b) read in the RCP2.6 sea level projections - percentile = 50.0
        df_m_rcp26 = read_regional_sea_level_projections(
            root_dir, data_region, loc_abbrev, 'rcp26', pcile=50.0)
        sl_rcp26 = df_m_rcp26['sum'].values
        years_rcp26 = df_m_rcp26['year'].values

        # (c) Fit a straight line to the mid-point estimates
        m, c, _, _, _ = linregress(years_rcp26, sl_rcp26)

        # Extrapolate fitted straight line to get values for the baseline
        # years and hence the mean value for the baseline.
        model_sl_in_baseline = [m * base_year + c for base_year in
                                list(range(baseline_years[0],
                                           baseline_years[1] + 1))]
        model_mean_baseline_sl = np.mean(np.array(model_sl_in_baseline))

        # Extrapolate fitted straight line to get modelled values for the
        # years of the observations, then the mean value for this period
        model_sl_in_obs_years = [m * obs_year + c for obs_year in years]
        model_mean_obs_sl = np.mean(np.array(model_sl_in_obs_years))

        # Calculate the change in sea level between the baseline and
        # observed periods using the modelled data
        delta_sl = model_mean_baseline_sl - model_mean_obs_sl

        # (d) Add modelled SL change from (c) to mean sea level from
        # tide gauge data
        obs_mean_sl = np.nanmean(annual_means)
        baseline_sea_level = obs_mean_sl + delta_sl

    return baseline_sea_level, delta_sl


def find_common_years(years, baseline_years):
    """
    Finds the years that lie within the specified baseline period
    :param years: List of years in the observed data
    :param baseline_years: 2-element list holding the first and last years
    of the period of interest
    :return: list of all common years within the baseline period
    """
    common_years = sorted(list(set(list(range(baseline_years[0],
                                              baseline_years[1] + 1)))
                               & set(years)))
    if not common_years:
        print(f'No common years found between tide gauge data and ' +
              'baseline years')
    else:
        print(f'The common years in the tide gauge data are ' +
              f'between {common_years[0]} and {common_years[-1]}')

    return common_years


def read_regional_sea_level_projections(root_dir, data_region, loc_abbrev,
                                        scenario, pcile=None):
    """
    Reads in the regional sea level projections calculated from the CMIP
    projections in a previous step. These data are changes in sea level
    relative to a baseline of 1981-2010.
    :param root_dir: root directory containing all code and input data
    :param data_region: region (used for file paths)
    :param loc_abbrev: abbreviated name of tide gauge
    :param scenario: RCP scenario
    :param pcile: If not None, return data for this percentile; otherwise,
    return data for all percentiles
    :return df: DataFrame containing regional sea level projections,
    as annual means.
    """
    path = os.path.join(root_dir, data_region, 'data', 'sea_level_projections')
    filename = os.path.join(
        path, f'{loc_abbrev}_{scenario}_projection_2100_v6_regional.csv')
    df = pd.read_csv(filename, header=0, usecols=['year', 'percentile', 'sum'])

    if pcile is not None:
        df = df.loc[df['percentile'] == pcile]
        df.reset_index(inplace=True, drop=True)

    # Convert the projections from metres to mm.
    df['sum'] = df['sum'] * 1000.0

    return df


def read_rlr_annual_mean_sea_level(station_id):
    """
    Reads in annual mean sea levels from a .rlrdata file - downloaded
    from PSMSL (https://www.psmsl.org/data/obtaining/complete.php and
    https://www.psmsl.org/data/obtaining/notes.php). Replaces all -99999 with
    NAN values - for use in plotting
    :param station_id: PSMSL Station ID for location
    :return years: numpy array of available years for requested station
    :return annual_means: corresponding annual mean sea level (mm)
    :return flag_missing: corresponding flag for missing data
    :return flag_exception: corresponding 'flag for attention'
    """
    print('running function read_rlr_annual_mean_sea_level')
    psmsl_basedir = settings["tidegaugeinfo"]["psmsldir"]
    path = os.path.join(psmsl_basedir, 'rlr_annual', 'data')

    filename = os.path.join(path, f'{station_id}.rlrdata')
    print(f'Reading in data for {filename}')
    rlr_data = np.genfromtxt(filename, dtype=None, skip_header=0, names=[
        'Year', 'RLR_data', 'Flag_missing', 'Flag_exception'],
                             delimiter=";", encoding=None)
    years = []
    annual_means = []
    flag_missing = []
    flag_exception = []
    for row in rlr_data:
        year = row[0]
        rlr = row[1]
        missing = row[2]
        exception = row[3]
        years = np.append(years, int(year))
        if float(rlr) != -99999:
            annual_means = np.append(annual_means, float(rlr))
        else:
            annual_means = np.append(annual_means, np.nan)

        flag_missing = np.append(flag_missing, str(missing))
        flag_exception = np.append(flag_exception, float(exception))

    return years, annual_means, flag_missing, flag_exception
