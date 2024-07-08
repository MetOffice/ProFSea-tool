"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

from profsea.config import settings
from profsea.slr_pkg import extract_dyn_steric_regression  # found in __init.py__
from profsea.slr_pkg import models
from profsea.tide_gauge_locations import extract_site_info


def extract_cmip5_steric_dyn_regression(df):
    """
    Finds all CMIP model names, then calculates the regression parameters
    between global and local sea level projections for the given locations.
    :param df: DataFrame of site location's metadata
    """
    print('running function extract_cmip5_steric_dyn_regression')
    # Specify the emission scenarios
    scenarios = ['rcp26', 'rcp45', 'rcp85']

    # Select CMIP5 models to use
    if settings["cmipinfo"]["cmip_sea"] == 'all':
        model_names = models.cmip5_names()
    elif settings["cmipinfo"]["cmip_sea"] == 'marginal':
        model_names = models.cmip5_names_marginal()
    else:
        raise UnboundLocalError(
            'The selected CMIP5 models to use - cmip_sea = ' +
            f'{settings["cmipinfo"]["cmip_sea"]} - ' +
            'is not recognised')

    # Calculate the regression parameters and plot the results
    # Note the x and y limits of the plot are set to 0.6 - this can be
    # updated in extract_dyn_steric_regression() function
    extract_dyn_steric_regression(model_names, df, scenarios)


def main():
    """
    Calculate the regression between local sea level change and global mean
    sea level rise from thermal expansion.
    If the regression slope is steeper/shallower than 1, then the local change
    is increasing faster/slower than the global average.
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

    # Extract site data from station list (e.g. tide gauge location) or
    # construct based on user input
    df_site_data = extract_site_info(settings["tidegaugeinfo"]["source"],
                                     settings["tidegaugeinfo"]["datafq"],
                                     settings["siteinfo"]["region"],
                                     settings["siteinfo"]["sitename"],
                                     settings["siteinfo"]["sitelatlon"])

    extract_cmip5_steric_dyn_regression(df_site_data)


if __name__ == '__main__':
    main()
