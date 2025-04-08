"""
Copyright (c) 2023, Met Office
All rights reserved.
"""
from profsea.config import settings
from profsea.directories import read_dir, makefolder
from profsea.slr_pkg import models, cmip, cubedata, process, get_cube_years

import numpy as np
from sklearn.linear_model import LinearRegression


def extract_dyn_steric_regression(models, scenarios):
    """
    Calculate, plot, and save regression between the global mean (
    thermosteric) and local (sterodynamic) sea level change from CMIP models.
    :param models: list of model names to be extracted
    :param scenarios: list of RCP scenarios
    """
    # Base directory for CMIP "zos" and "zostoga" data
    datadir = settings["cmipinfo"]["sealevelbasedir"]
    # Dictionary of CMIP models and experiments
    zos_dict = cmip.zos_dictionary()
    zos_masks = np.empty((len(models), len(scenarios), 180, 360)) # (model, scenario, lat, lon)
    regr_slopes = np.empty((len(models), len(scenarios), 180, 360)) # (model, scenario, lat, lon)
    for m, model in enumerate(models):
        print(f'Calculating regression for {model} all over the grid')
        for s, scenario in enumerate(scenarios):
            try:
                print(f'Scenario: {scenario}')
                # dynamic sea level (zos)
                zos_date = zos_dict[model][scenario]['driftcorr']
                zos_file = f'{datadir}normalized_zos_Omon_{model}_' \
                            f'{scenario}_{zos_date}_driftcorr.nc'
                zos = cubedata.read_zos_cube(zos_file)[0]
                zos_masks[m, s] = zos[0].data.mask

                # --------------------------------------------------------
                # global mean thermosteric (zostoga)
                # Extract, and drift-correct CMIP "zostoga" data
                # Normal (concatenated)
                zostoga_date = zos_dict[model][scenario]['zostoga']
                zostoga_file = f'{datadir}zostoga_Omon_{model}_' \
                                f'{scenario}_{zostoga_date}.nc'
                zostoga_raw = cubedata.read_zos_cube(zostoga_file)[0]
                # piControl (concatenated)
                piControl_date = zos_dict[model][scenario]['piControl']
                zostoga_pic_file = f'{datadir}zostoga_Omon_' \
                                    f'{model}_piControl_{piControl_date}.nc'
                zostoga_pic = cubedata.read_zos_cube(zostoga_pic_file)[0]

                regr = process.Regress('linear')
                cube_drift, _ = regr.regress_t_scalar(zostoga_pic)
                zostoga, _ = regr.detrend_scalar(zostoga_raw, cube_drift)

                # --------------------------------------------------------

                yrs = get_cube_years(zos)
                if model == 'bcc-csm1-1':
                    index = np.where(yrs >= 2100)
                    zostoga.data[index] = zostoga.data[index] +  \
                        zostoga.data[index[0][0] - 1]
                # Calculate regression coefficients for periods 2005-2100
                # and 2050-2100
                idx = ((2005 <= yrs) & (yrs <= 2100))

                zostoga.data = zostoga.data - zostoga.data[0:10].mean()
                zos.data = zos.data - zos.data[0:10].mean()

                # Calculate the slope and intercepts of linear fits of the
                # global and local sea level projections
                regr = LinearRegression()
                regr.fit(
                    zostoga.data[idx].reshape(-1, 1),
                    zos.data[idx].reshape(zos.data[idx].shape[0], -1))
                slopes = regr.coef_.reshape(180, 360)

                regr_slopes[m, s] = slopes
            except Exception: 
                print(f'Error calculating regression for {model}, {scenario}')
                print('Adding NaN to regression slope')

                regr_slopes[m, s] = np.nan
                zos_masks[m, s] = np.nan
                continue

    # Replace rcp26 regression slopes with all np.nan with rcp45
    for m in range(len(models)):
        if regr_slopes[m, 0].all() == np.nan:
            regr_slopes[m, 0] = regr_slopes[m, 1]
            zos_masks[m, 0] = zos_masks[m, 1]
        
    assert regr_slopes.all() != np.nan, \
        'Regression slopes contain NaN values. Check the data and try again.'

    # Create the output data file directory and filename
    out_zosddir = read_dir()[2]
    makefolder(out_zosddir)
    outdatafile = f'{out_zosddir}zos_regression.npy'
    outdatafile_mask = f'{out_zosddir}zos_regression_masks.npy'

    # Save the sea level regressions data to file
    np.save(outdatafile, regr_slopes)
    np.save(outdatafile_mask, zos_masks)
    print(f'Saved regression data to {outdatafile}')

    # Plot the regression results



def main():
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
    extract_dyn_steric_regression(model_names, scenarios)


if __name__ == '__main__':
    main()
