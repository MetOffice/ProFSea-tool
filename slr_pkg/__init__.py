"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs

from config import settings
from slr_pkg import cmip, cubeplot, cubeutils, cubedata, process, whichbox
from directories import makefolder, read_dir


def abbreviate_location_name(name, chars_not_needed=' ,()'):
    """
    Returns an abbreviated version of the location name, by removing the
    specified characters.
    Check for a comma in the name of the location; then remove the part after
    the comma.
    :param name: site location
    :param chars_not_needed: characters in location name to remove
    :return: the name after removing spaces and brackets
    """
    if ',' in name:
        abbrev_name = name.split(',')[0]
    else:
        abbrev_name = name[:]

    return ''.join([c for c in abbrev_name if c not in chars_not_needed])


def extract_dyn_steric_regression(models, df, scenarios):
    """
    Calculate, plot, and save regression between the global mean (
    thermosteric) and local (sterodynamic) sea level change from CMIP models.
    :param models: list of model names to be extracted
    :param df: DataFrame of all site location's metadata
    :param scenarios: list of RCP scenarios
    """
    # Base directory for CMIP "zos" and "zostoga" data
    datadir = settings["cmipinfo"]["sealevelbasedir"]
    # Dictionary of CMIP models and experiments
    zos_dict = cmip.zos_dictionary()

    mstyle_dict = {'rcp26': 'bo',
                   'rcp45': 'co',
                   'rcp60': 'yo',
                   'rcp85': 'ro'}

    # Update as required for regression plots
    xmin = -0.05
    xmax = 0.6
    ymin = -0.05
    ymax = 0.6

    matplotlib.rcParams['font.size'] = 10

    # For indices of dataframe which will hold the slopes of the sea level
    # regressions
    iterables = [models, scenarios]
    mi = pd.MultiIndex.from_product(iterables, names=['Model', 'Scenario'])

    for loc_name in df.index.values:
        # Abbreviate the site location name
        loc_abbrev = abbreviate_location_name(loc_name)

        # Empty list used to construct the output dataframe
        result = []

        # Read in the grid indices of the ocean points
        in_cmipdir = read_dir()[0]
        df = read_ij_1x1_coord(in_cmipdir, loc_abbrev)

        for model in models:
            i = df.at[model, 'i']
            j = df.at[model, 'j']
            print(f'Calculating regression for {model} at grid box indices: '
                  f'i = {i} and j = {j}')
            plt.figure(figsize=(5, 4.5))
            plt.plot([-1, 3.0], [-1, 3.0], 'k:', linewidth=3)

            for scenario in scenarios:
                print(f'Scenario: {scenario}')
                try:
                    # dynamic sea level (zos)
                    zos_date = zos_dict[model][scenario]['driftcorr']
                    zos_file = f'{datadir}normalized_zos_Omon_{model}_' \
                               f'{scenario}_{zos_date}_driftcorr.nc'
                    zos = cubedata.read_zos_cube(zos_file)[0][:, j, i]
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

                    plotlon = zos.coord('longitude').points[0]
                    plotlat = zos.coord('latitude').points[0]

                    if plotlon > 180:
                        plotlon -= 360

                    yrs = get_cube_years(zos)

                    if model == 'bcc-csm1-1':
                        index = np.where(yrs >= 2100)
                        zostoga.data[index] = zostoga.data[index] +  \
                            zostoga.data[index[0][0] - 1]

                    # Calculate regression coefficients for periods 2005-2100
                    # and 2050-2100
                    idx1 = ((2005 <= yrs) & (yrs <= 2100))
                    idx2 = ((2050 <= yrs) & (yrs <= 2100))

                    zostoga.data = zostoga.data - zostoga.data[0:10].mean()
                    zos.data = zos.data - zos.data[0:10].mean()

                    # Calculate the slope and intercepts of linear fits of the
                    # global and local sea level projections
                    slope1, _ = np.polyfit(zostoga.data[idx1], zostoga.data[
                        idx1] + zos.data[idx1], 1)
                    slope2, _ = np.polyfit(zostoga.data[idx2], zostoga.data[
                        idx2] + zos.data[idx2], 1)

                    result.append([i, j, plotlon, plotlat, slope1, slope2])

                    # Add the data points to the figure
                    leglabel = scenario.upper()
                    plt.plot(zostoga.data[idx1],
                             zostoga.data[idx1] + zos.data[idx1],
                             mstyle_dict[scenario],
                             markersize=8, markeredgecolor='None',
                             label=leglabel)

                except IOError:
                    result.append([i, j, -99, -99, np.nan, np.nan])
                    continue

            plt.xlabel('Global thermal expansion (m)')
            plt.ylabel('Local sea level (m)')
            plt.title(model)
            plt.xlim([xmin, xmax])
            plt.ylim([ymin, ymax])
            plt.legend(loc='upper left', numpoints=1, frameon=False)
            plt.tight_layout()

            # Create the output figure file directory and filename
            out_zosfdir = read_dir()[3]
            makefolder(out_zosfdir)
            outfigfile = f'{out_zosfdir}{loc_abbrev}_{model}' + \
                '_zos_regression_2005-2100.png'

            # Save the sea level regression figures to file
            print(f'SAVING: {loc_abbrev}_{model}_zos_regression_2005-2100.png')
            plt.savefig(outfigfile, dpi=200)
            plt.close()

        # Save the regression slopes for all models and scenarios
        df_out = pd.DataFrame(result, columns=['i', 'j', 'lon', 'lat',
                                               'slope_05_00', 'slope_50_00'],
                              index=mi)
        df_out['slope_05_00'] = df_out['slope_05_00'].apply(
            lambda x: round(x, 4))
        df_out['slope_50_00'] = df_out['slope_50_00'].apply(
            lambda x: round(x, 4))

        # Create the output data file directory and filename
        out_zosddir = read_dir()[2]
        makefolder(out_zosddir)
        outdatafile = f'{out_zosddir}{loc_abbrev}_zos_regression.csv'

        # Save the sea level regressions data to file
        df_out.to_csv(outdatafile, na_rep='NA')


def get_cube_years(cube):
    """
    Extract a numpy array of years from Iris cube
    :param cube: iris cube containing local sea level (zos)
    :return: array of years in cube
    """
    time = cube.coord('time')
    dates = time.units.num2date(time.points)
    years = [date.year for date in dates]

    return np.array(years, dtype=int)


def plot_ij(cube, model, location, idx, lat, lon, save_map=True, rad=5):
    """
    Plots the location of the requested site (based on lat, lon - red
    cross), the location of the nearest  model grid box (based on model indices
    i,j - black circle) and the sea surface height above the geoid in meters
    (re-gridded onto a 1x1m grid) for each CMIP model.
    :param cube: data cube containing the zos field for the CMIP models
    :param model: CMIP model name
    :param location: site location
    :param idx: latitude and longitude of the closest ocean grid point
    :param lat: latitude of site
    :param lon: longitude of site
    :param save_map: determine is maps are saved to file, default is True
    :param rad: plotting radius, default is 5
    """
    i, j = idx

    # Define region for plotting
    minlon, maxlon = lon - rad * 1.5, lon + rad * 1.5
    minlat, maxlat = lat - rad, lat + rad

    # For sites that cross 180deg. e.g. minlon = 175, maxlon = 185
    if maxlon > 180 or minlon > 180:
        maxlon -= 360
        minlon -= 360

    region = [minlon, minlat, maxlon, maxlat]

    # plotlat, plotlon are the coordinate of the nearest ocean point
    plotlat = cube.coord('latitude').points[j]
    plotlon = cube.coord('longitude').points[i]

    # targetlat, targetlon are the coordinate of the site
    targetlat = lat
    targetlon = lon

    if plotlon > 180:
        plotlon -= 360
    if targetlon > 180:
        targetlon -= 360

    fig = plt.figure()
    ax = cubeplot.block(cube, land=False, region=region, cmin=-1, cmax=1,
                        plotcbar=True, nlevels=25, cent_lon=targetlon,
                        title='{} (1x1 grid) - SSH above geoid'.format(model))

    # Transform the points onto the projection used by the map. May not be
    # necessary, but is done to avoid possible position errors (i.e. the points
    # plotted in the wrong place).
    # Set up projections of the map, and the tide gauge and ocean points.
    MAP_CRS = ccrs.PlateCarree(central_longitude=targetlon)
    SRC_CRS = ccrs.PlateCarree()

    orig_lons = np.array([plotlon, targetlon])
    orig_lats = np.array([plotlat, targetlat])

    # Transform the points onto the same projection used by the map.
    # The function returns x, y and z-coordinates.
    # The z-coords are not used, hence the underscore.
    new_lons, new_lats, _ = MAP_CRS.transform_points(
        SRC_CRS, orig_lons, orig_lats).T

    # Plot symbols showing the ocean point and the tide gauge
    pred, = ax.plot(new_lons[0], new_lats[0], 'ok')
    ax.plot(new_lons[1], new_lats[1], 'xr')

    if save_map:
        # Create the output file directory location
        out_mapdir = read_dir()[1]
        makefolder(out_mapdir)

        # Abbreviate the site location name suitable to use as a filename
        loc_abbrev = abbreviate_location_name(location)
        figfile = os.path.join(out_mapdir,
                               f'{loc_abbrev}_{model}_ij_figure.png')

        # Save the CMIP grid box selection map to file
        plt.savefig(figfile)
        plt.close()
    else:
        selected_lat = cube.coord('latitude').points[j]
        selected_lon = cube.coord('longitude').points[i]
        
        def onclick(event):
            nonlocal selected_lat, selected_lon, pred
            nonlocal i, j
            selected_lon, selected_lat = event.xdata, event.ydata
            
            # Transform onto original projection
            MAP_CRS = ccrs.PlateCarree(central_longitude=targetlon)
            SRC_CRS = ccrs.PlateCarree()
                
            selected_lon, selected_lat = SRC_CRS.transform_point(
                selected_lon, selected_lat, MAP_CRS)

            (i, j), = whichbox.find_gridbox_indicies(cube,[(selected_lon, selected_lat)])
            
            selected_lon = cube.coord('longitude').points[i]
            selected_lat = cube.coord('latitude').points[j]
                
            MAP_CRS = ccrs.PlateCarree(central_longitude=targetlon)
            SRC_CRS = ccrs.PlateCarree()
                
            plot_lon, plot_lat = MAP_CRS.transform_point(
                selected_lon, selected_lat, SRC_CRS)
            
            pred.remove()
            pred, = ax.plot(plot_lon, plot_lat, 'ok')
            fig.canvas.draw()
            
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        
        
        # Create the output file directory location
        out_mapdir = read_dir()[1]
        makefolder(out_mapdir)

        # Abbreviate the site location name suitable to use as a filename
        loc_abbrev = abbreviate_location_name(location)
        figfile = os.path.join(out_mapdir,
                               f'{loc_abbrev}_{model}_ij_figure.png')

        # Save the CMIP grid box selection map to file
        fig.savefig(figfile)
        plt.close()
        
        return i, j, selected_lon, selected_lat


def read_ar5_component(datadir, rcp, var, value='mid'):
    """
    Return yrs and data for specified global mean sea level component loaded
    from file provided as part of supplementary material for AR5 Ch13.
    :param datadir: directory containing datafiles provided as supplementary
        material for CH13 of AR5.
    :param rcp: emissions scenario
    :param var: sea level component name e.g. 'greendyn'
    :param value: 'mid', 'upper', or 'lower', default is 'mid'
    :return: array of yrs and data
    """
    f = '%s%s_%s%s.txt' % (datadir, rcp, var, value)
    arr = np.loadtxt(f)

    yrs = arr[:, 0]
    dat = arr[:, 1]

    return yrs, dat


def read_ij_1x1_coord(datadir, location):
    """
    Read ij coordinate pair for specified model.
    :param datadir: directory in which the CMIP model grid box indices for
    each site location are stored
    :param location: site location
    :return: DataFrame of CMIP model grid box indices
    """
    filename = '{}{}_ij_1x1_coords.csv'.format(datadir, location)

    df = pd.read_csv(filename, skiprows=3, header=0, index_col='Model')

    return df


def choose_montecarlo_dir():
    """
    Choose the Monte Carlo directory based on the projection end year.
    """
    end_yr = settings["projection_end_year"]
    if (end_yr >= 2050) & (end_yr <= 2100):
        mcdir = settings["short_montecarlodir"]
    elif (end_yr > 2100) & (end_yr <= 2300):
        mcdir = settings["long_montecarlodir"]
    else:
        raise ValueError('Projection end year must be between 2050 and 2300')
    
    return mcdir


if __name__ == '__main__':
    pass
