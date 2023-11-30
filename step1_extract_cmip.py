"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import os
import numpy as np
import pandas as pd

from config import settings
from directories import read_dir, makefolder
from slr_pkg import abbreviate_location_name, plot_ij  # found in __init.py__
from slr_pkg import cmip, cubeutils, models, whichbox
from tide_gauge_locations import extract_site_info


def accept_reject_cmip(cube, model, site_loc, cmip_i, cmip_j, site_lat,
                       site_lon, unit_test=False):
    """
    Accept or reject selected CMIP grid box based on a user input.
    If CMIP grid box is rejected, search neighbouring grid boxes until a
    suitable one is found.
    :param cube: cube containing zos field from CMIP models
    :param model: CMIP model name
    :param site_loc: name of the site location
    :param cmip_i: CMIP coord of site location's latitude
    :param cmip_j: CMIP coord of site location's longitude
    :param site_lat: latitude of the site location
    :param site_lon: longitude of the site location
    :param unit_test: flag to disable plotting for unit testing purposes
    :return: Selected CMIP coords or None if user doesn't confirm grid box
    selection
    """
    # Plot a map of the site location and selected CMIP grid box for user
    # validation. At this stage don't save the figure to file.
    if not unit_test:
        plot_ij(cube, model, site_loc, [cmip_i, cmip_j], site_lat, site_lon,
                save_map=False)

    # Ask user to accept cmip grid box or re-select from neighbouring cells
    decision = input(f'Is selected CMIP grid box {cmip_i, cmip_j} '
                     f'appropriate for {model}? Y or N: ')
    if decision == 'Y' or decision == 'y':
        # Save map to file and return CMIP grid box coords
        if not unit_test:
            plot_ij(cube, model, site_loc, [cmip_i, cmip_j],
                    site_lat, site_lon)
        return cmip_i, cmip_j
    elif decision == 'N' or decision == 'n':
        print('Selecting another CMIP grid box')
        return None, None
    else:
        raise TypeError('Response needs to be Y or N')


def calc_radius_range(radius):
    """
    Calculate the maximum distance to search for ocean grid point.
    :param radius: Maximum range to search for ocean point
    :return: x_radius_range, y_radius_range
    """
    # if radius > 1:
    #     rm1 = radius - 1
    # else:
    #     rm1 = radius

    x_radius_range = list(range(-radius, radius + 1))
    y_radius_range = list(range(-radius, radius + 1))

    return x_radius_range, y_radius_range


def check_cube_mask(cube):
    """
    Check if the cube has a scalar mask. If so, then re-mask the cube to
    remove grid points that are exactly equal to 0.
    **NOTES:
    - Land points have a value of 0, ocean points have non-zero (both
    positive and negative) values.
    - Because the CMIP models were interpolated onto a 1x1 grid before
    the mask was checked some interpolation issues may still arise.
    :param cube: cube containing zos field from CMIP models
    :return: original cube, or re-masked cube
    """
    apply_mask = False
    try:
        cube.data.mask
        apply_mask = not isinstance(cube.data.mask, (list, tuple, np.ndarray))
    except AttributeError:
        apply_mask = True

    if apply_mask:
        new_mask = (cube.data == 0.0)
        print(f'Scalar mask: Re-masking the cube to mask cells = 0')
        cube = cube.copy(data=np.ma.array(cube.data, mask=new_mask))

    return cube


def extract_lat_lon(df):
    """
    Extracts site location's metadata, calls wraplongitude() to convert
    longitude.
    :param df: DataFrame of site location's metadata
    :return: latitude, longitude, converted longitude
    """
    site = df.name
    station_id = df['Station ID']
    lat = df['Latitude']
    lon_orig = df['Longitude']
    print(f'Site location: {site}, ID: {station_id}, ' +
          f'Lat: {lat:.2f}, Lon: {lon_orig:.2f}')

    lon = whichbox.wraplongitude(lon_orig)

    return lat, lon_orig, lon


def extract_ssh_data(cmip_sea):
    """
    Get a list of appropriate CMIP models to use. Load SSH data for each
    model on a re-gridded 1x1 grid.
    :param cmip_sea: variable to distinguish between which CMIP models to use
    :return: CMIP model names, and associated SSH data cubes
    """
    cmip_dir = settings["cmipinfo"]["sealevelbasedir"]
    # Select CMIP models to use depending on whether location is within a
    # marginal sea
    if cmip_sea == 'all':
        model_names = models.cmip5_names()
    elif cmip_sea == 'marginal':
        model_names = models.cmip5_names_marginal()
    else:
        raise UnboundLocalError(f'The selected CMIP5 models to use - ' +
                                f'cmip_sea = {cmip_sea} - is not recognised')

    # Load SSH data for each model on a re-gridded 1x1 grid, store in a list
    cubes = []
    cmip_dict = cmip.model_dictionary()
    for model in model_names:
        print(f'Getting data for {model} model')
        cmip_date = cmip_dict[model]['historical']
        cmip_file = f'{cmip_dir}zos_Omon_{model}_historical_{cmip_date}.nc'
        cube = cubeutils.loadcube(cmip_file, ncvar='zos')[0]
        cubes.append(cube.slices(['latitude', 'longitude']).next())

    return model_names, cubes


def find_ocean_pt(zos_cube_in, model, site_loc, site_lat, site_lon):
    """
    Searches for the nearest appropriate ocean point(s) in the CMIP model
    adjacent to the site location. Initially, finds the model grid box
    indices of the given location. Then, searches surrounding boxes until an
    appropriate ocean point is found - needs to be accepted by the user.
    **NOTES:
    - The GCM data have been interpolated to a common 1 x 1 degree grid
    :param zos_cube_in: cube containing zos field from CMIP models
    :param model: CMIP model name
    :param site_loc: name of the site location
    :param site_lat: latitude of the site location
    :param site_lon: longitude of the site location
    :return: model grid box indices
    """
    # Find grid box indices of location
    (i, j), = whichbox.find_gridbox_indicies(zos_cube_in,
                                             [(site_lon, site_lat)])
    grid_lons = zos_cube_in.coord('longitude').points
    grid_lats = zos_cube_in.coord('latitude').points

    # Check to see if the cube has a scalar mask, and add mask where cmip
    zos_cube = check_cube_mask(zos_cube_in)

    # If the CMIP grid box of the exact site location is an ocean point
    # Get the user to check if it's appropriate and if so return the indices
    if not zos_cube.data.mask[j, i]:
        print('Checking CMIP grid box at site location')
        i_out, j_out = accept_reject_cmip(zos_cube, model, site_loc, i, j,
                                          site_lat, site_lon)
        if i_out is not None:
            pt_lon = grid_lons[i_out]
            pt_lat = grid_lats[j_out]
            return i_out, j_out, pt_lon, pt_lat

        # If no indices are returned then the CMIP grid box is not appropriate
        # for use. Check the CMIP grid boxes surrounding the site location
        # until an appropriate one is found.
        else:
            i_out, j_out, pt_lon, pt_lat = search_for_next_cmip(i, j,
                                                                zos_cube,
                                                                model,
                                                                site_loc,
                                                                site_lat,
                                                                site_lon)

    # If the CMIP grid box of the exact site location is masked, start by
    # checking the next set of cmip grid boxes
    else:
        i_out, j_out, pt_lon, pt_lat = search_for_next_cmip(i, j, zos_cube,
                                                            model, site_loc,
                                                            site_lat, site_lon)

    return i_out, j_out, pt_lon, pt_lat


def ocean_point_wrapper(df, model_names, cubes):
    """
    Wrapper script to extract relevant metadata for site location, needed to
    select the nearest CMIP grid box. Collates the CMIP model name, i and j
    coords selected and lat and lon value of the site.
    Writes the results to a single .csv file assuming write=True.
    :param df: DataFrame of site location's metadata
    :param model_names: CMIP model names
    :param cubes: cube containing zos field from CMIP models
    """

    # Get the metadata of either the site location or tide gauge location
    for site_loc in df.index.values:
        df_site = df.loc[site_loc]
        lat, lon_orig, lon = extract_lat_lon(df_site)

        # Setup empty 2D list to store results for each model
        # [name, i and j coords, lat and lon value]
        result = []
        for n, zos_cube in enumerate(cubes):
            model = model_names[n]
            i, j, pt_lon, pt_lat = find_ocean_pt(zos_cube, model, site_loc,
                                                 lat, lon)
            result.append([model, i, j, pt_lon, pt_lat])

        # Write the data to a file
        write_i_j(site_loc, result, lat, lon_orig)


def search_for_next_cmip(cmip_i, cmip_j, cube, model, site_loc, site_lat,
                         site_lon, unit_test=False):
    """
    Iteratively check the CMIP grid boxes surrounding the site location
    until a suitable option is found.
    :param cmip_i: CMIP coord of site location's latitude
    :param cmip_j: CMIP coord of site location's longitude
    :param cube: cube containing zos field from CMIP models
    :param model: CMIP model name
    :param site_loc: name of the site location
    :param site_lat: latitude of the site location
    :param site_lon: longitude of the site location
    :param unit_test: flag to disable plotting for unit testing purposes
    :return: Selected CMIP coords
    """
    grid_lons = cube.coord('longitude').points
    grid_lats = cube.coord('latitude').points

    # The radius limit of 7 is arbitrary but should be large enough.
    for radius in range(1, 8):  # grid boxes
        print(f'Checking CMIP grid boxes {radius} box removed ' +
              'from site location')
        x_radius_range, y_radius_range = calc_radius_range(radius)
        for ix in x_radius_range:
            for iy in y_radius_range:
                # Search the nearest grid cells.  If the new mask is False,
                # that grid cell is an ocean point
                limit_lo = radius * radius
                dd = ix * ix + iy * iy
                if dd >= limit_lo:
                    # modulus for when grid cell is close to 0deg.
                    i_try = (cmip_i + ix) % len(grid_lons)
                    j_try = cmip_j + iy

                    if not cube.data.mask[j_try, i_try]:
                        i_out, j_out = accept_reject_cmip(
                            cube, model, site_loc, i_try, j_try, site_lat,
                            site_lon, unit_test)
                        if i_out is not None:
                            pt_lon = grid_lons[i_out]
                            pt_lat = grid_lats[j_out]
                            return i_out, j_out, pt_lon, pt_lat


def write_i_j(site_loc, result, site_lat, lon_orig):
    """
    Convert the grid indices to a data frame and writes to file.
    :param site_loc: name of site location
    :param result: grid indices and coordinates of nearest ocean point
    :param site_lat: latitude of the site location
    :param lon_orig: longitude of site
    """
    # Store the grid indices and coordinates in a dataframe
    df_out = pd.DataFrame(result,
                          columns=['Model', 'i', 'j', 'box_lon', 'box_lat'])

    # Create the output file directory location
    out_cmipdir = read_dir()[0]
    makefolder(out_cmipdir)

    # Abbreviate the site location name suitable to use as a filename
    loc_abbrev = abbreviate_location_name(site_loc)
    outfile = os.path.join(out_cmipdir, f'{loc_abbrev}_ij_1x1_coords.csv')

    # Save the CMIP grid box selection (i and j coordinates) to file
    with open(outfile, 'w') as ofp:
        ofp.write(f'Location: {site_loc}' + '\n')
        ofp.write(f'Latitude: {site_lat:8.3f}' + '\n')
        ofp.write(f'Longitude: {lon_orig:8.3f}' + '\n')
        df_out.to_csv(ofp, index=False)


def main():
    """
    Find the nearest CMIP model grid box, to the user defined site
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

    # Find the nearest, appropriate ocean point in CMIP models to specified
    # site location
    cmip_models, ssh_cubes = extract_ssh_data(settings["cmipinfo"]["cmip_sea"])
    ocean_point_wrapper(df_site_data, cmip_models, ssh_cubes)


if __name__ == '__main__':
    main()
