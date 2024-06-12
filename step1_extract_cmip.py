"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import os
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

from config import settings
from directories import read_dir, makefolder
from slr_pkg import abbreviate_location_name, plot_ij  # found in __init.py__
from slr_pkg import cmip, cubeutils, models, whichbox
from tide_gauge_locations import extract_site_info
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")


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

    # Check to see if the cube has a scalar mask, and add mask where cmip
    zos_cube = check_cube_mask(zos_cube_in)
    
    best_i, best_j, best_lat, best_lon = find_best_gridcell(
        zos_cube, site_lat, site_lon)
    
    if settings["auto_site_selection"]:
        plot_ij(zos_cube, model, site_loc, [best_i, best_j], 
                site_lat, site_lon, save_map=True)
        return best_i, best_j, best_lon, best_lat
    
    best_i, best_j, best_lon, best_lat = plot_ij(
        zos_cube, model, site_loc, [best_i, best_j], 
        site_lat, site_lon, save_map=False)
        
    return best_i, best_j, best_lon, best_lat


def find_best_gridcell(
        cube, target_lat, target_lon, 
        max_distance=2, distance_weight=2, 
        difference_weight=0.0005):
    """
    Find the best grid cell in the CMIP model for the target latitude and 
    longitude. The best grid cell is the one that minimizes the weighted 
    score, which is computed as the distance to the target point minus a 
    grid cell difference parameter.
    :param cube: iris.cube.Cube containing zos field from CMIP models
    :param target_lat: Latitude of the target point
    :param target_lon: Longitude of the target point
    :param max_distance: Maximum distance to search for the best grid cell
    :param distance_weight: Weight for the distance parameter
    :param difference_weight: Weight for the difference parameter
    :return: Best grid cell indices and coordinates
    """
    lon_grid = cube.coord('longitude').points
    lat_grid = cube.coord('latitude').points
    data_grid = cube.data
    
    # Get lat/lons onto grids, flatten and get mask
    lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid, indexing='ij')
    points = np.vstack([lat_mesh.ravel(), lon_mesh.ravel()]).T
    masked_points = data_grid.mask.ravel()
    
    tree = cKDTree(points[~masked_points])
    dists, indices = tree.query([target_lat, target_lon], k=49) # 7x7 grid
    
    best_i = None
    best_j = None
    best_lon = None
    best_lat = None
    min_weighted_score = float('inf')
    
    def compute_weighted_score(lat_idx, lon_idx, dist):
        value = data_grid[lat_idx, lon_idx]
        
        # Pull out surrounding grid cell indices and values
        surrounding_points = tree.query_ball_point(
            [lat_mesh[lat_idx, lon_idx], lon_mesh[lat_idx, lon_idx]], 1)
        surrounding_indices = np.where(~masked_points)[0][surrounding_points]
        surrounding_lat_idx, surrounding_lon_idx = np.unravel_index(
            surrounding_indices, data_grid.shape)
        surrounding_values = data_grid[surrounding_lat_idx, 
                                       surrounding_lon_idx]
        
        # Compute difference parameter
        avg_surrounding_diffs = np.mean(np.abs(np.diff(surrounding_values)))
        difference = np.mean(np.abs(surrounding_values - value))
        diff_param = abs(avg_surrounding_diffs - difference)

        # Compute weighted score
        weighted_score = float((dist / distance_weight ) - 
                               (difference_weight / diff_param))
        return weighted_score
    
    def check_and_update_best(lat_idx, lon_idx, dist):
        nonlocal best_i, best_j, best_lat, best_lon
        nonlocal compute_weighted_score, min_weighted_score
        
        weighted_score = compute_weighted_score(lat_idx, lon_idx, dist)
        
        if weighted_score < min_weighted_score:
            min_weighted_score = weighted_score
            best_lat = lat_mesh[lat_idx, lon_idx]
            best_lon = lon_mesh[lat_idx, lon_idx]
            best_i = nearest_lon_idx
            best_j = nearest_lat_idx

    candidate_points = sorted(zip(dists, indices), key=lambda x: x[0])  
    for dist, idx in candidate_points:
        if dist > max_distance:
            break
        
        flat_idx = np.where(~masked_points)[0][idx]
        nearest_lat_idx, nearest_lon_idx = np.unravel_index(
            flat_idx, data_grid.shape)
        
        check_and_update_best(nearest_lat_idx, nearest_lon_idx, dist)
    
    if best_i is None or best_j is None:
        raise ValueError("No suitable unmasked point" 
                         "found within the specified distance.")
    
    return best_i, best_j, best_lon, best_lat


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
        print(f'\nExtracting grid cells for {site_loc}')
        for i in tqdm(range(len(cubes))):
            model = model_names[i]
            i, j, pt_lon, pt_lat = find_ocean_pt(cubes[i], model, site_loc,
                                                 lat, lon)
            result.append([model, i, j, pt_lon, pt_lat])

        # Write the data to a file
        write_i_j(site_loc, result, lat, lon_orig)


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
