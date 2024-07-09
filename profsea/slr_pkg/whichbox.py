"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import numpy as np


def find_gridbox_indicies(cube, points, xcoord_name='longitude',
                          ycoord_name='latitude', drop=False):
    """
    Find the grid box indices containing the specified points.
    **NOTES:
    - Points just west of zero degrees longitude can lie outside the
    coordinate bounds for some models.
    - If this is the case, subtract 360 from the points, so they lie within
    the bounds of the model longitude coordinate.
    - Similarly, add 360 to any points smaller than the lower limit.
    :param cube: iris.cube.Cube containing zos field from CMIP models (
    should be 2D)
    :param points: a 2D numpy array of pairs of points (longitude, latitude)
    with dimensions (number of points, 2)
    :param xcoord_name: name of x-coordinate in the cube
    :param ycoord_name: name of y-coordinate in the cube
    :param drop: If True, return indices for the points which lie within the
    cube boundaries;
    If False, return indices of -99 for points which lie outside the
    boundaries.
    :return: grid box indices of location
    """
    # Get the grid box coordinates. Add bounds if not present.
    grid_lons = cube.coord(xcoord_name)
    grid_lats = cube.coord(ycoord_name)

    if not grid_lons.has_bounds():
        grid_lons.guess_bounds()
    if not grid_lats.has_bounds():
        grid_lats.guess_bounds()

    # Find the minimums and maximums of the cube's coordinate boundaries
    lon_lims = [grid_lons.bounds.min(), grid_lons.bounds.max()]
    lat_lims = [grid_lats.bounds.min(), grid_lats.bounds.max()]

    # Get the lons and lats of the points of interest (as copies)
    if isinstance(points, list):
        points = np.array(points)
    pt_lons, pt_lats = points.T

    if iscoordglobal(grid_lons):
        widx, = np.where(pt_lons > lon_lims[1])
        if len(widx) > 0:
            pt_lons[widx] -= 360
        widx, = np.where(pt_lons < lon_lims[0])
        if len(widx) > 0:
            pt_lons[widx] += 360

    # Find all points which lie within or on the boundaries of the model grid
    # boxes
    valid_lats = (pt_lats >= lat_lims[0]) & (pt_lats <= lat_lims[1])
    valid_lons = (pt_lons >= lon_lims[0]) & (pt_lons <= lon_lims[1])
    valid_pts = valid_lats & valid_lons

    if np.sum(valid_pts) == 0:
        raise ValueError('find_gridbox_indicies: None of the points lie '
                         + 'within the range of the coordinates')
    else:
        # Find the indices of the grid boxes containing the points
        if drop:
            idx_lon = [grid_lons.nearest_neighbour_index(x)
                       for x in pt_lons[valid_pts]]
            idx_lat = [grid_lats.nearest_neighbour_index(y)
                       for y in pt_lats[valid_pts]]
        else:
            idx_lon = [grid_lons.nearest_neighbour_index(x)
                       if l else -99 for x, l in zip(pt_lons, valid_pts)]
            idx_lat = [grid_lats.nearest_neighbour_index(y)
                       if l else -99 for y, l in zip(pt_lats, valid_pts)]
        idx = np.array([idx_lon, idx_lat]).T

    return idx


def iscoordglobal(coord, tol=0.01):
    """
    Tests if the longitude coordinate is global.
    :param coord: iris.cube.Cube coordinate to be tested
    :param tol: decide whether floating point numbers are tolerably equal
    :return: True if longitude coordinate is global, otherwise false.
    """

    if not ((0 <= coord.points.min() <= 360) and
            (0 <= coord.points.max() <= 360)):
        raise ValueError('All points must lie within 0 to 360')

    if not coord.has_bounds():
        coord.guess_bounds()
    diff = coord.bounds.max() - coord.bounds.min()

    return (diff > 360.0 - tol) & (diff < 360.0 + tol)


def wraplongitude(longitude):
    """
    Converts input longitude to values within a range 0 - 360 plus a
    specified base.
    :param longitude: site location's longitude value
    :return: converted longitude
    """
    wrapped_lons = ((longitude + 720.0) % 360)
    print(f' Converted Longitude from {longitude} to {wrapped_lons}')

    return wrapped_lons