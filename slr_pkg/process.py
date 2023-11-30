"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import iris.analysis.cartography
import numpy
import scipy.stats

from slr_pkg import cubedata


class Regress:
    """
    Class to act as an interface to various regression functions.
    """
    def __init__(self, method='linear'):
        """
        Return an instance linking the appropriate regression functions to
        be used as a set of generically-named regression functions, given an
        initialization method.
        :param method: regression method to be used
        """
        self.method = method

        # Associate generic functions with method-specific functions
        if self.method == 'linear':
            self.detrend_scalar = self._linreg_detrend_scalar
            self.regress_t_scalar = self._linreg_regress_t_scalar
        # OTHER METHODS
        else:
            raise Exception(
                f'Given regression method is not implemented: {method}')

    def _linreg_detrend_scalar(self, cube_t, cube_slope):
        """
        Remove a linear trend from data in one cube with dimensions of (time)
        given by the slope in another cube.
        :param cube_t: iris.cube.Cube with a variable of dimensions (time)
        :param cube_slope: iris.cube.Cube with a dimensionless regression slope
        :return: iris.cube.Cube containing the detrended timeseries,
        and a numpy.ndarray containing the magnitude of the trend relative to
        the starting time corresponding to the times of cube_t
        """
        # Require cubes to have expected dimensions
        cube_t_coords = [i.name() for i in cube_t.dim_coords]
        cube_slope_coords = [i.name() for i in cube_slope.dim_coords]
        if not ('time' in cube_t_coords and len(cube_t_coords) == 1):
            raise Exception('cube_t does not have dimensions of (time)')
        if not len(cube_slope_coords) == 0:
            raise Exception('cube_slope is not dimensionless')

        # Output cube is based on input cube
        cube_detrend = cube_t.copy()

        # Convert times to the units of the slope cube
        if cube_detrend.coord('time').units != cube_slope.coord('time').units:
            convert_time_units(cube_detrend.coord('time'),
                               cube_slope.coord('time'))

        # Calculate the trend as a function of time
        nparr_trend = cube_detrend.coord('time').points * cube_slope.data
        nparr_trend -= nparr_trend[0]

        # Remove trend from the data
        cube_detrend.data -= nparr_trend

        # Value of the scalar evolving under the removed trend only
        nparr_trend += cube_t.data[0]

        # Metadata
        cubedata._derived(
            cube_detrend, [cube_t, cube_slope],
            derived_type='detrended',
            derived_long='%s, minus a linear trend as calculated from %s')
        cube_detrend.attributes['drift_correction'] = \
            f'{cube_detrend.var_name}: linear'

        return cube_detrend, nparr_trend

    def _linreg_regress_t_scalar(self, cube_t):
        """
        Regress (using a linear regression) data in a cube with dimensions
        of (time) against its time coordinate.
        :param cube_t: iris.cube.Cube with a variable of dimensions (time)
        :return: iris.cube.Cube containing the regression slope, and an
        iris.cube.Cube containing the correlation coefficient R^2
        """
        # Require cube to have expected dimensions
        cube_t_coords = [i.name() for i in cube_t.dim_coords]
        if not (cube_t.ndim == 1 and 'time' in cube_t_coords):
            raise Exception('cube_t does not have dimensions of (time)')
        if not len(cube_t.coord('time').points) > 1:
            raise Exception('The input cube must have at least 2 time ' +
                            'coordinates')

        # Output cube is based on cube_t
        cube_slope = cube_t.collapsed('time', iris.analysis.MEAN)
        cube_corr = cube_slope.copy()

        # Call scipy.stats.linregress on the cube data and its time coordinate
        nparr_regr = scipy.stats.linregress(cube_t.coord('time').points,
                                            cube_t.data)
        cube_slope.data = nparr_regr[0]
        cube_corr.data = numpy.square(nparr_regr[2])

        # Metadata
        cubedata._derived(
            cube_slope, [cube_t], var_name='linregslope',
            derived_type='linear_regression',
            derived_long='Slope of a linear regression of time (x) vs %s (y)')
        cubedata._derived(
            cube_corr, [cube_t], var_name='linregr2',
            derived_type='linear_correlation',
            derived_long='Coefficient of determination of time (x) vs %s (y)')

        return cube_slope, cube_corr


def convert_time_units(time_in, time_ref):
    """
    Convert one time coordinate to the units of another time coordinate.
    **METHOD:
    - First attempt the iris.coords.Coord.convert_units method.
    - If this fails (which it will do for certain units even if the resulting
    time coordinates would be valid), use the slightly more forgiving
    netCDF4 num2date/date2num method of converting. This will still fail if
    the attempted time coordinate conversion results in an illegal date.
    :param time_in: iris.coords.Coord (time) coordinate to be converted
    :param time_ref: iris.coords.Coord (time) coordinate containing the target
              units
    """
    # Check that the input coordinates are time coordinates
    if not ((type(time_in) is iris.coords.DimCoord or type(time_in) is
             iris.coords.AuxCoord) and
            (type(time_ref) is iris.coords.DimCoord or type(time_ref) is
             iris.coords.AuxCoord)):
        raise Exception('The provided inputs are not both iris coordinates')
    if not (time_in.standard_name == 'time' and
            time_ref.standard_name == 'time'):
        raise Exception('The coordinates provided do not have "time" as ' +
                        'the standard_name')

    # Time coordinate conversion
    try:
        # First attempt the iris.coords.Coord.convert_units method
        time_in.convert_units(time_ref)
    except Exception:

        # Otherwise attempt the slightly more forgiving netCDF4
        # num2date/date2num method
        date_points_in = time_in.units.num2date(time_in.points)
        date_bounds_in = time_in.units.num2date(time_in.bounds)

        num_points_out = time_ref.units.date2num(date_points_in)
        num_bounds_out = time_ref.units.date2num(date_bounds_in)

        time_in.points = num_points_out
        time_in.bounds = num_bounds_out
        time_in.units = time_ref.units

        date_points_out = time_in.units.num2date(time_in.points)
        date_bounds_out = time_in.units.num2date(time_in.bounds)

    # Check dates of converted times against those of original times
    if not ((date_points_in == date_points_out).all() and
            (date_bounds_in == date_bounds_out).all()):
        raise Exception('Unit conversion did not conserve original dates')


def _reject_auxcoord(cube_list):
    """
    Remove all cubes with AuxCoord time coordinates in a cube list.
    :param cube_list: iris.cube.CubeList
    :return: iris.cube.CubeList
    """
    # Require a cube or cubelist
    if not (type(cube_list) == iris.cube.CubeList or
            type(cube_list) == iris.cube.Cube):
        raise Exception('Input must be a cube or cubelist')

    # Convert cube to cubelist if necessary
    if isinstance(cube_list, iris.cube.Cube):
        cube_list = iris.cube.CubeList([cube_list])

    cube_list_out = iris.cube.CubeList(cube_list)  # Is this needed?

    for i_cube in cube_list:
        # Skip cube if no time coordinate
        coord_names = [i.name() for i in i_cube.coords()]
        if 'time' not in coord_names:
            continue

        # If time coordinate is an AuxCoord, print some info
        dc = i_cube.coord('time')
        if type(dc) == iris.coords.AuxCoord:
            time_min = dc.units.num2date(dc.points.min())
            time_max = dc.units.num2date(dc.points.max())
            print("    * slr.process.reject_auxcoord: Found a cube with an " +
                  "AuxCoord time coordinate (dates " +
                  f"{time_min.year}{time_min.month}{time_min.day}-" +
                  f"{time_max.year}{time_max.month}{time_max.day}),\n" +
                  "      which suggests a problem with the coordinate data. " +
                  "Skipping this cube..")

            # Remove cube from cubelist
            cube_list_out.remove(i_cube)

        # TEMPORARY: Guess time bounds if they are not present
        elif dc.bounds is None:
            dc.guess_bounds()

    return cube_list_out
