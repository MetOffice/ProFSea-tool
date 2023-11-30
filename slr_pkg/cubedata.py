"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

from datetime import datetime
import inspect
import os
import cf_units
import iris

from slr_pkg import process


def _derived(cube, cube_src, var_name=None, derived_type='unknown',
             derived_long=None):
    """
    Write metadata attributes to a cube describing the derived variable
    contained within it.
    NOTES:
    * Much of this metadata is probably unnecessary. However, derived_type
    is currently required in slr.component.Dynamic to recognise when a
    scaling map is provided as an input argument.
    * Currently the units of the derived variable are unset for
    compatibility purposes.
    :param cube: cube to which metadata will be written
    :param cube_src: list or cubelist of the cubes used to generate the
    derived variable
    :param var_name: new variable name. If None, the existing variable name
    is used.
    :param derived_type: short name describing the process for creating the
    derived variable
    :param derived_long: new long_name attribute. Can either be given as a
    regular string or as a formattable string corresponding to the cubes in
    cube_src. In the latter case, the derived_var cube attribute will be
    used to format the string if present, else the var_name attribute will
    be used
        i.e. "Product of %s and %s"%(cube1.var_name,cube2.var_name)
    If None is given for this keyword, the existing long_name is used
    """

    # Ensure cube_src is an iterable
    if type(cube_src) == iris.cube.Cube:
        cube_src = [cube_src]

    # New var_name, if applicable
    if var_name is None:
        var_name = cube.var_name

    # Coordinate information, describing lat/lon coordinate if sliced to scalar
    cube_coord = ''
    for i_cube in cube_src:
        coord_name = [i.name() for i in i_cube.coords()]
        if ('longitude' in coord_name and 'latitude' in coord_name) and \
                (len(i_cube.coord('longitude').points) == 1 and
                 len(i_cube.coord('latitude').points) == 1):
            if i_cube.coord('longitude').points[0] < 0:
                cube_lon = ('%sW' % i_cube.coord(
                    'longitude').points[0]).replace('-', '')
            else:
                cube_lon = '%sE' % i_cube.coord('longitude').points[0]
            if i_cube.coord('latitude').points[0] < 0:
                cube_lat = ('%sS' % i_cube.coord(
                    'latitude').points[0]).replace('-', '')
            else:
                cube_lat = '%sN' % i_cube.coord('latitude').points[0]
            cube_coord = '%s %s' % (cube_lat, cube_lon)
            break

    # Derived history, describing derived variable in a "function(args)"
    # format and inheriting previous values set by this function (can get
    # rather long..)
    derived_hist = []
    for i_cube in cube_src:
        if 'derived_hist' in list(i_cube.attributes.keys()):
            derived_hist += [i_cube.attributes['derived_hist']]
        else:
            derived_hist += [i_cube.var_name]

    # Derived variable, describing derived variable as for derived_hist but
    # not inheriting any previous values
    derived_var = [i_cube.var_name for i_cube in cube_src]

    # Derived period, describing data periods comprising the derived variable
    # in a form corresponding to derived_hist such that
    # "function_1(args,function_2(args))" corresponds to
    # "(dates_function_1,(dates_function_2))" (can also get rather long..)
    derived_period = []
    for i_cube in cube_src:
        if 'derived_period' in list(i_cube.attributes.keys()):
            derived_period += [i_cube.attributes['derived_period']]
        elif 'period' in list(i_cube.attributes.keys()):
            derived_period += [i_cube.attributes['period']]
        else:
            try:
                period_min = i_cube.coord('time').units.num2date(
                    i_cube.coord('time').bounds[:, 0].min())
                period_max = i_cube.coord('time').units.num2date(
                    i_cube.coord('time').bounds[:, 1].max())
                derived_period += ['-'.join(['%04d/%02d' % (i.year, i.month)
                                             for i in [period_min,
                                                       period_max]])]
            except Exception:
                derived_period += ['--']

    # Check derived_long keyword: if not set then use a generic string, check
    # string has sufficient
    # string formatting statements if a formattable string is being used
    # (contains %s, %i etc)
    if derived_long is None:
        derived_long = 'Unspecified derived variable based on ' + \
                       ', '.join(['%s'] * len(derived_hist))
    elif derived_long.count('%') != len(cube_src) and \
            derived_long.count('%') > 0:
        raise TypeError('The number of string formatting statements in ' +
                        f'derived_long ({derived_long.count("%")} does not'
                        f' match the number of source cubes ({len(cube_src)})')

    # History attribute entry; add the calling function and arguments
    try:
        cube_history = str(cube.attributes['history'])
    except KeyError:
        cube_history = ''
    history_callfunc = os.sep.join([inspect.stack()[1][1],
                                    inspect.stack()[1][3]])
    history = '%s: %s(%s)\n' % (datetime.now().ctime(), history_callfunc,
                                ','.join(derived_var))

    # Drift correction, a record of the drift-corrected variables
    drift_corr = []
    for i_cube in cube_src:
        if 'drift_correction' in list(i_cube.attributes.keys()):
            drift_corr += [i_cube.attributes['drift_correction']]

    # Apply string formatting where required
    if derived_long.count('%') > 0:
        derived_long = derived_long % tuple(derived_hist)
    derived_hist = '%s(%s)' % (derived_type, ','.join(derived_hist))
    derived_var = '%s(%s)' % (derived_type, ','.join(derived_var))
    derived_period = '(%s)' % (','.join(derived_period))

    # If a function of several drift-corrected variables, write as "func(a,b)".
    if len(drift_corr) > 1:
        drift_corr = '%s(%s)' % (var_name, ', '.join(drift_corr))

    # Otherwise if a function of one drift-corrected variable, i.e. "func(a)",
    # write "new_var: old_var" if the derived variable has a different name
    elif len(drift_corr) == 1:
        parse_result_1 = drift_corr[0].split(': ')[0]
        parse_result_2 = drift_corr[0].split('(')[0]
        if parse_result_1 != var_name and parse_result_2 != var_name:
            drift_corr = '%s(%s)' % (var_name, drift_corr[0])
        else:
            drift_corr = drift_corr[0]

    # Write attributes
    cube.long_name = derived_long
    cube.var_name = var_name
    if len(cube_coord) > 0:
        cube.attributes['coordinate'] = cube_coord
    # REPLACE WITH SPECIFIC CUBE.ADD_HISTORY CALL
    cube.attributes['history'] = history + cube_history
    cube.attributes['derived_hist'] = derived_hist
    cube.attributes['derived_var'] = derived_var
    cube.attributes['derived_type'] = derived_type
    cube.attributes['derived_period'] = derived_period
    if len(drift_corr) > 0:
        cube.attributes['drift_correction'] = drift_corr
    cube.units = cf_units.Unit(None)


def read_zos_cube(cmip_file):
    """
    Read cube of CMIP sea level data, applying appropriate time bounds.
    Could be: dynamic sea level (zos), global mean thermosteric (zostoga) or
    piControl (concatenated)
    :param cmip_file: filepath and filename as string
    :return: cube of sea level data
    """
    cube = iris.load_cube(cmip_file, iris.Constraint())

    # Guess time bounds if they are not present
    if 'time' in [i.name() for i in cube.coords()] and \
            cube.coord('time').bounds is None:
        cube.coord('time').guess_bounds()

    # Reject cube if it has an AuxCoord time coordinate
    cube = process._reject_auxcoord(cube)
    if len(cube) == 0:
        raise Exception(
            f'The file {cmip_file} did not have a CF-compliant time '
            f'coordinate')

    return cube
