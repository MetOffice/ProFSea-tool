"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import iris


def loadcube(files, ncvar=None, *args, **kwargs):
    """
    Load data using iris.load() with optional constraint by netcdf variable
    name.
    :param files: any sequence of file types accepted by iris.load()
    :param ncvar: netcdf variable name
    :param args: See iris.load() for other valid arguments
    :param kwargs: See iris.load() for other valid keyword arguments
    :return: an iris.cube.CubeList object
    """

    cubes = iris.load(files, *args, **kwargs)

    if ncvar:
        var_constraint = iris.Constraint(cube_func=lambda c:
                                         c.var_name == ncvar)
        cubes = cubes.extract(var_constraint)

    return cubes
