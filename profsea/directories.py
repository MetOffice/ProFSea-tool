"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import os

from profsea.config import settings


def makefolder(directory):
    """
    Check if subfolder exists, if not then creates the subfolder
    :param directory: directory of subfolder
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def read_dir():
    """
    Creates multiple directories from user settings
    :return: file paths required to run ProFSea tool
    """
    root_dir = settings["baseoutdir"]
    data_region = settings["siteinfo"]["region"]

    # output data directory - '' adds last /
    cmipdir = os.path.join(root_dir, data_region, 'data', 'cmip5', '')

    # output map directory
    mapdir = os.path.join(root_dir, data_region, 'figures', 'maps', '')

    # output zos data directory
    zosddir = os.path.join(root_dir, data_region, 'data', 'zos_regression', '')

    # output zos figure directory
    zosfdir = os.path.join(root_dir, data_region, 'figures',
                           'zos_regression', '')

    # output sea level projections data directory
    sealev_ddir = os.path.join(root_dir, data_region, 'data',
                               'sea_level_projections', '')

    # output sea level projections figure directory
    sealev_fdir = os.path.join(root_dir, data_region, 'figures',
                               'sea_level_projections', '')

    # output baseline sea level figure directory
    base_fdir = os.path.join(root_dir, data_region, 'figures',
                             'baseline_sea_level', '')

    return cmipdir, mapdir, zosddir, zosfdir, sealev_ddir, sealev_fdir, \
        base_fdir
