"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import numpy as np
import pandas as pd
import re
import itertools

from config import settings


def calc_xlim(str_id, tg_years, years):
    """
    Define the x-axis limits based on tide gauge and projection data
    :param str_id: identification variable to define between tide gauge or
    sea level projection
    :param tg_years: tide gauge years
    :param years: sea level projection years
    :return: minimum and maximum values
    """
    if str_id == 'proj':
        xmin = 2005
    elif str_id == 'tide':
        xmin = min([min(tg_years), min(years)])
        xmin = np.floor(xmin / 10) * 10
    xmax = settings['projection_end_year']

    return [xmin, xmax]


def calc_ylim(str_id, tg_amsl, df):
    """
    Define the y-axis limits based on tide gauge and projection data
    :param str_id: identification variable to define between tide gauge or
    sea level projection
    :param tg_amsl: tide gauge annual mean sea level data
    :param df: sea level projection data
    :return: minimum and maximum values
    """
    # Calculate y-axis limits based on sea level projections only
    # i.e. a list of dataframes for each RCP
    if isinstance(df, list):
        df_all = pd.concat(df, axis=0, ignore_index=True)
    # i.e. a single dataframe for one RCP
    else:
        df_all = df

    ymin = df_all.min(numeric_only=True).min()
    ymax = df_all.max(numeric_only=True).max()

    if str_id == 'tide':
        tg_ymin = min(tg_amsl)
        tg_ymin = np.floor(tg_ymin / 0.1) * 0.1
        # To include the zero line on plots
        if tg_ymin > -0.05:
            tg_ymin = -0.05
        ymin = min(tg_ymin, ymin)

    return [ymin-0.1, ymax+0.1]


def location_string(loc_string):
    """
    Re-format location string for use in plotting.
    Deals with , - _ ( ) and space
    :param loc_string: site location
    :return: simplified site location string
    """
    loc_title = []
    title_temp = re.split(r"[, |\-|_|)|(|\s]", loc_string)
    for string in title_temp:
        loc_title.append(string.capitalize())
    loc_title = " ".join(loc_title)

    if loc_title == 'Stanley Ii':
        loc_title = 'Stanley II'

    return loc_title


def plot_zeroline(ax, xvalues):
    """
    Plots a horizontal line where x=0
    :param ax: number of subplot
    :param xvalues: range of years
    """
    ax.plot(xvalues, [0., 0.], 'grey', linewidth=0.5)


def scenario_string(rcp_scen, i):
    """
    Re-format scenario string for use in plotting
    :param rcp_scen: RCP emission scenario
    :param i: loop level; -999 if not required
    :return: simplified scenario string
    """
    if i >= 0:
        rcp_label = rcp_scen[i].upper()
        rcp_label = rcp_label[:3] + ' ' + rcp_label[3] + '.' + rcp_label[4:]
    else:
        rcp_label = rcp_scen.upper()
        rcp_label = rcp_label[:3] + ' ' + rcp_label[3] + '.' + rcp_label[4:]

    return rcp_label


def ukcp18_colours():
    """
    Define plotting colour schemes as used in UKCP18 Marine Report
    :return: UKCP18 colour schemes
    """
    rcp_colours = {'rcp85': 'firebrick',
                   'rcp45': 'steelblue',
                   'rcp26': 'darkblue',
                   }

    comp_parts_colours = {'antnet': 'b',
                          'greennet': 'g',
                          'glacier': 'c',
                          'landwater': 'purple',
                          'sum': 'k',
                          'gia': 'orange',
                          'ocean': 'r'}

    return rcp_colours, comp_parts_colours


def get_emulator_colors(num_scenarios, get_all=False):
        colors = [
            '#031326', '#13385a', '#45587a', '#6a5e76', 
            '#8d616d', '#b86462', '#e37861', '#e8a077']
        
        def get_equidistant_indices(num_colors, num_iterations):
            if num_iterations >= num_colors:
                return list(range(num_colors))
            
            # Avoid first and last index if possible
            if num_iterations > 1:
                start = 1
                stop = num_colors - 2
                indices = np.linspace(start, stop, num=num_iterations, dtype=int)
            else:
                # If only one iteration, take the middle color
                indices = [num_colors // 2]
            
            return indices
        
        # Return a cyclical color generator 
        if (num_scenarios >= len(colors)) or get_all:
            return itertools.cycle(colors)
        
        color_idxs = get_equidistant_indices(len(colors), num_scenarios)
        return itertools.cycle([colors[i] for i in color_idxs])


def ukcp18_labels():
    """
    Define sea level component labels as used in UKCP18 Marine Report
    :return: UKCP18 component labels
    """
    comp_parts_labels = {'antnet': 'Antarctica',
                         'greennet': 'Greenland',
                         'glacier': 'Glaciers',
                         'landwater': 'Land Water',
                         'sum': 'Local Total',
                         'gia': 'GIA',
                         'ocean': 'Ocean'}

    return comp_parts_labels


def multi_index_values(df_list):
    """
    Get the values of the multi-index: years and percentile values.
    :param df_list: DataFrame list of regional sea level projections
    :return: years and percentile values
    """
    df = df_list[0]
    proj_years = np.sort(list(set(list(df.index.get_level_values('year')))))
    percentiles_all = np.sort(
        [float(v) for v in list(set(list(
            df.index.get_level_values('percentile'))))])

    return proj_years, percentiles_all


def extract_comp_sl(df, percentiles, comp):
    """
    Get the sums of all components of local sea level projections.
    :param df: global or regional DataFrame of sea level projections
    :param percentiles: specified percentiles
    :param comp: components of sea level
    :return: sum of sea level components at lower, middle and upper percentile
    """
    # 5th percentile - based on UKCP18 percentile levels
    rlow = df.xs(percentiles[0], level='percentile')[comp].to_numpy(copy=True)
    # 50th percentile
    rmid = df.xs(percentiles[4], level='percentile')[comp].to_numpy(copy=True)
    # 95th percentile
    rupp = df.xs(percentiles[8], level='percentile')[comp].to_numpy(copy=True)

    return rlow, rmid, rupp


def plot_tg_data(ax, nflag, flag, tg_years, non_missing, tg_amsl, tg_name):
    """
    Plot the annual mean sea levels from the tide gauge data.
    :param ax: subplot number
    :param nflag: number of flagged years
    :param flag: flagged data
    :param tg_years: tide gauge years
    :param non_missing: boolean to indicate NaN values
    :param tg_amsl: annual mean sea level data
    :param tg_name: tide gauge name
    """
    if nflag > 0:
        # There are some years with less than min_valid_fraction of flag data;
        # plot these annual means as open symbols.
        print(f'Tide gauge data has been flagged for attention - ' +
              f'{tg_years[(flag & non_missing)]}')
        ax.plot(tg_years[(flag & non_missing)], tg_amsl[(flag & non_missing)],
                marker='o', mec='black', mfc='None',
                markersize=3, linestyle='None', label='TG flagged')
    if nflag < len(flag):
        ax.plot(tg_years[(~flag & non_missing)],
                tg_amsl[(~flag & non_missing)], 'ko', markersize=3,
                label=f'{location_string(tg_name)} TG')
