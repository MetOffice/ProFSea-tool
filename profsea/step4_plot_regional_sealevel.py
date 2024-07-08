"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import os
import iris
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from profsea.config import settings
from profsea.tide_gauge_locations import extract_site_info, find_nearest_station_id
from profsea.slr_pkg import abbreviate_location_name, choose_montecarlo_dir  # found in __init.py__
from profsea.surge import tide_gauge_library as tgl
from profsea.directories import read_dir, makefolder
from profsea.plotting_libraries import location_string, scenario_string, \
    ukcp18_colours, ukcp18_labels, calc_xlim, calc_ylim, plot_zeroline


def compute_uncertainties(df_r_list, scenarios, tg_years, tg_amsl):
    """
    Function to estimate uncertainties following Hawkins and Sutton (2009).
    :param df_r_list: regional sea level projections
    :param scenarios: emissions scenario
    :param tg_years: tide gauge years
    :param tg_amsl: annual mean sea level data
    :return: years, scenario uncertainty, model uncertainty, internal
    variability
    """
    nyrs = np.range(2007, settings["projection_end_year"] + 1).size
    allpmid = np.zeros([3, nyrs])
    allunc = np.zeros([3, nyrs])

    # Estimate internal variability from de-trended gauge data
    tg_years_arr = np.array(tg_years, dtype='int')
    vals = np.ma.masked_values(tg_amsl, -99999.)
    IntV = compute_variability(tg_years_arr / 1000., vals / 1.)

    # Get the values of the multi-index: years and percentile values
    years, percentiles = multi_index_values(df_r_list)

    for rcp_count, _ in enumerate(scenarios):
        # Get the sums of local sea level projections
        df_R = df_r_list[rcp_count]
        rlow, rmid, rupp = extract_comp_sl(df_R, percentiles, 'sum')

        allpmid[rcp_count, :] = rmid
        allunc[rcp_count, :] = (rupp - rlow) * 0.5

    # Scenario Uncertainty (expressed as 90% confidence interval)
    UncS = np.std(allpmid, axis=0) * 1.645
    # Model Uncertainty (already expressed as 90% confidence interval)
    UncM = np.mean(allunc, axis=0)

    return years, UncS, UncM, IntV


def compute_variability(x, y, factor=1.645):
    """
    Estimate internal variability from de-trended gauge data.
    :param x: tide gauge temporal data
    :param y: tide gauge data
    :param factor: multiplication factor
    :return: internal variability component of uncertainty
    """
    mask = np.ma.getmask(y)
    index = np.where(mask is True)
    new_x = np.delete(x, index)
    new_y = np.delete(y, index)
    fit = np.polyfit(new_x, new_y, 1)
    fit_data = new_x * fit[0] + fit[1]
    stdev = np.std(new_y - fit_data)

    return stdev * factor


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


def plot_figure_one(r_df_list, site_name, scenarios, fig_dir):
    """
    Subplot 1 - Local sea level projections for all RCPs - sum total.
    Subplot 2 - Local sea level projections for RCP8.5 - sum total and
    component parts.
    :param r_df_list: regional sea level projections
    :param site_name: site location
    :param scenarios: emissions scenario
    :param fig_dir: figure directory
    """
    # Get the values of the multi-index: years and percentile values
    years, percentiles = multi_index_values(r_df_list)

    # UKCP18 colour scheme for sea level components
    rcp_colours = ukcp18_colours()[0]

    # Calculate the x-axis and y-axis limit
    xlim = calc_xlim('proj', [], years)
    ylim = calc_ylim('proj', [], r_df_list)

    fig = plt.figure(figsize=(7.68, 3.5))
    matplotlib.rcParams['font.size'] = 7.5

    # First figure, regional sea level projections and uncertainties under each
    # scenario
    ax = fig.add_subplot(1, 2, 1)

    for rcp_count, rcp_str in enumerate(scenarios):
        # Get the sums of all components of local sea level projections
        r_df_rcp = r_df_list[rcp_count]
        rlow, rmid, rupp = extract_comp_sl(r_df_rcp, percentiles, 'sum')

        label = scenario_string(scenarios, rcp_count)
        ax.fill_between(years, rlow, rupp, alpha=0.3,
                        color=rcp_colours[rcp_str], linewidth=0)
        ax.plot(years, rmid, color=rcp_colours[rcp_str], label=label)

    plot_zeroline(ax, xlim)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('left')

    ax.set_title(f'Local sea level - {location_string(site_name)}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Sea Level Change (m)')

    ax.legend(loc='upper left', frameon=False)

    # Second figure showing the various components of sea level projections
    bx = fig.add_subplot(1, 2, 2)

    # UKCP18 colour scheme for sea level components
    comp_colours = ukcp18_colours()[1]
    # UKCP18 labels for sea level components
    comp_labels = ukcp18_labels()

    # Show components for the RCP8.5 scenario (only plot the uncertainty for
    # sum and thermal expansion)
    for rcp_count, rcp_str in enumerate(scenarios):
        if scenarios[rcp_count] == 'rcp85':
            break
    else:
        raise ValueError('RCP8.5 scenario not in list')

    r_df_rcp = r_df_list[rcp_count]

    for comp in ['sum', 'ocean', 'greennet', 'antnet', 'glacier', 'landwater',
                 'gia']:
        # Get the components of regional sea level projections
        clow_85, cmid_85, cupp_85 = extract_comp_sl(r_df_rcp, percentiles,
                                                    comp)
        colour = comp_colours[comp]
        label = comp_labels[comp]

        if comp in ['sum', 'ocean']:
            bx.fill_between(years, cupp_85, clow_85, facecolor=colour,
                            alpha=0.3, edgecolor='None')

            if comp == 'sum':
                bx.plot(years, cmid_85, colour, linewidth=1.5, label=label)
            else:
                bx.plot(years, cmid_85, colour, linewidth=1.5, label=label)
        else:
            bx.plot(years, cmid_85, colour, linewidth=1.5, label=label)

    plot_zeroline(bx, xlim)
    bx.set_xlim(xlim)
    bx.set_ylim(ylim)
    bx.yaxis.set_ticks_position('both')
    bx.yaxis.set_label_position('left')

    bx.legend(loc='upper left', frameon=False)
    bx.set_title(f'Sea level components - {scenario_string(rcp_str, -999)}')
    bx.set_xlabel('Year')

    # Save the figure
    fig.tight_layout()
    ffile = f'{fig_dir}01_{site_name}.png'
    plt.savefig(ffile, dpi=300, format='png')
    plt.close()


def plot_figure_two(r_df_list, tg_name, nflag, flag, tg_years, non_missing,
                    tg_amsl, site_name, scenarios, fig_dir):
    """
    Local sea level projections for specified RCPs - sum total, and annual
    mean tide gauge observations.
    :param r_df_list: regional sea level projections
    :param tg_name: tide gauge name
    :param nflag: number of flagged years
    :param flag: flagged data
    :param tg_years: tide gauge years
    :param non_missing: boolean to indicate NaN values
    :param tg_amsl: annual mean sea level data
    :param site_name: site location
    :param scenarios: emissions scenario
    :param fig_dir:figure directory
    """
    # Get the values of the multi-index: years and percentile values
    years, percentiles = multi_index_values(r_df_list)

    # UKCP18 colour scheme for RCP scenarios
    rcp_colours = ukcp18_colours()[0]

    fig = plt.figure(figsize=(5, 4.5))
    matplotlib.rcParams['font.size'] = 10

    # Draw regional sea level projections and uncertainties under each scenario
    ax = fig.add_subplot(1, 1, 1)

    for rcp_count, rcp_str in enumerate(scenarios):
        # Get the sums of all components of local sea level projections
        r_df_rcp = r_df_list[rcp_count]
        rlow, rmid, rupp = extract_comp_sl(r_df_rcp, percentiles, 'sum')

        label = scenario_string(scenarios, rcp_count)
        ax.fill_between(years, rlow, rupp, alpha=0.3,
                        color=rcp_colours[rcp_str], linewidth=0)
        ax.plot(years, rmid, color=rcp_colours[rcp_str], label=label)

    # Calculate sensible x-axis range and y-axis range based on tide gauge data
    # and projections
    xlim = calc_xlim('tide', tg_years, years)
    df_tmp = pd.DataFrame([rlow, rmid, rupp])
    ylim = calc_ylim('tide', tg_amsl, df_tmp)

    plot_zeroline(ax, xlim)

    # Plot the annual mean sea levels from the tide gauge data
    plot_tg_data(ax, nflag, flag, tg_years, non_missing, tg_amsl, tg_name)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('left')
    ax.set_title(f'Local sea level - {location_string(site_name)}')
    ax.set_ylabel('Sea Level Change (m)')
    ax.set_xlabel('Year')
    ax.legend(loc='upper left', frameon=False)

    # Save the figure
    fig.tight_layout()
    ffile = f'{fig_dir}02_{site_name}.png'
    plt.savefig(ffile, dpi=300, format='png')
    plt.close()


def plot_figure_three(g_df_list, r_df_list, global_low, global_mid, global_upp,
                      site_name, scenarios, fig_dir):
    """
    Subplot 1 - Comparison of IPCC AR5 + Levermann global projections and
    global projections developed here.
    Subplot 2 - Comparison of global projections developed here and local sea
    level projections.
    :param g_df_list: global sea level projections
    :param r_df_list: regional sea level projections
    :param global_low: IPCC AR5 + Levermann global projections 5th percentile
    :param global_mid: IPCC AR5 + Levermann global projections 50th percentile
    :param global_upp: IPCC AR5 + Levermann global projections 95th percentile
    :param site_name: site location
    :param scenarios: emissions scenario
    :param fig_dir: figure directory
    """
    # Get the values of the multi-index: years and percentile values
    years, percentiles = multi_index_values(r_df_list)

    # UKCP18 colour scheme for RCP scenarios
    rcp_colours = ukcp18_colours()[0]

    # Calculate sensible x-axis range based on projections
    xlim = calc_xlim('proj', [], years)

    for rcp_count, rcp_str in enumerate(scenarios):
        # Get the sums of all components of global sea level projections
        g_df_rcp = g_df_list[rcp_count]
        glow, gmid, gupp = extract_comp_sl(g_df_rcp, percentiles, 'sum')

        # Get the sums of all components of global sea level projections
        r_df_rcp = r_df_list[rcp_count]
        rlow, rmid, rupp = extract_comp_sl(r_df_rcp, percentiles, 'sum')

        # Calculate sensible y-axis range based on projections
        r_ylim = calc_ylim('proj', [], r_df_rcp)
        g_ylim = calc_ylim('proj', [], g_df_rcp)
        ylim = [min(r_ylim[0], g_ylim[0]), max(r_ylim[1], g_ylim[1])]

        # Plot comparison of global sea level projections
        # Check that global is equivalent to the IPCC AR5 + Levermann sea
        # level projections
        fig = plt.figure(figsize=(7.68, 3.5))
        matplotlib.rcParams['font.size'] = 7.5

        ax = fig.add_subplot(1, 2, 1)
        # IPCC
        ax.plot(years, global_mid[rcp_count], color='k', linewidth=1.5,
                linestyle='-', label='IPCC AR5 + Levermann')
        ax.plot(years, global_low[rcp_count], color='k', linewidth=1.5,
                linestyle=':')
        ax.plot(years, global_upp[rcp_count], color='k', linewidth=1.5,
                linestyle=':')
        # Global
        ax.plot(years, gmid, color='orange', linewidth=1.5, linestyle='--',
                label='Global Total')
        ax.fill_between(years, glow, gupp, color='orange', alpha=0.3,
                        linewidth=0, edgecolor='None')

        plot_zeroline(ax, xlim)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.set_label_position('left')

        ax.set_title(f'Global sea level - {scenario_string(rcp_str, -999)}')
        ax.set_ylabel('Sea Level Change (m)')
        ax.set_xlabel('Year')
        ax.legend(loc='upper left', frameon=False)

        # Plot regional sea level projections, showing both error ranges
        bx = fig.add_subplot(1, 2, 2)
        # Global
        bx.plot(years, gmid, color='k', linewidth=1.5, linestyle='-',
                label='Global Total')
        bx.plot(years, glow, color='k', linewidth=1.5, linestyle=':')
        bx.plot(years, gupp, color='k', linewidth=1.5, linestyle=':')
        # Local
        bx.plot(years, rmid, color=rcp_colours[rcp_str], linewidth=1.5,
                linestyle='--', label=location_string(site_name))
        bx.fill_between(years, rlow, rupp, color=rcp_colours[rcp_str],
                        alpha=0.3, linewidth=0, edgecolor='None')

        plot_zeroline(bx, xlim)
        bx.set_xlim(xlim)
        bx.set_ylim(ylim)
        bx.yaxis.set_ticks_position('both')
        bx.yaxis.set_label_position('left')

        bx.set_title(f'Local sea level - {location_string(site_name)} - '
                     f'{scenario_string(rcp_str, -999)}')
        bx.set_xlabel('Year')
        bx.legend(loc='upper left', frameon=False)

        # Save the figure
        fig.tight_layout()
        outfile = f'{fig_dir}03_{site_name}_{rcp_str}.png'
        plt.savefig(outfile, dpi=300, format='png')
        plt.close()


def plot_figure_four(r_df_list, site_name, scenarios, fig_dir):
    """
    Local sea level projections for all RCPs - sum total.
    Same style as Figure 3.1.4 of UKCP18 Marine Report (pg 16).
    :param r_df_list: regional sea level projections
    :param site_name: site location
    :param scenarios: emissions scenario
    :param fig_dir: figure directory
    """
    # Get the values of the multi-index: years and percentile values
    years, percentiles = multi_index_values(r_df_list)

    # UKCP18 colour scheme for RCP scenarios
    rcp_colours = ukcp18_colours()[0]

    fig = plt.figure(figsize=(5, 4.5))
    matplotlib.rcParams['font.size'] = 10

    # First figure, regional sea level projections and uncertainties under each
    # scenario
    ax = fig.add_subplot(1, 1, 1)

    for rcp_count, rcp_str in enumerate(scenarios):
        # Get the sums of regional sea level projections
        r_df_rcp = r_df_list[rcp_count]
        rlow, rmid, rupp = extract_comp_sl(r_df_rcp, percentiles, 'sum')

        label = scenario_string(scenarios, rcp_count)
        plt.plot(years, rmid, color=rcp_colours[rcp_str], linewidth=4.0,
                 label=label)
        plt.plot(years, rupp, color=rcp_colours[rcp_str], linewidth=1.0,
                 linestyle='--')
        plt.plot(years, rlow, color=rcp_colours[rcp_str], linewidth=1.0,
                 linestyle='--')

    # Calculate sensible x-axis range and y-axis range based on tide gauge data
    # and projections
    xlim = calc_xlim('proj', [], years)
    df_tmp = pd.DataFrame([rlow, rmid, rupp])
    ylim = calc_ylim('proj', [], df_tmp)

    plot_zeroline(ax, xlim)
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('left')

    plt.title(f'Local sea level - {location_string(site_name)}')
    plt.ylabel('Sea Level Change (m)')
    plt.xlabel('Year')
    plt.legend(loc='upper left', frameon=False)

    # Save the figure
    fig.tight_layout()
    outfile = f'{fig_dir}04_{site_name}.png'
    plt.savefig(outfile, dpi=300, format='png')
    plt.close()


def plot_figure_five(g_df_list, r_df_list, site_name, scenarios, fig_dir):
    """
    Comparison of global mean sea level projections calculated by the tool and
    local sea level projections - sum total
    Also shown local sea level projections - component parts
    Subplot 1, 2, 3 - RCP2.6, RCP4.5, RCP8.5 respectively.
    :param g_df_list: global sea level projections
    :param r_df_list: regional sea level projections
    :param site_name: site location
    :param scenarios: emissions scenario
    :param fig_dir: figure directory
    """
    # Get the values of the multi-index: years and percentile values
    years, percentiles = multi_index_values(r_df_list)

    # UKCP18 colour scheme for sea level components
    comp_colours = ukcp18_colours()[1]
    # UKCP18 labels for sea level components
    comp_labels = ukcp18_labels()

    # Calculate sensible x-axis range and y-axis range based on projections
    xlim = calc_xlim('proj', [], years)
    r_ylim = calc_ylim('proj', [], r_df_list)
    g_ylim = calc_ylim('proj', [], g_df_list)
    ylim = [min(r_ylim[0], g_ylim[0]), max(r_ylim[1], g_ylim[1])]

    fig = plt.figure(figsize=(7.68, 3.5))
    matplotlib.rcParams['font.size'] = 7.5

    # Loop through RCPs for subplots
    for rcp_count, _ in enumerate(scenarios):
        ax = fig.add_subplot(1, 3, rcp_count + 1)

        # Get the sums of global sea level projections
        g_df_rcp = g_df_list[rcp_count]
        glow, gmid, gupp = extract_comp_sl(g_df_rcp, percentiles, 'sum')

        ax.plot(years, gmid, color='k', linewidth=1.5, linestyle='--',
                label='Global Total')
        ax.plot(years, glow, color='k', linewidth=1.5, linestyle=':')
        ax.plot(years, gupp, color='k', linewidth=1.5, linestyle=':')

        # Get the sums of local sea level projections
        r_df_rcp = r_df_list[rcp_count]

        for comp in ['sum', 'ocean', 'greennet', 'antnet', 'glacier',
                     'landwater', 'gia']:
            # Get the components of regional sea level projections
            clow, cmid, cupp = extract_comp_sl(r_df_rcp, percentiles, comp)

            colour = comp_colours[comp]
            label = comp_labels[comp]

            if comp in ['sum', 'ocean']:
                ax.plot(years, cmid, colour, linewidth=1.5, label=label)
                ax.fill_between(years, cupp, clow, facecolor=colour, alpha=0.3,
                                edgecolor='None')
            else:
                ax.plot(years, cmid, colour, linewidth=1.5, label=label)

        plot_zeroline(ax, xlim)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.set_label_position('left')

        ax.set_xlabel('Year')
        ax.set_title(f'{location_string(site_name)} - '
                     f'{scenario_string(scenarios[rcp_count], -999)}')
        if rcp_count == 0:
            ax.set_ylabel('Sea Level Change (m)')

    # Put legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the figure
    fig.tight_layout()
    outfile = f'{fig_dir}05_{site_name}.png'
    plt.savefig(outfile, dpi=300, format='png')
    plt.close()


def plot_figure_six(r_df_list, global_low, global_mid, global_upp,
                    site_name, scenarios, fig_dir):
    """
    Comparison of IPCC AR5 + Levermann projections and local sea level
    projections - sum total and component parts.
    Subplot 1, 2, 3 - RCP2.6, RCP4.5, RCP8.5 respectively.
    :param r_df_list: regional sea level projections
    :param global_low: IPCC AR5 + Levermann global projections 5th percentile
    :param global_mid: IPCC AR5 + Levermann global projections 50th percentile
    :param global_upp: IPCC AR5 + Levermann global projections 95th percentile
    :param site_name: site location
    :param scenarios: emissions scenario
    :param fig_dir: figure directory
    """
    # Get the values of the multi-index: years and percentile values
    years, percentiles = multi_index_values(r_df_list)

    # UKCP18 colour scheme for sea level components
    comp_colours = ukcp18_colours()[1]
    # UKCP18 labels for sea level components
    comp_labels = ukcp18_labels()

    # Calculate sensible x-axis range and y-axis range based on projections
    xlim = calc_xlim('proj', [], years)
    r_ylim = calc_ylim('proj', [], r_df_list)
    # IPCC AR5 + Levermann projections are read in as an array
    g_ylim = [min(global_low[0]-0.1), max(global_upp[2])+0.1]
    ylim = [min(r_ylim[0], g_ylim[0]), max(r_ylim[1], g_ylim[1])]

    fig = plt.figure(figsize=(7.68, 3.5))
    matplotlib.rcParams['font.size'] = 7.5

    # Loop through RCPs for subplots
    for rcp_count, _ in enumerate(scenarios):
        ax = fig.add_subplot(1, 3, rcp_count + 1)

        # Get the sums of all components of global sea level projections
        cmid_g = global_mid[rcp_count]
        clow_g = global_low[rcp_count]
        cupp_g = global_upp[rcp_count]

        ax.plot(years, cmid_g, color='k', linewidth=1.5, linestyle='--',
                label='IPCC AR5 + Levermann')
        ax.plot(years, clow_g, color='k', linewidth=1.5, linestyle=':')
        ax.plot(years, cupp_g, color='k', linewidth=1.5, linestyle=':')

        # Get the sums of all components of local sea level projections
        r_df_rcp = r_df_list[rcp_count]
        for comp in ['sum', 'ocean', 'greennet', 'antnet', 'glacier',
                     'landwater', 'gia']:
            # Get the sums of regional sea level projections
            clow, cmid, cupp = extract_comp_sl(r_df_rcp, percentiles, comp)

            colour = comp_colours[comp]
            label = comp_labels[comp]

            if comp in ['sum', 'ocean']:
                ax.plot(years, cmid, colour, linewidth=1.5, label=label)
                ax.fill_between(years, cupp, clow, facecolor=colour, alpha=0.3,
                                edgecolor='None')
            else:
                ax.plot(years, cmid, colour, linewidth=1.5, label=label)

        plot_zeroline(ax, xlim)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.set_label_position('left')

        ax.set_xlabel('Year')
        ax.set_title(f'{location_string(site_name)} - '
                     f'{scenario_string(scenarios[rcp_count], -999)}')
        if rcp_count == 0:
            ax.set_ylabel('Sea Level Change (m)')

    # Put legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the figure
    fig.tight_layout()
    outfile = f'{fig_dir}06_{site_name}.png'
    plt.savefig(outfile, dpi=300, format='png')
    plt.close()


def plot_figure_seven(g_df_list, r_df_list, site_name, scenarios, fig_dir):
    """
    Subplot 1 - Global sea level projections for RCP8.5 - total and component
    parts, including uncertainty range.
    Subplot 2 - Local sea level projections for RCP8.5 - total and component
    parts, including uncertainty range.
    Same style as Figure 4 Howard and Palmer (2019).
    :param g_df_list: global sea level projections
    :param r_df_list: regional sea level projections
    :param site_name: site location
    :param scenarios: emission scenario
    :param fig_dir: figure directory
    """
    # Get the values of the multi-index: years and percentile values
    _, percentiles = multi_index_values(r_df_list)

    # UKCP18 colour scheme for sea level components
    comp_colours = ukcp18_colours()[1]
    # UKCP18 labels for sea level components
    comp_labels = ukcp18_labels()

    # Calculate sensible x-axis range and y-axis range
    xlim = ([-0.4, 7.4])
    r_ylim = calc_ylim('proj', [], r_df_list)
    g_ylim = calc_ylim('proj', [], g_df_list)
    ylim = [min(r_ylim[0], g_ylim[0]), max(r_ylim[1], g_ylim[1])]

    # Show components for the RCP8.5 scenario (only plot the uncertainty for
    # sum and thermal expansion)
    for rcp_count, rcp_str in enumerate(scenarios):
        if scenarios[rcp_count] == 'rcp85':
            break
    else:
        raise ValueError('RCP8.5 scenario not in list')
    r_df_rcp = r_df_list[rcp_count]
    g_df_rcp = g_df_list[rcp_count]

    # -------------------------------------------------------------------------
    # First figure, global sea level projections and uncertainties under
    # RCP 8.5 and component parts
    fig = plt.figure(figsize=(7.68, 4.5))
    matplotlib.rcParams['font.size'] = 9

    ax_labels = []
    ax = fig.add_subplot(1, 2, 1)
    for cc, comp in enumerate(['sum', 'ocean', 'antnet', 'greennet', 'glacier',
                               'landwater']):
        # Get the components of regional sea level projections
        clow_85_G, cmid_85_G, cupp_85_G = extract_comp_sl(g_df_rcp,
                                                          percentiles, comp)

        colour = comp_colours[comp]
        label = comp_labels[comp]
        ax_labels.append(label)
        xpts = [cc - 0.4, cc + 0.4]

        ax.fill_between(xpts, clow_85_G[-1], cupp_85_G[-1],
                        facecolor=colour, alpha=0.35, label=label)
        ax.plot(xpts, [cmid_85_G[-1], cmid_85_G[-1]], linewidth=2.0,
                color=colour)

    plot_zeroline(ax, xlim)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_xticklabels(ax_labels, rotation=90)
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('left')
    plt.ylim(ylim)
    plt.ylabel('Sea Level Change (m)')
    plt.title(f'Global sea level - {scenario_string(rcp_str, -999)}')

    # -------------------------------------------------------------------------
    # Second figure, regional sea level projections and uncertainties under
    # RCP 8.5 and component parts
    bx = fig.add_subplot(1, 2, 2)
    bx_labels = []
    for cc, comp in enumerate(['sum', 'ocean', 'antnet', 'greennet', 'glacier',
                               'landwater', 'gia']):
        # Get the components of local sea level projections
        clow_85_R, cmid_85_R, cupp_85_R = extract_comp_sl(r_df_rcp,
                                                          percentiles, comp)

        colour = comp_colours[comp]
        label = comp_labels[comp]
        bx_labels.append(label)
        xpts = [cc - 0.4, cc + 0.4]

        bx.fill_between(xpts, clow_85_R[-1], cupp_85_R[-1], facecolor=colour,
                        alpha=0.35, label=label)
        bx.plot(xpts, [cmid_85_R[-1], cmid_85_R[-1]], linewidth=2.0,
                color=colour)

    plot_zeroline(bx, xlim)
    bx.set_xticks([0, 1, 2, 3, 4, 5, 6])
    bx.set_xticklabels(bx_labels, rotation=90)
    bx.yaxis.set_ticks_position('both')
    bx.yaxis.set_label_position('left')
    plt.ylim(ylim)
    plt.title(f'Local sea level - {location_string(site_name)} - '
              f'{scenario_string(scenarios[rcp_count], -999)}')

    # Save the figure
    fig.tight_layout()
    outfile = f'{fig_dir}07_{site_name}_rcp85.png'
    plt.savefig(outfile, dpi=200, format='png')
    plt.close()


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


def read_G_R_sl_projections(site_name, scenarios):
    """
    Reads in the global and regional sea level projections calculated from
    the CMIP projections (in metres). These data are relative to a baseline
    of 1981-2010.
    :param site_name: site location
    :param scenarios: emission scenarios
    :return: DataFrame of global and regional sea level projections
    """
    print('running function read_regional_sea_level_projections')

    # Read in the global and regional sea level projections
    in_slddir = read_dir()[4]
    loc_abbrev = abbreviate_location_name(site_name)

    G_df_list = []
    R_df_list = []
    for sce in scenarios:
        G_filename = '{}{}_{}_projection_{}_global.csv'.format(
            in_slddir, loc_abbrev, sce, settings["projection_end_year"])
        R_filename = '{}{}_{}_projection_{}_regional.csv'.format(
            in_slddir, loc_abbrev, sce, settings["projection_end_year"])
        try:
            G_df = pd.read_csv(G_filename, header=0,
                               index_col=['year', 'percentile'])
            G_df.rename(columns={'exp': 'ocean', 'GIA': 'gia'}, inplace=True)
            R_df = pd.read_csv(R_filename, header=0,
                               index_col=['year', 'percentile'])
            R_df.rename(columns={'exp': 'ocean', 'GIA': 'gia'}, inplace=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                sce, '- scenario selected does not currently exist')

        G_df_list.append(G_df)
        R_df_list.append(R_df)

    return G_df_list, R_df_list


def read_IPCC_AR5_Levermann_proj(scenarios, refname='sum'):
    """
    Read in the IPCC AR5 + Levermann global sea level projections.
    :param scenarios: emission scenario
    :param refname: name of the component to plot e.g. "expansion",
    "antsmb", "sum"
    :return: lower, middle and upper time series of IPCC AR5 + Levermann
    global sea level projections
    """
    print('running function read_IPCC_AR5_Levermann_proj')

    # Directory of Monte Carlo time series for new projections
    mcdir = choose_montecarlo_dir()
    
    nyrs = np.arange(2007, settings["projection_end_year"] + 1).size

    ar5_low = []
    ar5_mid = []
    ar5_upp = []

    for sce in scenarios:
        reflow = iris.load_cube(
            os.path.join(mcdir, sce + '_' + refname + 'lower.nc'))[:nyrs].data
        refmid = iris.load_cube(
            os.path.join(mcdir, sce + '_' + refname + 'mid.nc'))[:nyrs].data
        refupp = iris.load_cube(
            os.path.join(mcdir, sce + '_' + refname + 'upper.nc'))[:nyrs].data

        ar5_low.append(reflow)
        ar5_mid.append(refmid)
        ar5_upp.append(refupp)

    return ar5_low, ar5_mid, ar5_upp


def read_PSMSL_tide_gauge_obs(root_dir, data_source, data_type, region,
                              df_site, site_name, base_fdir):
    """
    Read annual mean sea level observations from PSMSL.
    :param root_dir: base directory
    :param data_source: data source of tide gauge data
    :param data_type: data type of tide gauge data
    :param region: user specified region (for folder structure)
    :param df_site: DataFrame of site metadata
    :param site_name: site locaiton
    :param base_fdir: figure directory
    :return: annual mean sea level data and relevant metadata on missing /
    flagged data
    """
    # Determine Station ID - indicator for .rlr file
    station_id = df_site.loc[site_name, 'Station ID']

    if station_id == 'NA':
        print(f'{site_name} is not a recognised tide gauge on PSMSLs ' +
              'station list')
        latitude = df_site.loc[site_name, 'Latitude']
        longitude = df_site.loc[site_name, 'Longitude']
        station_id, tg_name = find_nearest_station_id(
            root_dir, data_source, data_type, region, latitude, longitude)
    else:
        tg_name = location_string(site_name)

    baseline_years = [1981, 2010]
    min_valid_fraction = 0.5

    # Read in the annual mean sea levels, downloaded from PSMSL
    tg_years, tg_amsl, _, tg_flag_ex = \
        tgl.read_rlr_annual_mean_sea_level(station_id)
    non_missing = ~np.isnan(tg_amsl)

    # Find the years with 'flagged' data, flagged for attention '001' or MTL
    # in MSL time series '010'
    flag = (tg_flag_ex >= min_valid_fraction)

    nflag = sum(flag)

    # Calculate the baseline sea level and difference between the observed
    # period and the baseline.
    # Latter will be zero if sufficent observations lie within the baseline
    loc_abbrev = abbreviate_location_name(site_name)
    baseline_sl, _ = tgl.calc_baseline_sl(root_dir, region, loc_abbrev,
                                          tg_years, tg_amsl, baseline_years)

    # Adjust the tide gauge data to be sea levels relative to the baseline,
    # to match the regional projections
    tg_amsl = tg_amsl - baseline_sl

    # Convert the tide gauge sea levels from mm to metres.
    tg_amsl /= 1000.0

    return tg_name, nflag, flag, tg_years, non_missing, tg_amsl


def main():
    """
    Read in the global and regional sea level projections to create several
    graphs.
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

    # Create the output sea level projections figure directory to use across
    # multiple functions
    sealev_fdir = read_dir()[5]
    makefolder(sealev_fdir)

    rcp_scenarios = ['rcp26', 'rcp45', 'rcp85']
    for df_loc in df_site_data.index.values:
        # Read global and regional sea level projections calculated previously
        g_df_list, r_df_list = read_G_R_sl_projections(df_loc, rcp_scenarios)

        # Read in the IPCC AR5 + Levermann global sea level projections
        ar5_low, ar5_mid, ar5_upp = read_IPCC_AR5_Levermann_proj(rcp_scenarios)

        # Read annual mean sea level observations from PSMSL
        tg_name, nflag, flag, tg_years, non_missing, tg_amsl = \
            read_PSMSL_tide_gauge_obs(settings["baseoutdir"], settings[
                "tidegaugeinfo"]["source"], settings["tidegaugeinfo"][
                "datafq"], settings["siteinfo"]["region"], df_site_data,
                                      df_loc, sealev_fdir)
    # -------------------------------------------------------------------------
    # Plotting section - comment functions in and out depending on which graphs
    # you wish to save
        # Figure 1
        # Subplot 1 - Local sea level projections for all RCPs - sum total
        # Subplot 2 - Local sea level projections for RCP8.5 - total and
        # component parts
        plot_figure_one(r_df_list, df_loc, rcp_scenarios, sealev_fdir)

        # Figure 2
        # Local sea level projections for specified RCPs - sum total,
        # and annual mean tide gauge observations
        plot_figure_two(r_df_list, tg_name, nflag, flag, tg_years,
                        non_missing, tg_amsl, df_loc, rcp_scenarios,
                        sealev_fdir)

        # Figure 3 (one output per RCP)
        # Subplot 1 - Comparison of IPCC AR5 + Levermann global projections
        # and global projections developed here
        # Subplot 2 - Comparison of global projections and local sea level
        # projections
        plot_figure_three(g_df_list, r_df_list, ar5_low, ar5_mid, ar5_upp,
                          df_loc, rcp_scenarios, sealev_fdir)

        # Figure 4
        # Local sea level projections for all RCPs - sum total
        # Same style as Figure 3.1.4 of UKCP18 Marine Report (pg 16)
        plot_figure_four(r_df_list, df_loc, rcp_scenarios, sealev_fdir)

        # Figure 5
        # Comparison of global mean sea level projections calculated by the
        # tool and local sea level projections - sum total
        # Also shown local sea level projections - component parts
        # Subplot 1, 2, 3 - RCP2.6, RCP4.5, RCP8.5 respectively
        plot_figure_five(g_df_list, r_df_list, df_loc, rcp_scenarios,
                         sealev_fdir)

        # Figure 6
        # Comparison of IPCC AR5 + Levermann projections and local sea level
        # projections - sum total
        # Also shown local sea level projections - component parts
        # Subplot 1, 2, 3 - RCP2.6, RCP4.5, RCP8.5 respectively
        if settings["projection_end_year"] <= 2100: 
            plot_figure_six(r_df_list, ar5_low, ar5_mid, ar5_upp, df_loc,
                            rcp_scenarios, sealev_fdir)

        # Figure 7
        # Subplot 1 - Global sea level projections for RCP8.5 - total and
        # component parts, including uncertainty range
        # Subplot 2 - Local sea level projections for RCP8.5 - total and
        # component parts, including uncertainty range
        # Same style as Figure 4 Howard and Palmer (2019)
        plot_figure_seven(g_df_list, r_df_list, df_loc, rcp_scenarios,
                          sealev_fdir)


if __name__ == '__main__':
    main()
