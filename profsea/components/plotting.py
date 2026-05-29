"""
Copyright (c) 2023, Met Office
All rights reserved.
"""
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from profsea.plotting_libraries import (
    calc_xlim, calc_ylim, plot_zeroline, multi_index_values,
    extract_comp_sl, plot_tg_data, location_string, ukcp18_labels,
    get_emulator_colors)


def plot_slr(r_df_list, tg_name, nflag, flag, tg_years, 
             non_missing, tg_amsl, site_name, scenarios, fig_dir):
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

    fig = plt.figure(figsize=(5, 4.5))
    matplotlib.rcParams['font.size'] = 10
    
    scenario_colours = get_emulator_colors(len(scenarios))

    # Draw regional sea level projections and uncertainties under each scenario
    ax = fig.add_subplot(1, 1, 1)

    for i, scen in enumerate(scenarios):
        scen_col = next(scenario_colours)
        # Get the sums of all components of local sea level projections
        r_df = r_df_list[i]
        rlow, rmid, rupp = extract_comp_sl(r_df, percentiles, 'sum')
        ax.fill_between(years, rlow, rupp, alpha=0.3,
                        color=scen_col, linewidth=0)
        ax.plot(years, rmid, color=scen_col, label=scen)

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
    ffile = f'{fig_dir}emulator_scenarios_SLR_{site_name}.png'
    plt.savefig(ffile, dpi=300, format='png')
    plt.close()


def plot_slr_components(g_df_list, r_df_list, site_name, scenarios, fig_dir):
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

    # Calculate sensible x-axis range and y-axis range
    xlim = ([-0.4, 7.4])
    r_ylim = calc_ylim('proj', [], r_df_list)
    g_ylim = calc_ylim('proj', [], g_df_list)
    ylim = [min(r_ylim[0], g_ylim[0]), max(r_ylim[1], g_ylim[1])]

    # -------------------------------------------------------------------------
    # Global sea level projections, uncertainties and component parts
    fig = plt.figure(figsize=(7.68*len(scenarios), 4.5*len(scenarios)))
    matplotlib.rcParams['font.size'] = 9

    for i, scen in enumerate(scenarios):
        scenario_colors = get_emulator_colors(len(scenarios), get_all=True)
        r_df = r_df_list[i]
        g_df = g_df_list[i]
        
        ax_labels = []
        ax = fig.add_subplot(len(scenarios), 2, (i*2)+1, label=f'Scenario_{i + 1}_Plot_1')
        for cc, comp in enumerate(['sum', 'ocean', 'antnet', 'greennet', 'glacier',
                                'landwater']):
            # Get the components of regional sea level projections
            clow_G, cmid_G, cupp_G = extract_comp_sl(g_df,
                                                            percentiles, comp)

            colour = next(scenario_colors)
            label = ukcp18_labels()[comp]
            
            ax_labels.append(label)
            xpts = [cc - 0.4, cc + 0.4]

            ax.fill_between(xpts, clow_G[-1], cupp_G[-1],
                            facecolor=colour, alpha=0.85, label=label)
            ax.plot(xpts, [cmid_G[-1], cmid_G[-1]], linewidth=2.0,
                    color=colour)

        plot_zeroline(ax, xlim)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels(ax_labels, rotation=90)
        ax.yaxis.set_ticks_position('both')
        ax.yaxis.set_label_position('left')
        ax.set_ylim(ylim)
        ax.set_ylabel('Sea Level Change (m)')
        ax.set_title(f'Global sea level - {scen}')

        # -------------------------------------------------------------------------
        # Second figure, regional sea level projections and uncertainties under
        # RCP 8.5 and component parts
        
        scenario_colors = get_emulator_colors(len(scenarios), get_all=True)
        bx = fig.add_subplot(len(scenarios), 2, (i*2)+2, label=f'Scenario_{i + 1}_Plot_2')
        bx_labels = []
        for cc, comp in enumerate(['sum', 'ocean', 'antnet', 'greennet', 'glacier',
                                'landwater', 'gia']):
            # Get the components of local sea level projections
            clow_R, cmid_R, cupp_R = extract_comp_sl(r_df,
                                                            percentiles, comp)

            colour = next(scenario_colors)
            label = ukcp18_labels()[comp]
            
            bx_labels.append(label)
            xpts = [cc - 0.4, cc + 0.4]

            bx.fill_between(xpts, clow_R[-1], cupp_R[-1], facecolor=colour,
                            alpha=0.85, label=label)
            bx.plot(xpts, [cmid_R[-1], cmid_R[-1]], linewidth=2.0,
                    color=colour)

        plot_zeroline(bx, xlim)
        bx.set_xticks([0, 1, 2, 3, 4, 5, 6])
        bx.set_xticklabels(bx_labels, rotation=90)
        bx.yaxis.set_ticks_position('both')
        bx.yaxis.set_label_position('left')
        bx.set_ylim(ylim)
        bx.set_title(f'Local sea level - {location_string(site_name)} - {scen}')

    # Save the figure
    fig.tight_layout()
    outfile = f'{fig_dir}emulator_components_{site_name}.png'
    plt.savefig(outfile, dpi=300, format='png')
    plt.close()