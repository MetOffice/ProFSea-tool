"""
Copyright (c) 2023, Met Office
All rights reserved.
"""

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
import numpy as np
import iris
import iris.plot as iplt


def block(cube, **kwargs):
    """
    Draw a block color plot for a given cube. Invokes contourf() with
    pcolormesh=True.
    :param cube: iris.cube.Cube
    :param kwargs: See contourf() for other valid keyword arguments
    :return: a block color plot for a given cube
    """

    return contourf(cube, pcolormesh=True, **kwargs)


def contourf(cube, anom=False, subplot=111, cent_lon=0, land=True,
             coast=True, title=None, region=None, plotcbar=True,
             charsize=12, cmin=None, cmax=None, nlevels=11, levels=None,
             cbarlabel=None, cbar_orien='horizontal', pcolormesh=False,
             reproject=False, nx=400, ny=200, proj=None, xticks=None,
             yticks=None, cmap=None, **kwargs):
    """
    Draw a filled contour plot for a given cube.
    :param cube: iris.cube.Cube
    :param anom: boolean, use blue/red colormap for anomalies
        (def=False)
    :param subplot: 3-digit integer specifying subplot position
        (def=111)
    :param cent_lon: central longitude of the map projection
        (def=0)
    :param land: boolean, if True, filled land drawn
        (def=True)
    :param coast: boolean, if True, coastlines drawn
        (def=True)
    :param title: map title
        (def=None)
    :param region: 4-element list [E,S,W,N] defining sub-region to plot
        (def=None)
    :param plotcbar: boolean, if True, colorbar plotted
        (def=True)
    :param charsize: font size to use
        (def=12)
    :param cmin: minimum value for colormap/bar
        (def=None)
    :param cmax: maximum value for colormap/bar
        (def=None)
    :param nlevels: set number of bands within colormap
        (def=11)
    :param levels: set minimum, maximum and number of bands within colormap
        (def=None)
    :param cbarlabel: colorbar label
        (def=None)
    :param cbar_orien: orientation for colorbar
        (def='horizontal')
    :param pcolormesh: boolean:, if True, iris.plot.pcolormesh() used for
    plotting
        (def=False)
    :param reproject: boolean, if True, reproject cube to PlateCarree using a
    specified number of sample points (nx & ny) before plotting. Used for
    fields with 2D latitude/longitude fields
        (def=False)
    :param nx: number of sample points in x-direction used for cube
    re-projection
        (def=400)
    :param ny: number of sample points in y-direction used for cube
    re-projection
        (def=200)
    :param proj: If not defined, use ccrs.PlateCarree()
        (def=None)
    :param xticks: plot xticks on colormap
        (def=None)
    :param yticks: plot yticks on colormap
        (def=None)
    :param cmap: set colormap to blue/red
        (def=None)
    :param kwargs: defined in iplt.pcolormesh() and iplt.contourf()
    :return: filled contour plot
    """
    # Map projection
    if reproject:
        cube, extent = iris.analysis.cartography.project(
            cube, ccrs.PlateCarree(), nx=nx, ny=ny)

    if proj is not None:
        map_proj = proj
    else:
        map_proj = ccrs.PlateCarree(central_longitude=cent_lon)

    # Color map
    if cmap is None:
        cmap = plt.cm.RdBu_r if anom else plt.cm.jet

    # Create subplot, if specified.
    if isinstance(subplot, list):
        if len(subplot) != 3:
            raise ValueError('"subplot" needs to be defined as a'
                             ' 3-element list or a 3-digit integer')

        ax = plt.subplot(subplot[0], subplot[1], subplot[2],
                         projection=map_proj)
    else:
        ax = plt.subplot(subplot, projection=map_proj)
    ax.set_global()

    # Color levels
    if levels is None:
        if cmin is None:
            cmin = cube.data.min()
        if cmax is None:
            cmax = cube.data.max()
        levels = np.linspace(cmin, cmax, nlevels)

    # Plot using pcolormesh or contourf
    if pcolormesh:
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        plot_handle = iplt.pcolormesh(cube, cmap=cmap, norm=norm,
                                      rasterized=True, **kwargs)
    else:
        plot_handle = iplt.contourf(cube, cmap=cmap, levels=levels,
                                    rasterized=True, **kwargs)

    # Land and coastlines
    if land:
        ax.add_feature(cartopy.feature.LAND)
    if coast:
        ax.coastlines(resolution='50m')

    # Extract region
    if region:
        if len(region) != 4:
            raise ValueError('"Region" needs to be defined as a'
                             ' 4-element list')
        ax.set_ylim([region[1], region[3]])
        x0 = (cent_lon - region[0]) * -1
        x1 = (region[2] - cent_lon)
        ax.set_xlim([x0, x1])

    # Color bar
    if title is None:
        title = cube.name()
    plt.title(title, fontsize=charsize)

    # Color bar
    if plotcbar:
        cbar = plt.colorbar(plot_handle, orientation=cbar_orien)
        cbar.ax.tick_params(labelsize=charsize * 0.85)
        if cbarlabel is None:
            cbarlabel = cube.units

        cbar.set_label(cbarlabel, fontsize=charsize * 0.85)

    #  Tick labels and grid lines
    if (xticks is not None) or (yticks is not None):
        ax = plt.gca()

        # First do labels
        gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray',
                          linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlines = False
        gl.ylines = False

        if xticks is not None:
            gl.xlocator = mticker.FixedLocator(xticks)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.xlabel_style = {'size': charsize * 0.8}

        else:
            gl.xlocator = mticker.FixedLocator([-1e6])

        if yticks is not None:
            gl.ylocator = mticker.FixedLocator(yticks)
            gl.yformatter = LATITUDE_FORMATTER
            gl.ylabel_style = {'size': charsize * 0.8}

        else:
            gl.ylocator = mticker.FixedLocator([-1e6])

        # Now do grid lines
        if xticks is not None:
            # xticks += xticks
            gl_x = ax.gridlines(linewidth=1, color='gray',
                                linestyle=':')
            gl_x.xlocator = mticker.FixedLocator(xticks)
            gl_x.ylines = False

        if yticks is not None:
            gl_y = ax.gridlines(linewidth=1, color='gray',
                                linestyle=':')
            gl_y.ylocator = mticker.FixedLocator(yticks)
            gl_y.xlines = False

    return ax
