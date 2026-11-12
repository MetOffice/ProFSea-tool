import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle

from profsea.components.core.global_model import Global
from profsea.components.core.spatial_model import Spatial
from profsea.components.core.local_model import Local

from profsea.components.global_ import (
    AntarcticaDynAR5,
    AntarcticaSMBAR5,
    Glacier,
    GreenlandDynAR5,
    GreenlandSMBAR5,
    LandwaterAR5,
    ThermalExpansion,
)
from profsea.components.spatial import GIA, Fingerprint, SterodynamicCMIP6

# other python modules
from profsea.utils import sample_members_2D
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import yaml
import json


import numpy as np

#---------------------------------------------------------------------
def validate_input_ts(mean_ts, std_ts, var):

    # Validation
    if mean_ts.ndim != 1:
        raise ValueError(f"{var} mean_ts must be 1D")

    if std_ts.ndim != 1:
        raise ValueError(f"{var} std_ts must be 1D")

    if len(mean_ts) != len(std_ts):
        raise ValueError(f"{var} mean_ts and std_ts must have the same length")

    if np.any(std_ts < 0):
        raise ValueError(f"{var} std_ts contains negative values")

    if not np.all(np.isfinite(mean_ts)):
        raise ValueError(f"{var} mean_ts contains non-finite values")

    if not np.all(np.isfinite(std_ts)):
        raise ValueError(f"{var} std_ts contains non-finite values")
    
    return



#---------------------------------------------------------------------
def generate_mc_ensembles(tas_mean_ts, tas_std_ts, zostoga_mean_ts, zostoga_std_ts, n_members=450, seed=42):
    """
    Generate Monte Carlo time series using Jonathan Gregory's method:

        Q_i(t) = Q_mean(t) + r_i * Q_std(t)

    where r_i is constant in time for each member.

    Parameters
    ----------
    VAR_mean_ts : array-like, shape (ntime,)
        Ensemble mean time series for VAR either tas or zostoga.
    VAR_std_ts : array-like, shape (ntime,)
        Ensemble standard deviation time series for VAR either tas or zostoga.
    n_members : int
        Number of Monte Carlo members (default 450)
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tas_ensemble : ndarray, shape (n_members, ntime)
    zostoga_ensemble : ndarray, shape (n_members, ntime)
    """

    tas_mean_ts = np.asarray(tas_mean_ts)
    tas_std_ts = np.asarray(tas_std_ts)
    zostoga_mean_ts = np.asarray(zostoga_mean_ts)
    zostoga_std_ts = np.asarray(zostoga_std_ts)

    # A few checks on the input time series
    validate_input_ts(mean_ts=tas_mean_ts, std_ts=tas_std_ts, var='tas')
    validate_input_ts(mean_ts=zostoga_mean_ts, std_ts=zostoga_std_ts, var='zostoga')

    if len(tas_mean_ts) != len(zostoga_mean_ts):
        raise ValueError("tas and zostoga time series have different lengths")
    
    # Reproducible RNG
    rng = np.random.default_rng(seed)

    # One random number per ensemble member
    r = rng.standard_normal(n_members)

    # Broadcast into (n_members, ntime)
    tas_ensemble = tas_mean_ts[None, :] + r[:, None] * tas_std_ts[None, :]
    zostoga_ensemble = zostoga_mean_ts[None, :] + r[:, None] * zostoga_std_ts[None, :]


    # Check for perfect correlation (the same random number variates were used for tas and zostoga)
    tas_z = (tas_ensemble - tas_mean_ts[None, :]) / tas_std_ts[None, :]
    zostoga_z = (zostoga_ensemble - zostoga_mean_ts[None, :]) / zostoga_std_ts[None, :]
    assert np.allclose(tas_z, zostoga_z)

    
    return tas_ensemble, zostoga_ensemble

#==========================================================================================================
def main():

    fig_path = "/home/users/lesley.allison/sealevel/profsea/profsea-v3/figs/"

    fig_cols = {
        "expansion": 'r',
        "greenland_dyn": 'g',
        "greenland_smb": 'b',
        "landwater": 'y',
        "antarctica_dyn": 'aquamarine',
        "antarctica_smb": 'm',
        "glacier": 'c',
        "total_gmslr": 'k'
    }

    ##### Read in the mean CMIP5 time series for tas and zostoga ###

    f_tas26 = "/home/users/lesley.allison/old-home/ocean/sealevel/montecarlo/ar5_input/rcp26_temperature_mean.nc"
    ds_tas26 = xr.open_dataset(f_tas26)
    print(ds_tas26)
    print("\nVariables:")
    print(list(ds_tas26.data_vars))
    for var in ds_tas26.data_vars:
        print(var, ds_tas26[var].shape)

        
    f_tas85 = "/home/users/lesley.allison/old-home/ocean/sealevel/montecarlo/ar5_input/rcp85_temperature_mean.nc"
    ds_tas85 = xr.open_dataset(f_tas85)
    print(ds_tas85)
    print("\nVariables:")
    print(list(ds_tas85.data_vars))
    for var in ds_tas85.data_vars:
        print(var, ds_tas85[var].shape)
        
    f_exp26 = "/home/users/lesley.allison/old-home/ocean/sealevel/montecarlo/ar5_input/rcp26_expansion_mean.nc"
    ds_exp26 = xr.open_dataset(f_exp26)
    print(ds_exp26)
    print("\nVariables:")
    print(list(ds_exp26.data_vars))
    for var in ds_exp26.data_vars:
        print(var, ds_exp26[var].shape)
        
    f_exp85 = "/home/users/lesley.allison/old-home/ocean/sealevel/montecarlo/ar5_input/rcp85_expansion_mean.nc"
    ds_exp85 = xr.open_dataset(f_exp85)
    print(ds_exp85)
    print("\nVariables:")
    print(list(ds_exp85.data_vars))
    for var in ds_exp85.data_vars:
        print(var, ds_exp85[var].shape)


    ##### Read in the CMIP5 STANDARD DEVIATION time series for tas and zostoga ###

    f_tas26_sd = "/home/users/lesley.allison/old-home/ocean/sealevel/montecarlo/ar5_input/rcp26_temperature_sd.nc"
    ds_tas26_sd = xr.open_dataset(f_tas26_sd)
    print(ds_tas26_sd)
    print("\nVariables:")
    print(list(ds_tas26_sd.data_vars))
    for var in ds_tas26_sd.data_vars:
        print(var, ds_tas26_sd[var].shape)

    
    f_tas85_sd = "/home/users/lesley.allison/old-home/ocean/sealevel/montecarlo/ar5_input/rcp85_temperature_sd.nc"
    ds_tas85_sd = xr.open_dataset(f_tas85_sd)
    print(ds_tas85_sd)
    print("\nVariables:")
    print(list(ds_tas85_sd.data_vars))
    for var in ds_tas85_sd.data_vars:
        print(var, ds_tas85_sd[var].shape)
        
    f_exp26_sd = "/home/users/lesley.allison/old-home/ocean/sealevel/montecarlo/ar5_input/rcp26_expansion_sd.nc"
    ds_exp26_sd = xr.open_dataset(f_exp26_sd)
    print(ds_exp26_sd)
    print("\nVariables:")
    print(list(ds_exp26_sd.data_vars))
    for var in ds_exp26_sd.data_vars:
        print(var, ds_exp26_sd[var].shape)
        
    f_exp85_sd = "/home/users/lesley.allison/old-home/ocean/sealevel/montecarlo/ar5_input/rcp85_expansion_sd.nc"
    ds_exp85_sd = xr.open_dataset(f_exp85_sd)
    print(ds_exp85_sd)
    print("\nVariables:")
    print(list(ds_exp85_sd.data_vars))
    for var in ds_exp85_sd.data_vars:
        print(var, ds_exp85_sd[var].shape)



        

    # Make np arrays
    # NOTE WE ARE LOSING METADATA HERE --> Consider revising methods?
    # CMIP5 ensemble mean
    rcp26_tas_mean_ts = ds_tas26["air_temperature"]
    rcp26_zostoga_mean_ts = ds_exp26["global_average_thermosteric_sea_level_change"]
    rcp85_tas_mean_ts = ds_tas85["air_temperature"]
    rcp85_zostoga_mean_ts = ds_exp85["global_average_thermosteric_sea_level_change"]
    # CMIP5 ensemble standard deviation
    rcp26_tas_std_ts = ds_tas26_sd["air_temperature"]
    rcp26_zostoga_std_ts = ds_exp26_sd["global_average_thermosteric_sea_level_change"]
    rcp85_tas_std_ts = ds_tas85_sd["air_temperature"]
    rcp85_zostoga_std_ts = ds_exp85_sd["global_average_thermosteric_sea_level_change"]

        
    #-----------------------------------------------------------------------------------------------
    # CREATE THE CLIMATE MONTE CARLO USING TIME SERIES OF ENSEMBLE MEAN AND STANDARD DEVIATION
    n_members = 450
    seed=42
    
    rcp26_tas_ensemble, rcp26_zostoga_ensemble = generate_mc_ensembles(rcp26_tas_mean_ts, rcp26_tas_std_ts, rcp26_zostoga_mean_ts, rcp26_zostoga_std_ts, n_members=n_members, seed=seed)
    rcp85_tas_ensemble, rcp85_zostoga_ensemble = generate_mc_ensembles(rcp85_tas_mean_ts, rcp85_tas_std_ts, rcp85_zostoga_mean_ts, rcp85_zostoga_std_ts, n_members=n_members, seed=seed)

    #-----------------------------------------------------------------------------------------------

    # Plot multi-model mean tas and zostoga with the 450-member Monte Carlo ensemble that has just been created
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)


    # ax1: TAS
    ds_tas26.air_temperature.plot(ax=ax1, color="tab:blue", label="RCP2.6")
    ds_tas85.air_temperature.plot(ax=ax1, color="tab:red", label="RCP8.5")
    ax1.plot(ds_tas26.time.values, rcp26_tas_ensemble.T, color='tab:blue', alpha=0.02)  # 1000 faint lines
    ax1.plot(ds_tas85.time.values, rcp85_tas_ensemble.T, color='tab:red', alpha=0.02)  # 1000 faint lines

    # ax2: ZOSTOGA
    ds_exp26.global_average_thermosteric_sea_level_change.plot(ax=ax2, color="tab:blue", label="RCP2.6")
    ds_exp85.global_average_thermosteric_sea_level_change.plot(ax=ax2, color="tab:red", label="RCP8.5")
    ax2.plot(ds_exp26.time.values, rcp26_zostoga_ensemble.T, color='tab:blue', alpha=0.02)  # 1000 faint lines
    ax2.plot(ds_exp85.time.values, rcp85_zostoga_ensemble.T, color='tab:red', alpha=0.02)  # 1000 faint lines


    
    ax1.set_title("Global surface air temperature change")
    ax2.set_title("Global thermosteric sea-level change")

    ax1.legend()
    ax2.legend()
    
    fig.savefig(
        fig_path+"CMIP5_tas_zostoga_from_mean_sd.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    

    

    
#    # Plot multi-model mean tas and zostoga
#    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)

#    ds_tas26.air_temperature.plot(ax=ax1, color="tab:blue", label="RCP2.6")
#    ds_tas85.air_temperature.plot(ax=ax1, color="tab:red", label="RCP8.5")
#    ds_exp26.global_average_thermosteric_sea_level_change.plot(ax=ax2, color="tab:blue", label="RCP2.6")
#    ds_exp85.global_average_thermosteric_sea_level_change.plot(ax=ax2, color="tab:red", label="RCP8.5")

#    ax1.set_title("Global surface air temperature change")
#    ax2.set_title("Global thermosteric sea-level change")

#    ax1.legend()
#    ax2.legend()
    
#    fig.savefig(
#        "/home/users/lesley.allison/sealevel/profsea/profsea-v3/figs/CMIP5_multi_model_mean_tas_zostoga.png",
#        dpi=300,
#        bbox_inches="tight"
#    )
#    plt.show()




    ####-------------- GMSL projection using the Monte Carlo we have just created from CMIP5 MM mean and stddev ----------------- ####

    ### Global projections first ###
    global_components = {
        "expansion": ThermalExpansion(rcp85_zostoga_ensemble, OHC_change=False),
        "greenland_dyn": GreenlandDynAR5(),
        "greenland_smb": GreenlandSMBAR5(),
        "landwater": LandwaterAR5(),
        "antarctica_dyn": AntarcticaDynAR5(),
        "antarctica_smb": AntarcticaSMBAR5(),
        "glacier": Glacier(glaciermip=False),
    }

 

    
    # Pass to the global model (Initialise the global model with the consituent components)
#    global_model = Global(components=global_components, end_yr=2301)  # creating an instance of the Global class
    global_model = Global(components=global_components, end_yr=2101)  # creating an instance of the Global class
    
    # Run the global model
    projections = global_model.run(
        scenario="rcp85",
        T_change=rcp85_tas_ensemble,
        member_seed=42,
    )

    
    # Sum the components for the global mean total
    global_model.sum_components(projections)
    gmslr = global_model.results["total_gmslr"]

#    # plot all members of gmslr
#    fig, ax = plt.subplots(figsize=(10, 6))
#    ax.plot(gmslr.T, color='k', alpha=0.02)  # 1000 faint lines
#    ax.set_xlabel('Time')
#    ax.set_ylabel('GMSLR')
#    ax.set_title('All ensemble members')
#    plt.show()


    #=========================================================
    # Read in JG's projections to compare

    jg_proj_dir = "/data/users/lesley.allison/project_sea_level_copy/sea_level_projections/JGregory/RCP/project_rcp_v6/"
    jg_rcp85_summid_file = jg_proj_dir + "rcp85_summid.nc"
    jg_rcp85_summid = xr.open_dataset(jg_rcp85_summid_file)
    print(jg_rcp85_summid)
    print("\nVariables:")
    print(list(jg_rcp85_summid.data_vars))
    for var in jg_rcp85_summid.data_vars:
        print(var, jg_rcp85_summid[var].shape)
    jg_years = np.array([t.year-1 for t in jg_rcp85_summid.time.values])
    #### NOTE I HAVE SUBTRACTED 1 FROM JG YEARS TO GET THEM TO MATCH --> NEEDS LOOKING INTO

    jg_rcp85_sumlower_file = jg_proj_dir + "rcp85_sumlower.nc"
    jg_rcp85_sumlower = xr.open_dataset(jg_rcp85_sumlower_file)
    jg_lower_years = np.array([t.year-1 for t in jg_rcp85_sumlower.time.values])
    #### NOTE I HAVE SUBTRACTED 1 FROM JG YEARS TO GET THEM TO MATCH --> NEEDS LOOKING INTO

    jg_rcp85_sumupper_file = jg_proj_dir + "rcp85_sumupper.nc"
    jg_rcp85_sumupper = xr.open_dataset(jg_rcp85_sumupper_file)
    jg_upper_years = np.array([t.year-1 for t in jg_rcp85_sumupper.time.values])
    #### NOTE I HAVE SUBTRACTED 1 FROM JG YEARS TO GET THEM TO MATCH --> NEEDS LOOKING INTO


    #=======================
    # READ IN JG COMPONENTS

    jg_names = {
        "expansion": "exp",
        "greenland_dyn": "greendyn",
        "greenland_smb": "greensmb",
        "landwater": "landwater",
        "antarctica_dyn": "antdyn",
        "antarctica_smb": "antsmb",
        "glacier": "glacier",
        }

    jg_comps = {}
    
    scenario='rcp85'
    
    for comp in global_model.components:
        print(comp)
        jg_comp_file = jg_proj_dir + scenario + '_' + jg_names[comp] + '.nc'
        print(jg_comp_file)
        jg_comps[comp] = xr.open_dataset(jg_comp_file)

    

    #=========================================================

        
    ### Plot GMSLR (ax) and components (ax2) ###
    fig = plt.figure(figsize=(10, 4), layout="constrained")

    ax = fig.add_subplot(121)

    # Global projections
#    yrs = np.arange(2006, 2301)
    yrs = np.arange(2006, 2100)
    ax.set_ylim(-0.05,1)
    ax.plot(
        yrs,
        np.median(gmslr, axis=0),
        label="Global Projection",
        color="royalblue",
    )
    # fill between 1 and 4 members
    ax.fill_between(
        yrs,
        np.percentile(gmslr, 17, axis=0),
        np.percentile(gmslr, 83, axis=0),
        alpha=0.3,
        color="skyblue",
    )
    ax.plot(yrs, np.percentile(gmslr, 5, axis=0), color="royalblue", linestyle='--')
    ax.plot(yrs, np.percentile(gmslr, 95, axis=0), color="royalblue", linestyle='--')

    
    # Plot JG's projections for comparison
    ax.plot(jg_years, jg_rcp85_summid.global_average_sea_level_change.data, color="m", label="JG")
    ax.plot(jg_lower_years, jg_rcp85_sumlower.global_average_sea_level_change.data, color="m", linestyle="--")
    ax.plot(jg_upper_years, jg_rcp85_sumupper.global_average_sea_level_change.data, color="m", linestyle="--")
#    ax.fill_between(
#        jg_years,
#        jg_rcp85_sumlower.global_average_sea_level_change.data,
#        jg_rcp85_sumupper.global_average_sea_level_change.data,
#        alpha=0.3,
#        color="m",
#    )

    
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Global Mean Sea Level Rise (m)")

    ax.legend()

    ax2 = fig.add_subplot(122)
    yrs = np.arange(2006, 2100)
    ax2.set_ylim(-0.05,1)
    #ax2.plot(yrs, np.median(gmslr, axis=0), label="Global Projection", color="royalblue")
    for comp in global_model.results.keys():
        print(comp)
        ax2.plot(yrs, np.median(global_model.results[comp],axis=0), label=comp, color=fig_cols[comp])
        ax2.plot(yrs, np.percentile(global_model.results[comp], 17 ,axis=0), color=fig_cols[comp], linestyle='--')
        ax2.plot(yrs, np.percentile(global_model.results[comp], 83 ,axis=0), color=fig_cols[comp], linestyle='--')
    ax2.set_xlabel("Year", fontsize=14)
    ax2.set_ylabel("Global Mean Sea Level Rise (m)")
    ax2.legend()

    fig.savefig(fig_path+"GMSLR_using_MC_from_CMIP5_mean_stdev_tas_zostoga_with_compspread_glaciermipfalse.png", dpi=300,bbox_inches="tight")
    
    plt.show()





    #===================================================================================
    # Comparison of individual components
    #===================================================================================

    # Create figure and axes
    fig, axs = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(14, 6),
        constrained_layout=True
    )

    # Loop through all panels
    for ax, comp in zip(axs.flat, global_model.components):
        ax.plot(yrs, np.median(global_model.results[comp],axis=0), label=comp, color=fig_cols[comp])
        ax.plot(yrs, np.percentile(global_model.results[comp], 5, axis=0), label=comp, color=fig_cols[comp], linestyle='--')
        ax.plot(yrs, np.percentile(global_model.results[comp], 95, axis=0), label=comp, color=fig_cols[comp], linestyle='--')

        jg_comp = jg_comps[comp]["global_average_sea_level_change"]
        jg_yrs = np.array([t.year-1 for t in jg_comp.time.values])
        #### NOTE I HAVE SUBTRACTED 1 FROM JG YEARS TO GET THEM TO MATCH --> NEEDS LOOKING INTO
        ax.plot(jg_yrs, np.median(jg_comp.values, axis=1), label='JG_'+comp, color='k')
        ax.plot(jg_yrs, np.percentile(jg_comp.values, 5, axis=1), label='JG_'+comp, color='k', linestyle='--')
        ax.plot(jg_yrs, np.percentile(jg_comp.values, 95, axis=1), label='JG_'+comp, color='k', linestyle='--')
        ax.plot(jg_yrs+1, np.median(jg_comp.values, axis=1), label='JG_'+comp, color='k', linestyle=':')
        ax.plot(jg_yrs+1, np.percentile(jg_comp.values, 5, axis=1), label='JG_'+comp, color='k', linestyle=':')
        ax.plot(jg_yrs+1, np.percentile(jg_comp.values, 95, axis=1), label='JG_'+comp, color='k', linestyle=':')
        
    
        ax.set_title(comp)
        #ax.grid(True)

    # Last panel - total GMSLR
    ax=axs.flat[-1]
    ax.plot(yrs, np.median(gmslr, axis=0), color="royalblue", linestyle='-')
    ax.plot(yrs, np.percentile(gmslr, 5, axis=0), color="royalblue", linestyle='--')
    ax.plot(yrs, np.percentile(gmslr, 95, axis=0), color="royalblue", linestyle='--')

    ax.plot(jg_years, jg_rcp85_summid.global_average_sea_level_change.data, color="k", label="JG")
    ax.plot(jg_lower_years, jg_rcp85_sumlower.global_average_sea_level_change.data, color="k", linestyle="--")
    ax.plot(jg_upper_years, jg_rcp85_sumupper.global_average_sea_level_change.data, color="k", linestyle="--")
    ax.plot(jg_years+1, jg_rcp85_summid.global_average_sea_level_change.data, color="k", label="JG", linestyle=":")
    ax.plot(jg_lower_years+1, jg_rcp85_sumlower.global_average_sea_level_change.data, color="k", linestyle=":")
    ax.plot(jg_upper_years+1, jg_rcp85_sumupper.global_average_sea_level_change.data, color="k", linestyle=":")
 
    ax.set_title('Total')

    fig.savefig(fig_path+"GMSLR_using_MC_from_CMIP5_mean_stdev_tas_zostoga_with_components_glaciermipfalse.png", dpi=300,bbox_inches="tight")

    
    plt.show()


    
    ###------------------------------------------------------------------------------------------
    ### Now spatial projections ###
    spatial_components = {
        "sterodynamic": SterodynamicCMIP6(
            projections["expansion"],
        ),
        "greenland_dyn": Fingerprint(
            projections["greenland_dyn"], fingerprint_component="greendyn"
        ),
        "greenland_smb": Fingerprint(
            projections["greenland_smb"],
            fingerprint_component="greensmb",
        ),
        "landwater": Fingerprint(
            projections["landwater"],
            fingerprint_component="landwater",
        ),
        "antarctica_dyn": Fingerprint(
            projections["antarctica_dyn"],
            fingerprint_component="antdyn",
        ),
        "antarctica_smb": Fingerprint(
            projections["antarctica_smb"],
            fingerprint_component="antsmb",
        ),
        "glacier": Fingerprint(
            projections["glacier"],
            fingerprint_component="glacier",
        ),
        "gia": GIA(
            sample_spatial=True,
        ),
    }

    # Pass to the spatial model
    spatial_model = Spatial(components=spatial_components, num_local_members=4)
    # LA: I have updated the following line to assign the returned value from Spatial.run() to a variable (spatial_model_results)
    #     to make the output explicit and easier to inspect and pass to subsequent steps
    #     (spatial_model_results is equivalent to spatial_model.results but easier to follow and inspect)
    
    spatial_model_results = spatial_model.run(member_seed=42)   

    pdb.set_trace()
    
    spatial_model.sum_components(spatial_model.results)
    spatial_model.save_components(
        spatial_model.results, scenario_name="rcp85_test", output_format="zarr"
    )


    ### Plot example ###
    fig = plt.figure(figsize=(10, 4), layout="constrained")

    total_rsl = spatial_model.results["total_rsl"][2, -1, :, :]
    vmax = np.nanmax(np.abs(total_rsl))
    vmin = -vmax

 

    ax = fig.add_subplot(122, projection=ccrs.PlateCarree())
    ax.set_title("50th Percentile, Year 2100")
    ax.pcolormesh(
        spatial_model.grid_lons,
        spatial_model.grid_lats,
        total_rsl,
        transform=ccrs.PlateCarree(),
        cmap="PuOr_r",
        vmin=vmin,
        vmax=vmax,
    )
    ax.coastlines()
    fig.colorbar(
        mappable=ax.collections[0],
        label="Relative Sea Level (m)",
        orientation="horizontal",
        pad=0.02,
    )
    plt.show()



    locations = {
        "Aberdeen": (57.144, -2.080),
        "Holyhead": (53.314, -4.620),
        "Llandudno": (53.330, -3.830),
        "Land point": (-34.6037, -58.3816),
        "Durban": (-29.874, 31.051),
    }

    local_components = {
        "sterodynamic": SterodynamicCMIP6(
            projections["expansion"],  # passing in the global thermal expansion projection
        ),
        "greenland_smb": Fingerprint(
            projections["greenland_smb"], 
            fingerprint_component="greensmb"
        ),
        "greenland_dyn": Fingerprint(
            projections["greenland_dyn"], 
            fingerprint_component="greendyn"
        ),
        "landwater": Fingerprint(
            projections["landwater"],
            fingerprint_component="landwater",
        ),
        "antarctica_dyn": Fingerprint(
            projections["antarctica_dyn"],
            fingerprint_component="antdyn",
        ),
        "antarctica_smb": Fingerprint(
            projections["antarctica_smb"],
            fingerprint_component="antsmb",
        ),
        "glacier": Fingerprint(
            projections["glacier"],
            fingerprint_component="glacier",
        ),
        "gia": GIA(
            sample_spatial=False,
        ),
    }

    local_model = Local(components=local_components, locations=locations)
    local_timeseries = local_model.run(member_seed=42)
    total_rsl = local_model.sum_components(local_model.results)

    loc_ts = local_timeseries["total_rsl"].sel(site=["Aberdeen", "Holyhead", "Llandudno", "Land point", "Durban"], percentile=[17, 50, 83])
    fig = plt.figure(figsize=(6, 4), layout="constrained")
    ax = fig.add_subplot(111)
    for loc in loc_ts.site.values:
        loc_data = loc_ts.sel(site=loc)
        ax.fill_between(loc_data.time, loc_data.sel(percentile=17), loc_data.sel(percentile=83), alpha=0.3)
        ax.plot(loc_data.time, loc_data.sel(percentile=50), label=loc)
    ax.set_xlabel("Year")
    ax.set_ylabel("Relative Sea Level (m)")
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    plt.show()

    print('end')
    pdb.set_trace()



    
if __name__ == '__main__':
    main()

