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
    #ThermalExpansion_via_zostoga
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

def main():


#    ### Global projections first ###
#    global_components = {
#        "expansion": ThermalExpansion(
#            OHC_change=np.linspace(1, 5, 295).reshape(1, -1) * 1e24
#        ),
#        "greenland_dyn": GreenlandDynAR5(),
#        "greenland_smb": GreenlandSMBAR5(),
#        "landwater": LandwaterAR5(),
#        "antarctica_dyn": AntarcticaDynAR5(),
#        "antarctica_smb": AntarcticaSMBAR5(),
#        "glacier": Glacier(),
#    }


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

    

    # Prepare CMIP5 inputs
    cmip5_dir = '/data/users/lesley.allison/project_sea_level_copy/sea_level_projections/cmip5_timeseries/'
    scenario = 'rcp26'
    with open (cmip5_dir+scenario+'_zostoga_tas_timeseries.pickle', 'rb') as f:
        data=pickle.load(f, encoding='latin1')  # encoding to get around UnicodeDecodeError due to old pickle (python2)
    zostoga_years = data['zostoga_yrs']
    tas_years = data['tas_yrs']
    tas_series = data['tas_series']
    zostoga_series = data['zostoga_series']
    models = list(data['tas_series'].keys())
    print(models)

    fig = plt.figure(figsize=(10, 4), layout="constrained")
    ax1 = fig.add_subplot(121)
    ax1.set_title('tas')
    ax1.set_xlabel('years')
    ax1.set_ylabel('tas (C)')
    for model in models:
        try:
            ax1.plot(tas_years[model], tas_series[model],color='k')
        except:
            pass
    ax2 = fig.add_subplot(122)
    ax2.set_title('zostoga')
    ax2.set_xlabel('years')
    ax2.set_ylabel('zostoga (m)')
    for model in models:
        try:
            ax2.plot(zostoga_years[model], zostoga_series[model],color='k')
        except:
            pass
    plt.show()
                 
    
    model_pick = models[6]
    single_tas_ts = tas_series[model_pick]
    single_tas_yrs = tas_years[model_pick]
    single_zostoga_ts = zostoga_series[model_pick]
    single_zostoga_yrs = zostoga_years[model_pick]

    # Temporary workaround to get tas and zostoga the same length
    # tas starts in 2006, zostoga starts in 2007
    single_tas_ts = single_tas_ts[1:]
    single_tas_yrs = single_tas_yrs[1:]

    fig = plt.figure(figsize=(10, 4), layout="constrained")
    ax1 = fig.add_subplot(121)
    ax1.plot(single_tas_yrs, single_tas_ts)
    ax2 = fig.add_subplot(122)
    ax2.plot(single_zostoga_yrs, single_zostoga_ts)

    fig.savefig(
        "/home/users/lesley.allison/sealevel/profsea/profsea-v3/figs/CMIP5_single_model_tas_zostoga.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()



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

    
    # Plot multi-model mean tas and zostoga
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 6), sharex=True)

    ds_tas26.air_temperature.plot(ax=ax1, color="tab:blue", label="RCP2.6")
    ds_tas85.air_temperature.plot(ax=ax1, color="tab:red", label="RCP8.5")
    ds_exp26.global_average_thermosteric_sea_level_change.plot(ax=ax2, color="tab:blue", label="RCP2.6")
    ds_exp85.global_average_thermosteric_sea_level_change.plot(ax=ax2, color="tab:red", label="RCP8.5")

    ax1.set_title("Global surface air temperature change")
    ax2.set_title("Global thermosteric sea-level change")

    ax1.legend()
    ax2.legend()
    
    fig.savefig(
        "/home/users/lesley.allison/sealevel/profsea/profsea-v3/figs/CMIP5_multi_model_mean_tas_zostoga.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    

    ####-------------- MME GMSL projection ----------------- ####

    ### Global projections first ###
    global_components = {
        "expansion": ThermalExpansion(ds_exp85.global_average_thermosteric_sea_level_change.values, OHC_change=False),
        "greenland_dyn": GreenlandDynAR5(),
        "greenland_smb": GreenlandSMBAR5(),
        "landwater": LandwaterAR5(),
        "antarctica_dyn": AntarcticaDynAR5(),
        "antarctica_smb": AntarcticaSMBAR5(),
        "glacier": Glacier(),
    }

    # Pass to the global model (Initialise the global model with the consituent components)
#    global_model = Global(components=global_components, end_yr=2301)  # creating an instance of the Global class
    global_model = Global(components=global_components, end_yr=2101)  # creating an instance of the Global class
    
    # Run the global model
    projections = global_model.run(
        scenario="rcp85",
        #T_change=np.linspace(1, 5, 295).reshape(1, -1),  # reshape is making a row vector
        T_change=ds_tas85.air_temperature.values,
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


    fig.savefig( 
        "/home/users/lesley.allison/sealevel/profsea/profsea-v3/figs/GMSLR_using_CMIP5_multi_model_mean_tas_zostoga_with_compspread.png",
        dpi=300,
        bbox_inches="tight"
    )
    
    plt.show()






    
    pdb.set_trace()






    ####-------------- Single member GMSL projection ----------------- ####

    ### Global projections first ###
    global_components = {
        "expansion": ThermalExpansion(single_zostoga_ts, OHC_change=False),
        "greenland_dyn": GreenlandDynAR5(),
        "greenland_smb": GreenlandSMBAR5(),
        "landwater": LandwaterAR5(),
        "antarctica_dyn": AntarcticaDynAR5(),
        "antarctica_smb": AntarcticaSMBAR5(),
        "glacier": Glacier(),
    }

    # Pass to the global model (Initialise the global model with the consituent components)
#    global_model = Global(components=global_components, end_yr=2301)  # creating an instance of the Global class
    global_model = Global(components=global_components, end_yr=2101)  # creating an instance of the Global class
    
    # Run the global model
    projections = global_model.run(
        scenario="rcp85",
        #T_change=np.linspace(1, 5, 295).reshape(1, -1),  # reshape is making a row vector
        T_change=single_tas_ts,
        member_seed=42,
    )

    # Sum the components for the global mean total
    global_model.sum_components(projections)
    gmslr = global_model.results["total_gmslr"]

    # plot all members of gmslr
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gmslr.T, color='k', alpha=0.02)  # 1000 faint lines
    ax.set_xlabel('Time')
    ax.set_ylabel('GMSLR')
    ax.set_title('All ensemble members')



    
    ### Plot example ###
    fig = plt.figure(figsize=(10, 4), layout="constrained")

    ax = fig.add_subplot(121)

    # Global projections
#    yrs = np.arange(2006, 2301)
    yrs = np.arange(2006, 2100)
    ax.set_ylim(-0.05,0.8)
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
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Global Mean Sea Level Rise (m)")


    ax2 = fig.add_subplot(122)
    yrs = np.arange(2006, 2100)
    ax2.set_ylim(-0.05,0.8)
    #ax2.plot(yrs, np.median(gmslr, axis=0), label="Global Projection", color="royalblue")
    for comp in global_model.results.keys():
        print(comp)
        ax2.plot(yrs, np.median(global_model.results[comp],axis=0), label=comp, color=fig_cols[comp])
    ax2.set_xlabel("Year", fontsize=14)
    ax2.set_ylabel("Global Mean Sea Level Rise (m)")
    ax2.legend()
    
    plt.show()



    exp_data = global_model.results['expansion'].data

    # Are all 1000 members of expansion ensemble identical?
    np.all(exp_data == exp_data[0])


    # Check shape of each component and whether all members are identical or not
    print('------')
    for comp in global_model.results.keys():
        print(comp, global_model.results[comp].data.shape , np.all(global_model.results[comp].data == global_model.results[comp].data[0]))
    print('------')

    pdb.set_trace()
        
    # arr.shape = (1000, 94)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(exp_data.T, color='k', alpha=0.02)  # 1000 faint lines
    ax.set_xlabel('Time')
    ax.set_ylabel('expansion')
    ax.set_title('All ensemble members')

    plt.show()
    
    
    pdb.set_trace()

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
            sample_spatial=False,
        ),
    }

    # Pass to the spatial model
    spatial_model = Spatial(components=spatial_components)
    spatial_model.run(member_seed=42)

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

